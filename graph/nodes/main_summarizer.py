"""汇总验证节点 - 聚合 + 双重验证 + 冲突消解 + 幻觉检测

流程：
1. 收集所有子图回答和对应的检索文档
2. LCS 快速检测幻觉（阈值 30%）
3. LLM 精细验证语义忠实度
4. 多源投票仲裁（对比多个子图回答的冲突）
5. 生成最终综合回答，标记可信度
"""
import json
from difflib import SequenceMatcher
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage


# ==================== LCS 幻觉检测 ====================

def _lcs_ratio(text: str, reference: str) -> float:
    """计算 text 与 reference 的最长公共子序列比例
    用于快速判断回答是否有检索依据支撑
    """
    if not text or not reference:
        return 0.0

    # 预处理：去除标点、空格，统一小写
    def normalize(s):
        import re
        s = re.sub(r'[^\w\s]', '', s)
        s = re.sub(r'\s+', ' ', s).strip().lower()
        return s

    text_norm = normalize(text)
    ref_norm = normalize(reference)

    if not text_norm or not ref_norm:
        return 0.0

    # 使用 SequenceMatcher 计算相似度（比动态规划更快）
    matcher = SequenceMatcher(None, text_norm, ref_norm)
    return matcher.ratio()


def _detect_hallucination_lcs(answer: str, retrieval_docs: List[dict], threshold: float = 0.3) -> Dict[str, Any]:
    """LCS 快速幻觉检测
    将回答拆分为句子，检查每个句子是否有检索依据支撑
    """
    import re

    # 将回答拆分为句子
    sentences = re.split(r'[。！？\n]', answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    # 合并所有检索文档为参考文本
    all_references = " ".join([doc.get("text", "") for doc in retrieval_docs])

    if not all_references:
        return {
            "has_hallucination_risk": True,
            "risk_reason": "无检索文档支撑",
            "verified_sentences": len(sentences),
            "unverified_sentences": len(sentences),
            "unverified_details": sentences[:5],  # 最多显示5个
        }

    unverified = []
    verified = []

    for sentence in sentences:
        ratio = _lcs_ratio(sentence, all_references)
        if ratio < threshold:
            unverified.append({"sentence": sentence[:100], "lcs_ratio": round(ratio, 3)})
        else:
            verified.append({"sentence": sentence[:100], "lcs_ratio": round(ratio, 3)})

    return {
        "has_hallucination_risk": len(unverified) > len(sentences) * 0.5,  # 超过50%无依据则风险高
        "total_sentences": len(sentences),
        "verified_count": len(verified),
        "unverified_count": len(unverified),
        "unverified_details": unverified[:10],
        "risk_level": "high" if len(unverified) > len(sentences) * 0.5 else "medium" if len(unverified) > len(sentences) * 0.3 else "low",
    }


# ==================== LLM 验证 ====================

VERIFY_PROMPT = """你是一个严格的事实核查专家。

## 任务
验证以下回答是否忠实于提供的检索文档，检测可能的幻觉和不准确信息。

## 验证标准
1. **事实准确性**：回答中的数据、日期、名称是否与文档一致
2. **无编造内容**：是否添加了文档中没有的信息
3. **语义一致性**：改写是否保持了原意
4. **引用正确性**：引用的来源是否对应正确的文档

## 输出格式
```json
{
  "is_faithful": true/false,
  "confidence": 0.0-1.0,
  "hallucination_points": [
    {
      "claim": "可疑陈述",
      "reason": "问题原因",
      "severity": "high/medium/low"
    }
  ],
  "verified_claims": ["已验证正确的陈述"],
  "overall_assessment": "整体评估说明"
}
```

只输出 JSON，不要输出其他内容。"""


def _verify_with_llm(llm, answer: str, retrieval_docs: List[dict]) -> Dict[str, Any]:
    """LLM 精细验证"""
    docs_text = "\n\n".join([
        f"[文档{i+1}] {doc.get('metadata', {}).get('url', 'N/A')}\n{doc.get('text', '')[:500]}"
        for i, doc in enumerate(retrieval_docs[:5])
    ])

    messages = [
        SystemMessage(content=VERIFY_PROMPT),
        HumanMessage(content=f"## 检索文档\n{docs_text}\n\n## 待验证回答\n{answer}\n\n请验证回答的忠实度。"),
    ]

    resp = llm.invoke(messages)

    try:
        content = resp.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)
    except (json.JSONDecodeError, IndexError):
        return {
            "is_faithful": True,  # 解析失败默认通过
            "confidence": 0.5,
            "hallucination_points": [],
            "verified_claims": [],
            "overall_assessment": f"验证结果解析失败: {resp.content[:200]}",
        }


# ==================== 冲突检测 ====================

CONFLICT_DETECT_PROMPT = """你是一个多源信息冲突检测专家。

## 任务
分析以下多个子任务的回答，检测是否存在事实性冲突（如数据矛盾、结论相反等）。

## 冲突类型
1. **数据冲突**：同一指标的不同数值
2. **事实冲突**：同一事件的不同描述
3. **结论冲突**：分析结论相反或矛盾

## 输出格式
```json
{
  "has_conflict": true/false,
  "conflicts": [
    {
      "topic": "冲突主题",
      "task_ids": [1, 2],
      "conflict_type": "data/factual/conclusion",
      "description": "冲突描述",
      "details": {
        "task_1_claims": "任务1的说法",
        "task_2_claims": "任务2的说法"
      },
      "resolution": "建议的解决方案或采信哪个来源"
    }
  ],
  "consensus_points": ["多个任务一致的观点"],
  "overall_conflict_level": "none/low/medium/high"
}
```

只输出 JSON，不要输出其他内容。"""


def _detect_conflicts(llm, answers: List[dict]) -> Dict[str, Any]:
    """检测多个子图回答之间的冲突"""
    if len(answers) < 2:
        return {"has_conflict": False, "conflicts": [], "consensus_points": [], "overall_conflict_level": "none"}

    answers_text = "\n\n".join([
        f"### 任务 {ans.get('task_id')}: {ans.get('query', '')}\n{ans.get('sub_answer', '')[:800]}"
        for ans in answers
    ])

    messages = [
        SystemMessage(content=CONFLICT_DETECT_PROMPT),
        HumanMessage(content=f"## 多个子任务回答\n{answers_text}\n\n请检测冲突。"),
    ]

    resp = llm.invoke(messages)

    try:
        content = resp.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)
    except (json.JSONDecodeError, IndexError):
        return {
            "has_conflict": False,
            "conflicts": [],
            "consensus_points": [],
            "overall_conflict_level": "unknown",
            "error": f"解析失败: {resp.content[:200]}",
        }


# ==================== 冲突消解 ====================

def _resolve_conflicts(conflict_result: dict, answers: List[dict], verification_results: List[dict]) -> dict:
    """冲突消解：基于多源投票和可信度仲裁"""
    if not conflict_result.get("has_conflict"):
        return {"resolved": True, "resolutions": []}

    resolutions = []
    for conflict in conflict_result.get("conflicts", []):
        conflict_type = conflict.get("conflict_type", "unknown")
        task_ids = conflict.get("task_ids", [])
        details = conflict.get("details", {})

        # 策略1: 基于可信度选择
        task_credibility = {}
        for task_id in task_ids:
            idx = next((i for i, a in enumerate(answers) if a.get("task_id") == task_id), None)
            if idx is not None and idx < len(verification_results):
                vr = verification_results[idx]
                confidence = vr.get("confidence", 0.5)
                hallucination_count = len(vr.get("hallucination_points", []))
                task_credibility[task_id] = confidence * (1 - hallucination_count * 0.1)

        # 策略2: 基于检索文档数量（更多文档支撑 = 更可靠）
        for task_id in task_ids:
            idx = next((i for i, a in enumerate(answers) if a.get("task_id") == task_id), None)
            if idx is not None:
                doc_count = len(answers[idx].get("retrieval_docs", []))
                task_credibility[task_id] = task_credibility.get(task_id, 0.5) + doc_count * 0.05

        # 选择可信度最高的
        if task_credibility:
            best_task = max(task_credibility, key=task_credibility.get)
            resolution = {
                "conflict_topic": conflict.get("topic", ""),
                "conflict_type": conflict_type,
                "chosen_task_id": best_task,
                "reason": f"基于可信度评分 ({task_credibility[best_task]:.2f})",
                "all_scores": task_credibility,
            }
        else:
            resolution = {
                "conflict_topic": conflict.get("topic", ""),
                "conflict_type": conflict_type,
                "chosen_task_id": None,
                "reason": "无法自动消解，需用户判断",
                "all_scores": {},
            }

        resolutions.append(resolution)

    return {"resolved": True, "resolutions": resolutions}


# ==================== 最终汇总 ====================

SUMMARIZE_PROMPT = """你是一个专业的信息汇总专家。

## 任务
将多个子任务的回答整合为一个结构化的最终回答。

## 冲突处理规则
1. 如果检测到冲突，优先采信【已仲裁】中标记为"采纳"的数据
2. 对于无法自动消解的冲突，在回答中明确标注【存在争议】
3. 不要编造数据来解决冲突

## 要求
1. **保留引用**：保留原始引用标记 [1][2] 等
2. **消除冗余**：合并重复信息
3. **处理冲突**：按照上述规则处理
4. **标记可信度**：对有幻觉风险的内容标注 [待验证]
5. **结构清晰**：使用分点、分段组织

## 输出格式
直接输出最终回答，使用 Markdown 格式。"""


def _generate_final_answer(llm, query: str, answers: List[dict], verification_results: List[dict], conflict_result: dict, conflict_resolution: dict) -> str:
    """生成最终综合回答，包含冲突消解结果"""
    # 构建上下文
    context_parts = []
    for i, ans in enumerate(answers):
        task_id = ans.get("task_id", i)
        sub_answer = ans.get("sub_answer", "")
        verification = verification_results[i] if i < len(verification_results) else {}

        # 添加可信度标记
        risk_level = "low"
        if verification.get("hallucination_points"):
            risk_level = "high" if len(verification["hallucination_points"]) > 3 else "medium"

        context_parts.append(f"### 子任务 {task_id} (可信度: {risk_level})\n{sub_answer}")

    answers_text = "\n\n".join(context_parts)

    # 冲突信息 + 仲裁结果
    conflict_text = ""
    if conflict_result.get("has_conflict"):
        conflict_text = "\n\n## 检测到的冲突与仲裁结果\n"
        for i, c in enumerate(conflict_result.get("conflicts", [])):
            conflict_text += f"\n### 冲突 {i+1}: {c.get('topic', '')}\n"
            conflict_text += f"- **类型**: {c.get('conflict_type', '')}\n"
            conflict_text += f"- **描述**: {c.get('description', '')}\n"

            # 添加仲裁结果
            if i < len(conflict_resolution.get("resolutions", [])):
                res = conflict_resolution["resolutions"][i]
                if res.get("chosen_task_id"):
                    conflict_text += f"- **【已仲裁】**: 采纳任务 {res['chosen_task_id']} 的数据\n"
                    conflict_text += f"  - 原因: {res.get('reason', '')}\n"
                else:
                    conflict_text += f"- **【存在争议】**: {res.get('reason', '无法自动消解')}\n"
            else:
                conflict_text += f"- **建议**: {c.get('resolution', '')}\n"

    messages = [
        SystemMessage(content=SUMMARIZE_PROMPT),
        HumanMessage(content=f"## 原始查询\n{query}\n\n## 子任务回答\n{answers_text}{conflict_text}\n\n请生成最终综合回答。"),
    ]

    resp = llm.invoke(messages)
    return resp.content


# ==================== 主函数 ====================

def main_summarizer(state, llm) -> dict:
    """汇总验证：合并所有子图结果，双重验证，冲突消解，生成最终回答"""
    answers = state.get("agent_answers", [])
    query = state.get("originalQuery", "")

    if not answers:
        return {"final_answer": "未能生成有效回答。"}

    print(f"\n[Summarizer] 开始汇总验证，共 {len(answers)} 个子任务回答")

    # ========== Step 1: LCS 快速幻觉检测 ==========
    print("[Summarizer] Step 1: LCS 幻觉检测")
    lcs_results = []
    for ans in answers:
        sub_answer = ans.get("sub_answer", "")
        retrieval_docs = ans.get("retrieval_docs", [])
        lcs_result = _detect_hallucination_lcs(sub_answer, retrieval_docs)
        lcs_results.append(lcs_result)
        print(f"  任务 {ans.get('task_id')}: 风险等级={lcs_result.get('risk_level', 'unknown')}")

    # ========== Step 2: LLM 精细验证（仅对高风险进行） ==========
    print("[Summarizer] Step 2: LLM 精细验证")
    verification_results = []
    for i, ans in enumerate(answers):
        lcs = lcs_results[i]
        # 仅对 LCS 检测为中高风险的进行 LLM 验证，降低成本
        if lcs.get("risk_level") in ["high", "medium"]:
            print(f"  任务 {ans.get('task_id')}: 进行 LLM 验证 (风险={lcs.get('risk_level')})")
            verification = _verify_with_llm(llm, ans.get("sub_answer", ""), ans.get("retrieval_docs", []))
        else:
            print(f"  任务 {ans.get('task_id')}: LCS 通过，跳过 LLM 验证")
            verification = {"is_faithful": True, "confidence": 0.8, "hallucination_points": []}
        verification_results.append(verification)

    # ========== Step 3: 多源冲突检测 ==========
    print("[Summarizer] Step 3: 冲突检测")
    conflict_result = _detect_conflicts(llm, answers)
    if conflict_result.get("has_conflict"):
        print(f"  检测到 {len(conflict_result.get('conflicts', []))} 个冲突")
    else:
        print("  未检测到冲突")

    # ========== Step 3.5: 冲突消解（多源投票仲裁） ==========
    conflict_resolution = {"resolved": False, "resolutions": []}
    if conflict_result.get("has_conflict"):
        print("[Summarizer] Step 3.5: 冲突消解")
        conflict_resolution = _resolve_conflicts(conflict_result, answers, verification_results)
        for res in conflict_resolution.get("resolutions", []):
            if res.get("chosen_task_id"):
                print(f"  冲突 '{res.get('conflict_topic', '')}': 采纳任务 {res['chosen_task_id']}")
            else:
                print(f"  冲突 '{res.get('conflict_topic', '')}': 无法自动消解")

    # ========== Step 4: 生成最终回答 ==========
    print("[Summarizer] Step 4: 生成最终回答")
    final_answer = _generate_final_answer(llm, query, answers, verification_results, conflict_result, conflict_resolution)

    # ========== 构建验证报告 ==========
    verification_report = {
        "total_tasks": len(answers),
        "lcs_results": lcs_results,
        "llm_verification": verification_results,
        "conflict_detection": conflict_result,
        "conflict_resolution": conflict_resolution,
        "hallucination_risk_tasks": [
            ans.get("task_id") for i, ans in enumerate(answers)
            if lcs_results[i].get("risk_level") in ["high", "medium"]
        ],
    }

    print(f"[Summarizer] 汇总完成，最终回答长度: {len(final_answer)}")

    return {
        "final_answer": final_answer,
        "verification_report": verification_report,
    }
