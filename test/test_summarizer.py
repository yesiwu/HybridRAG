"""测试 main_summarizer 汇总验证流程"""
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.llm_client import LLMClient
from graph.nodes.main_summarizer import (
    main_summarizer,
    _detect_hallucination_lcs,
    _lcs_ratio,
)

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "summarizer_output.json")


# ==================== LCS 测试 ====================

def test_lcs_ratio():
    """测试 LCS 相似度计算"""
    # 完全相同
    assert _lcs_ratio("hello world", "hello world") > 0.9

    # 部分匹配
    ratio = _lcs_ratio("2025年AI芯片市场规模达到160亿美元", "根据报告，2025年AI芯片市场规模约为160亿美元")
    print(f"部分匹配 LCS: {ratio:.3f}")
    assert ratio > 0.3  # 应该超过阈值

    # 无关联
    ratio = _lcs_ratio("今天天气真好", "2025年AI芯片市场规模达到160亿美元")
    print(f"无关联 LCS: {ratio:.3f}")
    assert ratio < 0.3  # 应该低于阈值


def test_hallucination_detection():
    """测试幻觉检测"""
    answer = """根据检索结果，2025年AI芯片市场规模达到160亿美元。
NVIDIA占据全球75%的算力份额。
中国智能算力规模约占全球15%。
未来三年市场将继续保持高速增长。"""

    retrieval_docs = [
        {"text": "2025年上半年，中国AI芯片市场规模达到160亿美元，出货量超过190万张"},
        {"text": "美国智能算力规模约占全球的75%，中国约占15%"},
    ]

    result = _detect_hallucination_lcs(answer, retrieval_docs, threshold=0.3)

    print(f"\n幻觉检测结果:")
    print(f"  风险等级: {result.get('risk_level')}")
    print(f"  已验证: {result.get('verified_count')}")
    print(f"  未验证: {result.get('unverified_count')}")

    assert "risk_level" in result
    assert result["total_sentences"] > 0


# ==================== 完整汇总测试 ====================

def test_summarizer_with_multiple_answers():
    """测试多个子图回答的汇总验证"""
    client = LLMClient()
    llm = client.get_llm()

    # 模拟多个子图的回答
    agent_answers = [
        {
            "task_id": 1,
            "query": "2025年全球AI芯片市场规模和主要厂商",
            "sub_answer": """根据最新数据，2025年全球AI芯片市场规模约为961.9亿美元。

主要厂商包括：
1. NVIDIA - 占据全球约75%的算力份额
2. AMD - 市场份额约15%
3. Intel - 份额约5%

中国厂商方面：
- 华为昇腾 - 出货量领跑国产AI芯片
- 寒武纪 - 2025上半年营收28.8亿元，同比增长43倍
- 摩尔线程 - 价值3100亿元""",
            "retrieval_docs": [
                {"text": "2025年全球AI芯片市场规模约为961.9亿美元，北美地区以36.98%的份额主导全球市场", "metadata": {"url": "http://example.com/1"}},
                {"text": "美国智能算力规模约占全球的75%，NVIDIA占据绝对主流", "metadata": {"url": "http://example.com/2"}},
                {"text": "华为昇腾出货量领跑，寒武纪2025上半年营收28.8亿元，同比增长43倍", "metadata": {"url": "http://example.com/3"}},
            ],
        },
        {
            "task_id": 2,
            "query": "中国AI芯片市场份额和国产替代进展",
            "sub_answer": """2025年中国AI芯片市场出现重要拐点：

- 国产GPU与AI芯片厂商市场份额首次攀升至41%
- 英伟达在中国市场份额从95%下滑至55%

第一梯队厂商：
1. 华为昇腾
2. 阿里平头哥
3. 百度昆仑芯
4. 寒武纪

技术路线方面，国产芯片在中端芯片制造和封装测试环节形成优势。""",
            "retrieval_docs": [
                {"text": "2025年国产GPU与AI芯片厂商市场份额首次攀升至41%，英伟达在中国市场份额从95%下滑至55%", "metadata": {"url": "http://example.com/4"}},
                {"text": "第一梯队包括华为昇腾、阿里平头哥、百度昆仑芯、寒武纪", "metadata": {"url": "http://example.com/5"}},
            ],
        },
    ]

    state = {
        "originalQuery": "2025年全球和中国AI芯片市场分析",
        "agent_answers": agent_answers,
    }

    print("=" * 60)
    print("测试 main_summarizer 汇总验证")
    print("=" * 60)

    result = main_summarizer(state, llm)

    # 保存结果
    output = {
        "original_query": state["originalQuery"],
        "final_answer": result.get("final_answer", ""),
        "verification_report": result.get("verification_report", {}),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 断言
    assert "final_answer" in result
    assert len(result["final_answer"]) > 0
    assert "verification_report" in result

    report = result.get("verification_report", {})
    print(f"\n验证报告:")
    print(f"  子任务数: {report.get('total_tasks')}")
    print(f"  幻觉风险任务: {report.get('hallucination_risk_tasks')}")
    print(f"  冲突检测: {report.get('conflict_detection', {}).get('has_conflict')}")

    print(f"\n最终回答长度: {len(result.get('final_answer', ''))}")
    print(f"结果已写入: {OUTPUT_FILE}")


def test_summarizer_with_conflict():
    """测试包含冲突的子图回答汇总"""
    client = LLMClient()
    llm = client.get_llm()

    # 模拟有冲突的子图回答
    agent_answers = [
        {
            "task_id": 1,
            "query": "寒武纪2025年营收",
            "sub_answer": "寒武纪2025年上半年营收28.8亿元，同比增长43倍。",
            "retrieval_docs": [
                {"text": "寒武纪2025上半年营收28.8亿元，同比增长43倍", "metadata": {"url": "http://source1.com"}},
            ],
        },
        {
            "task_id": 2,
            "query": "寒武纪财务数据",
            "sub_answer": "寒武纪2025年全年营收预计达到50亿元，上半年约20亿元。",
            "retrieval_docs": [
                {"text": "寒武纪2025年全年营收预计50亿元", "metadata": {"url": "http://source2.com"}},
            ],
        },
    ]

    state = {
        "originalQuery": "寒武纪2025年财务数据",
        "agent_answers": agent_answers,
    }

    print("\n" + "=" * 60)
    print("测试冲突检测与消解")
    print("=" * 60)

    result = main_summarizer(state, llm)

    # 保存冲突测试输出
    conflict_output_file = os.path.join(os.path.dirname(__file__), "conflict_resolution_output.json")
    with open(conflict_output_file, "w", encoding="utf-8") as f:
        json.dump({
            "original_query": state["originalQuery"],
            "final_answer": result.get("final_answer", ""),
            "verification_report": result.get("verification_report", {}),
        }, f, ensure_ascii=False, indent=2)

    report = result.get("verification_report", {})
    conflict = report.get("conflict_detection", {})
    resolution = report.get("conflict_resolution", {})

    print(f"\n冲突检测结果:")
    print(f"  has_conflict: {conflict.get('has_conflict')}")
    print(f"  conflict_level: {conflict.get('overall_conflict_level')}")

    if conflict.get("has_conflict"):
        for c in conflict.get("conflicts", []):
            print(f"  冲突: {c.get('topic')} - {c.get('description')[:100]}")

    print(f"\n冲突消解结果:")
    for res in resolution.get("resolutions", []):
        print(f"  主题: {res.get('conflict_topic')}")
        print(f"  采纳任务: {res.get('chosen_task_id')}")
        print(f"  原因: {res.get('reason')}")

    print(f"\n最终回答:\n{result.get('final_answer', '')[:500]}")
    print(f"\n结果已写入: {conflict_output_file}")


if __name__ == "__main__":
    test_lcs_ratio()
    test_hallucination_detection()
