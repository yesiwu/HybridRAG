"""汇总验证节点 - 聚合 + 双重验证 + 冲突消解"""


def main_summarizer(state, llm) -> dict:
    """汇总验证：合并所有子图结果，生成最终回答"""
    answers = state.get("agent_answers", [])
    query = state.get("originalQuery", "")

    # === 伪代码：拼接所有子答案 ===
    parts = []
    for ans in answers:
        task_id = ans.get("task_id", "?")
        sub = ans.get("sub_answer", "无结果")
        parts.append(f"[任务{task_id}] {sub}")

    final = f"针对「{query}」的综合回答：\n\n" + "\n\n".join(parts) if parts else "未能生成有效回答。"

    return {"final_answer": final}
