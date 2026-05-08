"""子图查询重写节点 - 优化检索策略"""


def rewrite_query_node(state, llm) -> dict:
    """查询重写：基于反思结果优化搜索词"""
    question = state.get("question", "")
    reflection = state.get("reflection", "")

    # === 伪代码：重写查询后重新触发规划节点 ===
    rewritten = f"{question}（优化检索）"

    return {
        "question": rewritten,
        "need_tool_call": True,
    }
