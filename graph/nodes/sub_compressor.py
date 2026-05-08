"""动态上下文压缩节点 - 防爆炸、去重"""


def compress_context_node(state, llm) -> dict:
    """上下文压缩：将检索到的零散内容压缩为结构化摘要"""
    question = state.get("question", "")
    context_history = state.get("context_history", [])

    # === 伪代码：生成压缩摘要 ===
    summary = f"[伪代码] 针对「{question}」的检索上下文压缩摘要。"

    return {
        "context_summary": summary,
        "context_history": context_history + [summary],
    }
