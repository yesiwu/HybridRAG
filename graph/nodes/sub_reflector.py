"""子图反思节点 - 判断信息是否充足"""


def sub_reflector(state, llm) -> dict:
    """反思：评估当前回答质量，决定是否需要补充检索"""
    answer = state.get("final_answer", "")

    # === 伪代码：直接判定信息充足 ===
    return {
        "need_reflect": False,
        "reflection": f"[伪代码] 回答质量评估：信息充足，无需补充检索。",
    }
