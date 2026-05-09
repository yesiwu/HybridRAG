"""动态上下文压缩节点 - 防爆炸、去重

这个子图反思过长容易导致长程上下文问题，所以每次反思之后我都会对回答长度进行一个判断，
是否进入上下文压缩节点，进行一个上下文压缩（子state的message包括模型回答和上下文片段还有工具调用过程。
压缩输出一个结构化的摘要（摘要就是根据反思结果来进行一个总结，总结有用内容，因为上一次的检索可能有误或者残缺，
另外过滤本次的工具调用过程）并**记住"已检索过什么网页url"**，让 后续查询重写**避免重复搜索**），
提取核心内容，压缩节点还会让llm判断哪些网页不太相关，进行一个去除，保留高质量的召回块。还有限制工具调用次数。
"""



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
