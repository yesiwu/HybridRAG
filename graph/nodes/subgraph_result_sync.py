"""子图结果同步节点 - 将子图执行结果同步到主图 State"""
from graph.graph_state import AgentState


def subgraph_result_sync(state: AgentState) -> dict:
    """
    子图结果同步节点：构造结果并同步到主图 State。
    作为图节点使用，返回 dict → LangGraph 自动合并到主图状态。
    """
    task_id = state.get("task_id", 0)
    task_query = state.get("task_query", "")
    final_answer = state.get("final_answer", "")

    # 获取检索文档：优先使用过滤后的高质量块，否则使用最新召回块
    filtered_chunks = state.get("filtered_chunks", [])
    retrieved_chunks = state.get("retrieved_chunks", [])
    retrieval_docs = filtered_chunks if filtered_chunks else retrieved_chunks

    # 构造 agent_answers 需要的格式
    # 结构: {task_id: int, query: str, retrieval_docs: List[dict], sub_answer: str}
    result = {
        "task_id": task_id,
        "query": task_query,
        "retrieval_docs": retrieval_docs,
        "sub_answer": final_answer,
        "iteration_count": state.get("iteration_count", 0),
        "compressed_context": state.get("compressed_context", ""),
    }

    return {
        "agent_answers": [result],
        "completed_task_ids": {task_id},
    }
