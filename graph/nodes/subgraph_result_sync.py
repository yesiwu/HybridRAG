
def subgraph_result_sync(state: AgentState) -> dict:
    """
    子图结果同步节点：构造结果并同步到主图 State。
    作为图节点使用，返回 dict → LangGraph 自动合并到主图状态。
    """
    result = {
        "task_id": state["task_id"],
        "query": state["query"],
        "retrieval_docs": list(state.get("retrieval_keys", set())),
        "sub_answer": state.get("final_answer", ""),
    }

    return {
        "agent_answers": [result],
        "completed_task_ids": {state["task_id"]},
    }
