"""子图规划节点
接收主 agent 发送的任务 query，执行：搜索 -> RAG 索引检索 -> 基于召回块回答
"""
import json
from langchain_core.messages import SystemMessage, HumanMessage

REACT_SYSTEM_PROMPT = """你是一个专业的实时信息检索助手。

## 工作流程
系统已为你完成以下步骤：
1. 使用 Tavily 搜索引擎对用户问题进行了网页搜索
2. 将搜索结果切块、向量化存入 ChromaDB
3. 使用混合召回（向量相似度 + BM25 关键词）获取相关文档
4. 使用 RRF 融合排序 + Reranker 模型精排，筛选出最相关的文档片段

你将收到上述流程输出的精排文档片段（包含 rank、text、metadata、rerank_score）。

## 回答要求
- **基于事实**：严格依据检索到的文档内容回答，不要编造信息
- **引用来源**：回答时标注来源编号，如 [1]、[2]，方便用户溯源
- **结构清晰**：使用分点、分段等方式组织回答，突出关键信息
- **诚实告知**：若检索结果中没有相关信息，如实说明，不要猜测

## 输出格式
直接给出回答，无需重复说明检索流程。"""


def _get_tools():
    from tools.search import search_tavily
    from tools.rag import rag_index_and_retrieve
    return [search_tavily, rag_index_and_retrieve]


def _format_retrieved_chunks(chunks: list) -> str:
    """将召回块格式化为可读文本，供 LLM 阅读"""
    if not chunks:
        return "未检索到相关文档。"

    parts = []
    for chunk in chunks:
        rank = chunk.get("rank", "?")
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        url = metadata.get("url", "N/A")
        title = metadata.get("title", "")
        score = chunk.get("rerank_score", 0)

        part = f"[{rank}] (相关度: {score:.2f})\n"
        if title:
            part += f"标题: {title}\n"
        part += f"来源: {url}\n"
        part += f"内容: {text}\n"
        parts.append(part)

    return "\n---\n".join(parts)


def sub_planner(state, llm) -> dict:
    task_query = state.get("task_query", "")
    task_id = state.get("task_id", 0)
    messages = list(state.get("messages", []))

    # 如果已有消息（来自反思/重写迭代），直接让 LLM 回答
    if messages:
        final_resp = llm.invoke(messages)
        messages.append(final_resp)
        return {
            "messages": messages,
            "final_answer": final_resp.content,
            "retrieved_chunks": state.get("retrieved_chunks", []),
        }

    # ========== Step 1: 网页搜索 ==========
    print(f"--- [SubPlanner] 任务 {task_id}: 搜索 '{task_query}' ---")
    from tools.search import search_tavily
    search_result = search_tavily.invoke(task_query)
    search_results = search_result.get("results", [])
    print(f"[SubPlanner] 搜索返回 {len(search_results)} 条结果")

    # ========== Step 2: RAG 索引与检索 ==========
    print(f"[SubPlanner] 任务 {task_id}: RAG 索引与检索")
    from tools.rag import rag_index_and_retrieve
    rag_result = rag_index_and_retrieve.invoke({
        "task_id": str(task_id),
        "query": task_query,
        "search_results": search_result,
        "top_k": 5,
    })
    retrieved_chunks = rag_result.get("retrieved_chunks", [])
    print(f"[SubPlanner] 精排后返回 {len(retrieved_chunks)} 个召回块")

    # ========== Step 3: 基于召回块生成回答 ==========
    chunks_text = _format_retrieved_chunks(retrieved_chunks)

    messages = [
        SystemMessage(content=REACT_SYSTEM_PROMPT),
        HumanMessage(content=f"任务ID: {task_id}\n用户问题: {task_query}\n\n以下是检索到的相关文档：\n\n{chunks_text}\n\n请基于以上文档回答用户问题。"),
    ]

    print(f"[SubPlanner] 任务 {task_id}: 生成回答")
    final_resp = llm.invoke(messages)
    messages.append(final_resp)

    return {
        "messages": messages,
        "final_answer": final_resp.content,
        "search_results": search_results,
        "retrieved_chunks": retrieved_chunks,
    }
