"""子图规划节点
流程：LLM 调用 search_tavily -> 代码自动调用 rag_index_and_retrieve -> LLM 基于召回块回答
"""
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

REACT_SYSTEM_PROMPT = """你是一个专业的实时信息检索助手。

## 工作流程
1. 调用 `search_tavily` 搜索用户问题，获取网页内容
2. 系统会自动将搜索结果进行 RAG 处理（切块、向量化、混合召回、精排）
3. 你将收到精排后的文档片段，基于这些事实给出回答

## 回答要求
- **基于事实**：严格依据检索到的文档内容回答，不要编造信息
- **引用来源**：回答时标注来源编号，如 [1]、[2]，方便用户溯源
- **结构清晰**：使用分点、分段等方式组织回答，突出关键信息
- **诚实告知**：若检索结果中没有相关信息，如实说明

首先调用 `search_tavily` 工具进行搜索。"""


def _get_tools():
    from tools.search import search_tavily
    return [search_tavily]


def _get_tool_map():
    return {t.name: t for t in _get_tools()}


def _execute_tool_call(tool_call: dict) -> str:
    """执行工具调用并返回 JSON 字符串"""
    name = tool_call["name"]
    args = tool_call["args"]
    fn = _get_tool_map().get(name)
    if fn is None:
        return json.dumps({"error": f"tool not found: {name}"}, ensure_ascii=False)
    try:
        result = fn.invoke(args)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _format_retrieved_chunks(chunks: list) -> str:
    """将召回块格式化为可读文本"""
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

    # ========== Step 1: LLM 调用 search_tavily ==========
    print(f"[SubPlanner] 任务 {task_id}: 搜索 '{task_query}'")

    init_messages = [
        SystemMessage(content=REACT_SYSTEM_PROMPT),
        HumanMessage(content=f"任务ID: {task_id}\n请回答：{task_query}"),
    ]

    tools = _get_tools()
    llm_with_tools = llm.bind_tools(tools)
    resp = llm_with_tools.invoke(init_messages)

    search_results = []
    search_query = task_query  # 默认使用原始查询
    if resp.tool_calls:
        init_messages.append(resp)
        for tc in resp.tool_calls:
            print(f"[SubPlanner] 调用工具: {tc['name']}")
            tool_result_json = _execute_tool_call(tc)
            tool_result = json.loads(tool_result_json)

            if tc["name"] == "search_tavily":
                search_query = tc["args"].get("query", task_query)  # 记录实际搜索词
                search_results = tool_result.get("results", [])
                print(f"[SubPlanner] 搜索返回 {len(search_results)} 条结果")

            init_messages.append(ToolMessage(
                content=json.dumps(tool_result, ensure_ascii=False, default=str),
                tool_call_id=tc["id"]
            ))
    else:
        # LLM 没有调用工具，直接返回
        init_messages.append(resp)
        return {
            "messages": init_messages,
            "final_answer": resp.content,
            "search_results": [],
            "retrieved_chunks": [],
        }

    # ========== Step 2: 自动调用 RAG 索引与检索 ==========
    print(f"[SubPlanner] 任务 {task_id}: RAG 索引与检索")
    from tools.rag import rag_index_and_retrieve

    search_result = {"query": task_query, "results": search_results}
    rag_result = rag_index_and_retrieve.invoke({
        "task_id": str(task_id),
        "query": task_query,
        "search_results": search_result,
        "top_k": 5,
    })
    retrieved_chunks = rag_result.get("retrieved_chunks", [])
    print(f"[SubPlanner] 精排返回 {len(retrieved_chunks)} 个召回块")

    # ========== Step 3: LLM 基于召回块生成回答 ==========
    chunks_text = _format_retrieved_chunks(retrieved_chunks)

    # 将 RAG 结果作为补充信息加入消息流
    init_messages.append(HumanMessage(
        content=f"以下是经过 RAG 检索精排后的文档片段：\n\n{chunks_text}\n\n请基于以上文档回答用户问题，引用来源编号如 [1]、[2]。"
    ))

    print(f"[SubPlanner] 任务 {task_id}: 生成回答")
    final_resp = llm.invoke(init_messages)
    init_messages.append(final_resp)

    return {
        "messages": init_messages,
        "final_answer": final_resp.content,
        "search_query": search_query,
        "search_results": search_results,
        "retrieved_chunks": retrieved_chunks,
    }
