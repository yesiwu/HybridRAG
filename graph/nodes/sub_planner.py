"""子图规划节点
流程：LLM 调用 search_tavily -> 代码自动调用 rag_index_and_retrieve -> LLM 基于召回块回答

支持多轮迭代：
1. 首次调用：接收 task_query 进行搜索和回答
2. 迭代调用：根据 rewritten_query 进行补充检索，合并历史召回块
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
    rewritten_query = state.get("rewritten_query", "")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    crawled_urls = state.get("crawled_urls", [])
    retrieval_history = state.get("retrieval_history", [])
    answer_history = state.get("answer_history", [])
    compressed_context = state.get("compressed_context", "")
    filtered_chunks = state.get("filtered_chunks", [])
    prev_final_answer = state.get("final_answer", "")

    # ========== 迭代次数限制检查 ==========
    if iteration_count >= max_iterations:
        print(f"\n[SubPlanner] 任务 {task_id}: 已达到最大迭代次数 {max_iterations}，跳过执行")
        # 返回当前状态，不执行新的搜索
        return {
            "messages": state.get("messages", []),
            "final_answer": prev_final_answer,  # 保留上一轮的回答
            "search_query": state.get("search_query", ""),
            "search_results": state.get("search_results", []),
            "retrieved_chunks": state.get("retrieved_chunks", []),
            "crawled_urls": crawled_urls,
            "retrieval_history": retrieval_history,
            "answer_history": answer_history,
        }

    # ========== 判断调用场景 ==========
    search_query = rewritten_query if rewritten_query else task_query
    is_iteration = iteration_count > 0

    if is_iteration:
        print(f"\n[SubPlanner] 任务 {task_id}: 迭代调用 (第{iteration_count}轮/{max_iterations}轮)")
        print(f"[SubPlanner] 使用重写查询: '{search_query}'")
    else:
        print(f"\n[SubPlanner] 任务 {task_id}: 首次调用")
        print(f"[SubPlanner] 使用原始查询: '{search_query}'")

    # ========== Step 1: LLM 调用 search_tavily ==========
    init_messages = [
        SystemMessage(content=REACT_SYSTEM_PROMPT),
        HumanMessage(content=f"任务ID: {task_id}\n请回答：{search_query}"),
    ]

    tools = _get_tools()
    llm_with_tools = llm.bind_tools(tools)
    resp = llm_with_tools.invoke(init_messages)

    search_results = []
    actual_search_query = search_query

    if resp.tool_calls:
        init_messages.append(resp)
        for tc in resp.tool_calls:
            print(f"[SubPlanner] 调用工具: {tc['name']}")
            tool_result_json = _execute_tool_call(tc)
            tool_result = json.loads(tool_result_json)

            if tc["name"] == "search_tavily":
                actual_search_query = tc["args"].get("query", search_query)
                search_results = tool_result.get("results", [])
                print(f"[SubPlanner] 搜索返回 {len(search_results)} 条结果")

            init_messages.append(ToolMessage(
                content=json.dumps(tool_result, ensure_ascii=False, default=str),
                tool_call_id=tc["id"]
            ))
    else:
        init_messages.append(resp)
        return {
            "messages": init_messages,
            "final_answer": resp.content,
            "search_query": search_query,
            "search_results": [],
            "retrieved_chunks": [],
        }

    # ========== Step 2: RAG 索引与检索 ==========
    print(f"[SubPlanner] 任务 {task_id}: RAG 索引与检索")
    from tools.rag import rag_index_and_retrieve

    # 过滤掉已爬取的 URL
    new_results = [r for r in search_results if r.get("url") not in crawled_urls]
    if not new_results:
        print(f"[SubPlanner] 所有 URL 已检索过，使用全部结果")
        new_results = search_results

    search_result = {"query": actual_search_query, "results": new_results}
    rag_result = rag_index_and_retrieve.invoke({
        "task_id": str(task_id),
        "query": actual_search_query,
        "search_results": search_result,
        "top_k": 5,
    })
    retrieved_chunks = rag_result.get("retrieved_chunks", [])
    print(f"[SubPlanner] 精排返回 {len(retrieved_chunks)} 个召回块")

    # 更新已爬取 URL
    new_urls = [r.get("url", "") for r in search_results if r.get("url")]
    updated_crawled_urls = list(set(crawled_urls + new_urls))

    # ========== Step 3: 合并历史召回块 ==========
    # 将过滤后的高质量块与当前召回块合并
    if filtered_chunks:
        all_chunks = filtered_chunks + retrieved_chunks
        seen_texts = set()
        unique_chunks = []
        for chunk in sorted(all_chunks, key=lambda x: x.get("rerank_score", 0), reverse=True):
            text_prefix = chunk.get("text", "")[:100]
            if text_prefix not in seen_texts:
                seen_texts.add(text_prefix)
                unique_chunks.append(chunk)
        final_chunks = unique_chunks[:10]
    else:
        final_chunks = retrieved_chunks

    # ========== Step 4: LLM 基于召回块生成回答 ==========
    chunks_text = _format_retrieved_chunks(final_chunks)

    # 构建上下文提示
    context_parts = []
    if compressed_context:
        context_parts.append(f"## 历史检索关键信息\n{compressed_context}")
    if answer_history:
        context_parts.append("## 历史回答摘要")
        for hist in answer_history[-2:]:  # 只取最近 2 轮
            context_parts.append(f"- 第{hist['iteration']}轮: {hist.get('key_points', hist['answer'][:200])}")

    context_hint = "\n".join(context_parts) if context_parts else ""
    if is_iteration:
        context_hint += f"\n\n这是第 {iteration_count} 轮迭代检索，请综合所有信息给出更完善的回答。"

    init_messages.append(HumanMessage(
        content=f"以下是经过 RAG 检索精排后的文档片段：\n\n{chunks_text}\n\n{context_hint}\n\n请基于以上文档回答用户问题，引用来源编号如 [1]、[2]。"
    ))

    print(f"[SubPlanner] 任务 {task_id}: 生成回答 (调用 LLM)")
    import sys
    sys.stdout.flush()
    final_resp = llm.invoke(init_messages)
    print(f"[SubPlanner] 任务 {task_id}: LLM 调用完成")
    sys.stdout.flush()
    init_messages.append(final_resp)

    # ========== 保存历史记录 ==========
    new_retrieval_history = retrieval_history + [{
        "iteration": iteration_count,
        "query": actual_search_query,
        "chunks": retrieved_chunks,
    }]

    new_answer_history = answer_history + [{
        "iteration": iteration_count,
        "query": actual_search_query,
        "answer": final_resp.content,
        "key_points": final_resp.content[:300],  # 关键内容提取（后续压缩节点会优化）
    }]

    return {
        "messages": init_messages,
        "final_answer": final_resp.content,
        "search_query": actual_search_query,
        "search_results": search_results,
        "retrieved_chunks": retrieved_chunks,
        "crawled_urls": updated_crawled_urls,
        "retrieval_history": new_retrieval_history,
        "answer_history": new_answer_history,
    }
