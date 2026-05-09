"""子图规划节点 
你需要完成接收主agent发送过来的任务query还有子图查询重写，压缩后的新任务query，请设计好这样的逻辑
"""
import json
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

REACT_SYSTEM_PROMPT = """你是一个专业的实时信息检索助手。
你的工作流程是：
1. 如果用户问题需要最新信息，调用 `rag_search` 工具进行搜索。
2. 仔细阅读 `rag_search` 返回的检索片段。
3. 基于这些事实，直接给出准确、简洁的回答。

【行为准则】：
- 检索结果中包含来源 URL，回答时适当引用。
- 无法检索到信息时，如实告知，不要编造。
- 检索完成后直接回答，不再调用工具。"""

_tools_list = None

def _get_tools():
    global _tools_list
    if _tools_list is None:
        from tools.rag_search import rag_search
        _tools_list = [rag_search]
    return _tools_list

def _get_tool_map():
    return {t.name: t for t in _get_tools()}

def _run_async(coro):
    """安全地在同步上下文中运行异步协程"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)

def _execute_tool_call(tool_call: dict) -> str:
    name = tool_call["name"]
    args = tool_call["args"]
    fn = _get_tool_map().get(name)
    if fn is None:
        return json.dumps({"error": f"tool not found: {name}"}, ensure_ascii=False)
    try:
        # 正确识别 LangChain 异步工具
        if fn.coroutine is not None:
            result = _run_async(fn.ainvoke(args))
        else:
            result = fn.invoke(args)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

def sub_planner(state, llm) -> dict:
    question = state.get("question", "")
    question_index = state.get("question_index", "default_task")
    messages = list(state.get("messages", []))

    if not messages:
        messages = [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content=f"任务ID: {question_index}\n请回答：{question}"),
        ]

    tools = _get_tools()
    llm_with_tools = llm.bind_tools(tools)
    resp = llm_with_tools.invoke(messages)

    if resp.tool_calls:
        messages.append(resp)
        for tc in resp.tool_calls:
            if "task_id" not in tc["args"]:
                tc["args"]["task_id"] = str(question_index)
            tool_result_json = _execute_tool_call(tc)
            # 提取人类可读的 content（fallback 为完整 JSON）
            try:
                result_data = json.loads(tool_result_json)
                tool_content = result_data.get("content", tool_result_json)
            except json.JSONDecodeError:
                tool_content = tool_result_json
            messages.append(ToolMessage(content=tool_content, tool_call_id=tc["id"]))

        final_resp = llm.invoke(messages)
        messages.append(final_resp)
        return {
            "messages": messages,
            "final_answer": final_resp.content,
            "need_tool_call": False
        }
    else:
        messages.append(resp)
        return {
            "messages": messages,
            "final_answer": resp.content,
            "need_tool_call": False
        }