"""测试 sub_planner 完整流程：搜索 -> RAG 索引检索 -> 回答"""
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.llm_client import LLMClient
from graph.nodes.sub_planner import sub_planner

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "sub_planner_output.json")


def build_state(task_query: str, task_id: int = 1, messages: list = None):
    return {
        "task_query": task_query,
        "task_id": task_id,
        "messages": messages or [],
    }


# def test_sub_planner_search_rag_answer():
#     """测试完整的 sub_planner 流程：搜索 -> RAG -> 回答"""
#     client = LLMClient()
#     state = build_state("2025年发展最快的几家AI公司有哪些")

#     result = sub_planner(state, client.get_llm())

#     # 写入文件方便查看
#     with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#         output = {
#             "final_answer": result.get("final_answer", ""),
#             "search_results_count": len(result.get("search_results", [])),
#             "retrieved_chunks_count": len(result.get("retrieved_chunks", [])),
#             "retrieved_chunks": result.get("retrieved_chunks", []),
#             "messages": [
#                 {"type": type(m).__name__, "content": m.content[:200] if hasattr(m, "content") else str(m)}
#                 for m in result.get("messages", [])
#             ],
#         }
#         json.dump(output, f, ensure_ascii=False, indent=2)

#     # 基本断言
#     assert "final_answer" in result, "缺少 final_answer"
#     assert len(result["final_answer"]) > 0, "final_answer 为空"
#     assert "retrieved_chunks" in result, "缺少 retrieved_chunks"
#     assert len(result["retrieved_chunks"]) > 0, "未召回任何文档块"
#     assert "search_results" in result, "缺少 search_results"
#     assert len(result["search_results"]) > 0, "搜索结果为空"

#     # 验证消息流
#     messages = result.get("messages", [])
#     msg_types = [type(m).__name__ for m in messages]
#     print(f"\n消息流类型: {msg_types}")

#     # 验证召回块结构
#     for chunk in result["retrieved_chunks"]:
#         assert "rank" in chunk, "召回块缺少 rank"
#         assert "text" in chunk, "召回块缺少 text"
#         assert "rerank_score" in chunk, "召回块缺少 rerank_score"

#     print(f"\n搜索结果: {len(result['search_results'])} 条")
#     print(f"召回块数: {len(result['retrieved_chunks'])} 个")
#     print(f"\n=== 最终回答 ===\n{result['final_answer'][:500].encode('gbk', errors='replace').decode('gbk')}")
#     print(f"\n完整结果已写入: {OUTPUT_FILE}")


def test_sub_planner_react_mode():
    """测试 sub_planner ReAct 模式：LLM 绑定工具，自主调用 search_tavily 和 rag_index_and_retrieve"""
    client = LLMClient()
    state = build_state("2025年发展最快的几家AI公司有哪些")

    result = sub_planner(state, client.get_llm())

    # 写入文件方便查看
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        output = {
            "final_answer": result.get("final_answer", ""),
            "search_query": result.get("search_query", ""),
            "search_results_count": len(result.get("search_results", [])),
            "retrieved_chunks_count": len(result.get("retrieved_chunks", [])),
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "messages": [
                {"type": type(m).__name__, "content": m.content[:300] if hasattr(m, "content") else str(m)}
                for m in result.get("messages", [])
            ],
        }
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 基本断言
    assert "final_answer" in result, "缺少 final_answer"
    assert len(result["final_answer"]) > 0, "final_answer 为空"
    assert "search_query" in result, "缺少 search_query"
    assert len(result["search_query"]) > 0, "search_query 为空"
    assert "retrieved_chunks" in result, "缺少 retrieved_chunks"
    assert len(result["retrieved_chunks"]) > 0, "未召回任何文档块"
    assert "search_results" in result, "缺少 search_results"
    assert len(result["search_results"]) > 0, "搜索结果为空"

    # 验证消息流类型
    messages = result.get("messages", [])
    msg_types = [type(m).__name__ for m in messages]
    print(f"\n消息流类型: {msg_types}")

    # 验证 LLM 调用了 search_tavily（RAG 由代码自动调用）
    assert "ToolMessage" in msg_types, "LLM 未调用 search_tavily"
    tool_count = msg_types.count("ToolMessage")
    print(f"工具调用次数: {tool_count} (search_tavily 由 LLM 调用, rag 由代码自动调用)")
    assert tool_count >= 1, "LLM 应至少调用 search_tavily"

    # 验证召回块结构
    for chunk in result["retrieved_chunks"]:
        assert "rank" in chunk, "召回块缺少 rank"
        assert "text" in chunk, "召回块缺少 text"
        assert "rerank_score" in chunk, "召回块缺少 rerank_score"

    print(f"\n搜索查询: {result['search_query']}")
    print(f"搜索结果: {len(result['search_results'])} 条")
    print(f"召回块数: {len(result['retrieved_chunks'])} 个")
    print(f"\n=== 最终回答 ===\n{result['final_answer'][:500].encode('gbk', errors='replace').decode('gbk')}")
    print(f"\n完整结果已写入: {OUTPUT_FILE}")
