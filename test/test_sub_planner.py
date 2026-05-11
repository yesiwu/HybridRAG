"""测试 sub_planner 使用 search_tavily 进行搜索并回答"""
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.llm_client import LLMClient
from graph.nodes.sub_planner import sub_planner

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "sub_planner_output.json")


def build_state(task_query: str, task_id: int = 1):
    return {
        "task_query": task_query,
        "task_id": task_id,
        "messages": [],
    }


def test_sub_planner_search_and_answer():
    """sub_planner 应调用 search_tavily 搜索并返回回答"""
    client = LLMClient()
    state = build_state("2025年发展最快的几家AI公司有哪些")

    result = sub_planner(state, client.get_llm())

    # 写入文件方便查看完整内容
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"final_answer": result.get("final_answer", ""), "messages": [
                {"type": type(m).__name__, "content": m.content if hasattr(m, "content") else str(m)}
                for m in result.get("messages", [])
            ]},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 基本断言
    assert "final_answer" in result, "缺少 final_answer"
    assert len(result["final_answer"]) > 0, "final_answer 为空"
    assert result.get("need_tool_call") is False

    # 验证消息流：应包含 SystemMessage -> HumanMessage -> AIMessage(tool_call) -> ToolMessage -> AIMessage(回答)
    messages = result.get("messages", [])
    msg_types = [type(m).__name__ for m in messages]
    print(f"消息流类型: {msg_types}")

    assert "ToolMessage" in msg_types, "未检测到工具调用，search_tavily 可能未被调用"

    print(f"\n=== 最终回答 ===\n{result['final_answer'][:500].encode('gbk', errors='replace').decode('gbk')}")
