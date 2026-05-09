"""测试 main_planner 能否正确拆解任务并生成 DAG"""
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.llm_client import LLMClient
from graph.nodes.main_planner import main_planner


def build_state(query: str, summary: str = "", dag_tasks: list = None):
    return {
        "originalQuery": query,
        "conversation_summary": summary,
        "dag_tasks": dag_tasks or [],
    }


# def test_single_task_decomposition():
#     """简单查询应拆解为单个任务"""
#     client = LLMClient()
#     result = main_planner(build_state("2025年AI芯片市场规模"), client.get_llm())

#     assert "dag_tasks" in result
#     assert len(result["dag_tasks"]) >= 1
#     assert result["dag_tasks"][0]["task_id"] == 1
#     assert result["dag_tasks"][0]["depends_on"] == []


def test_multi_task_with_dependency():
    """复杂查询应拆解为多个带依赖关系的任务"""
    client = LLMClient()
    result = main_planner(
        build_state("2025年发展最快的ai公司，及他们的ai战略"),
        client.get_llm(),
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    assert len(result["dag_tasks"]) >= 2
    # 第一个任务无依赖
    assert result["dag_tasks"][0]["depends_on"] == []
    # 后续任务依赖前序任务
    assert any(len(t.get("depends_on", [])) > 0 for t in result["dag_tasks"])


# def test_skips_when_tasks_already_exist():
#     """dag_tasks 已有时应跳过拆解，返回空 dict"""
#     client = LLMClient()
#     existing = [{"task_id": 1, "query": "已有任务", "depends_on": [], "status": "pending"}]

#     result = main_planner(build_state("任意查询", dag_tasks=existing), client.get_llm())

#     assert result == {}


# def test_conversation_summary_is_updated():
#     """应将推理过程追加到 conversation_summary"""
#     client = LLMClient()
#     result = main_planner(build_state("2025年AI芯片市场规模", summary="已有摘要"), client.get_llm())

#     assert "已有摘要" in result["conversation_summary"]
#     assert len(result["conversation_summary"]) > len("已有摘要")
