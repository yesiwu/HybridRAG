"""测试 main_router 的 DAG 调度：分轮派发带依赖的任务给子图"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langgraph.types import Send
from graph.nodes.edges import main_router


DAG_TASKS = [
    {"task_id": 1, "query": "确定2025年发展最快的AI公司", "depends_on": [], "status": "pending"},
    {"task_id": 2, "query": "分析这些公司的AI战略", "depends_on": [1], "status": "pending"},
]


def build_state(dag_tasks, completed=None, agent_answers=None):
    return {
        "dag_tasks": dag_tasks,
        "completed_task_ids": completed or set(),
        "agent_answers": agent_answers or [],
    }


def test_first_round_dispatches_independent_task():
    """第一轮：task 1 无依赖应被派发，task 2 依赖 task 1 不应被派发"""
    result = main_router(build_state(DAG_TASKS, completed=set()))

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Send)
    assert result[0].arg["task_id"] == 1
    assert result[0].arg["query"] == "确定2025年发展最快的AI公司"
    assert result[0].arg["depends_on"] == []


def test_second_round_dispatches_dependent_task():
    """第二轮：task 1 已完成，task 2 的依赖满足，应被派发"""
    task1_answer = {
        "task_id": 1,
        "query": "确定2025年发展最快的AI公司",
        "retrieval_docs": [],
        "sub_answer": "DeepSeek, OpenAI, Anthropic...",
    }
    result = main_router(build_state(
        DAG_TASKS,
        completed={1},
        agent_answers=[task1_answer],
    ))

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].arg["task_id"] == 2
    assert result[0].arg["depends_on"] == [1]
    # 应携带 task 1 的结果作为上下文
    assert len(result[0].arg["dependency_context"]) == 1
    assert result[0].arg["dependency_context"][0]["task_id"] == 1


def test_third_round_goes_to_summarizer():
    """第三轮：所有任务完成，应路由到 main_summarizer"""
    result = main_router(build_state(DAG_TASKS, completed={1, 2}))

    assert result == "main_summarizer"


def test_parallel_dispatch():
    """两个无依赖的任务应同时被派发"""
    parallel_tasks = [
        {"task_id": 1, "query": "任务A", "depends_on": [], "status": "pending"},
        {"task_id": 2, "query": "任务B", "depends_on": [], "status": "pending"},
        {"task_id": 3, "query": "任务C", "depends_on": [1, 2], "status": "pending"},
    ]
    result = main_router(build_state(parallel_tasks, completed=set()))

    assert isinstance(result, list)
    assert len(result) == 2
    dispatched_ids = {r.arg["task_id"] for r in result}
    assert dispatched_ids == {1, 2}
