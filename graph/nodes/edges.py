from enum import Enum
from typing import Literal, Union, List, Dict, Any, Optional, Tuple
from langgraph.types import Send
from graph.graph_state import State, AgentState
from config.settings import settings

MAX_ITERATIONS = settings.max_iterations
MAX_TOOL_CALLS = settings.max_tool_calls

# ============================================================
# DAG 辅助函数
# ============================================================

def validate_dag_tasks(dag_tasks: List[dict]) -> Tuple[bool, str]:
    """校验 DAG 任务列表的结构合法性"""
    if not dag_tasks:
        return False, "任务列表为空"
    task_ids = {t["task_id"] for t in dag_tasks}
    for t in dag_tasks:
        for dep in t.get("depends_on", []):
            if dep not in task_ids:
                return False, f"任务 {t['task_id']} 依赖的任务 {dep} 不存在"
    return True, ""


def is_task_ready(task: dict, completed: set) -> bool:
    """判断任务是否可执行：自身未完成 + 所有依赖已完成"""
    if task["task_id"] in completed:
        return False
    return all(dep in completed for dep in task.get("depends_on", []))

# ============================================================
# 主图路由
# ============================================================

def main_router(state: State) -> Union[List[Send], Literal["main_summarizer"]]:
    """
    核心调度路由：基于任务依赖关系 (DAG) 的动态并发发射器。
    检查哪些任务的依赖已满足，通过 Send 并行派发给子图。
    所有任务完成 → 进入汇总验证。
    """
    dag_tasks = state.get("dag_tasks", [])
    agent_answers = state.get("agent_answers", [])  # 获取所有子图的回答列表
    # 先校验 DAG 结构合法性
    valid, error = validate_dag_tasks(dag_tasks)
    if not valid:
        print(f"[Edges] DAG 验证失败: {error}，跳过调度直接汇总")
        return "main_summarizer"

    completed = state.get("completed_task_ids", set())

    # 找出可立即执行的任务（依赖全部已完成，自身未完成）
    executable_tasks = [t for t in dag_tasks if is_task_ready(t, completed)]

    # 有可执行任务 → Send 扇出并发
    if executable_tasks:
        return [
            Send("atomic_agent_subgraph",
                 {
                    "task_id": task["task_id"],
                    "task_query": task["query"],
                    "depends_on": task.get("depends_on", []),
                    "dependency_context": [
                    ans for ans in agent_answers
                    if ans["task_id"] in task.get("depends_on", [])
                ],
                 }
                 )
            for task in executable_tasks
        ]

    # 全部完成或无任务 → 汇总
    return "main_summarizer"


# ============================================================
# 子图路由
# ============================================================

def route_after_reflect(state: AgentState) -> Literal["sub_compressor", "subgraph_result_sync"]:
    """子图反思后路由：
    - need_reflect=True → 压缩节点（过滤无关内容，提取关键信息）→ 重新检索
    - need_reflect=False → 同步结果到主图
    """
    if state.get("need_reflect", False):
        rewritten_query = state.get("rewritten_query", "")
        if rewritten_query:
            print(f"[Edges] 反思未通过，重写查询: '{rewritten_query}'")
        print(f"[Edges] 进入压缩节点")
        return "sub_compressor"
    print(f"[Edges] 反思通过，同步结果")
    return "subgraph_result_sync"

