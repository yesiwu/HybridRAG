from enum import Enum
from typing import Literal, Union, List, Dict, Any, Optional, Tuple
from langgraph.types import Send
from state.graph_state import State, AgentState
from config import MAX_ITERATIONS, MAX_TOOL_CALLS


""""
全部完成。以下是改动总结：                                                                                                                                                           
                                                                                                                                                                                       
  graph/nodes/edges.py 新增内容                                                                                                                                                      
                                                                                                                                                                                       
  ┌──────────────────────────────┬───────────────────────────────────────────────────────────────┐                                                                                     
  │             组件             │                             说明                              │                                                                                     
  ├──────────────────────────────┼───────────────────────────────────────────────────────────────┤                                                                                     
  │ TaskStatus                   │ 任务生命周期枚举（PENDING/READY/RUNNING/COMPLETED/FAILED）    │                                                                                     
  ├──────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ is_task_ready()              │ 判断单个任务是否满足执行条件（未完成 + 依赖全部完成）         │
  ├──────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ build_send_args()            │ 为任务构建子图参数（查询、ID、依赖上下文、消息历史）          │
  ├──────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ validate_dag_tasks()         │ 验证 DAG 结构完整性（字段检查 + 依赖引用 + 循环检测拓扑排序） │
  ├──────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ launch_subgraphs()           │ 扫描 DAG，找出所有就绪任务并创建 Send 对象并行启动子图        │
  ├──────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ dispatch_to_subgraph()       │ 验证 + 调度一体化入口（非法 DAG 直接跳转汇总）                │
  ├──────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ aggregate_subgraph_results() │ 按 task_id 排序聚合作答结果                                   │
  ├──────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ main_router()                │ 重构为复用 is_task_ready + build_send_args                    │
  └──────────────────────────────┴───────────────────────────────────────────────────────────────┘

  test_edges.py — 59 个测试用例，全部通过

  - TaskStatus 枚举：值正确性、类型兼容
  - is_task_ready：无依赖、依赖满足/未满足、部分依赖、已完成、链式依赖
  ├──────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ launch_subgraphs()           │ 扫描 DAG，找出所有就绪任务并创建 Send 对象并行启动子图        │
  ├──────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ dispatch_to_subgraph()       │ 验证 + 调度一体化入口（非法 DAG 直接跳转汇总）                │
  ├──────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ aggregate_subgraph_results() │ 按 task_id 排序聚合作答结果                                   │
  ├──────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ main_router()                │ 重构为复用 is_task_ready + build_send_args                    │
  └──────────────────────────────┴───────────────────────────────────────────────────────────────┘

  test_edges.py — 59 个测试用例，全部通过

  - TaskStatus 枚举：值正确性、类型兼容
  - is_task_ready：无依赖、依赖满足/未满足、部分依赖、已完成、链式依赖
  - build_send_args：基础构建、依赖上下文注入、无关答案过滤、消息传递
  - validate_dag_tasks：空列表、单任务、链式/并行 DAG、缺少字段、非法 ID/depends_on、不存在依赖、循环依赖
  - launch_subgraphs：空任务、单任务、并行启动、依赖约束、跳过已完成、依赖解锁、多依赖、上下文注入
  - dispatch_to_subgraph：合法 DAG、全部完成、空任务、非法 DAG、循环 DAG
  - main_router：单根、多根、部分完成、全部完成、无任务
  - route_after_reflect / subgraph_final_router：路由逻辑、结果同步字段
  - aggregate_subgraph_results：空/单/多结果、排序、内容保留、缺省字段
  - 端到端：菱形 DAG 四层调度、子图结果往返、派发+聚合完整流程


"""

# ============================================================
# 任务状态枚举
# ============================================================

class TaskStatus(str, Enum):
    """任务生命周期状态"""
    PENDING = "pending"       # 等待依赖完成
    READY = "ready"           # 依赖已满足，可被调度
    RUNNING = "running"       # 子图执行中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 执行失败


# ============================================================
# 子图启动与任务派发核心函数
# ============================================================

def is_task_ready(task: dict, completed_ids: set) -> bool:
    """
    判断单个任务是否满足执行条件：
    1. 任务自身未完成
    2. 所有依赖任务均已完成
    """
    if task["id"] in completed_ids:
        return False
    dependencies = task.get("depends_on", [])
    return all(dep_id in completed_ids for dep_id in dependencies)


def build_send_args(task: dict, state: State) -> dict:
    """
    为单个任务构建发送给子图的参数字典。
    包含任务查询、任务ID、依赖上下文和消息历史。
    """
    return {
        "question": task["query"],
        "question_index": task["id"],
        "dependency_context": [
            ans for ans in state.get("agent_answers", [])
            if ans.get("task_id") in task.get("depends_on", [])
        ],
        "messages": state.get("messages", []),
    }


def validate_dag_tasks(dag_tasks: List[dict]) -> Tuple[bool, Optional[str]]:
    """
    验证 DAG 任务列表的结构完整性：
    1. 每个任务必须包含 id, query, depends_on 字段
    2. depends_on 引用的 id 必须在任务列表中存在
    3. 不能存在循环依赖（拓扑排序检测）
    返回 (is_valid, error_message)。
    """
    if not dag_tasks:
        return True, None

    task_ids = set()
    for i, task in enumerate(dag_tasks):
        for field in ("id", "query", "depends_on"):
            if field not in task:
                return False, f"任务索引 {i} 缺少必要字段 '{field}'"
        if not isinstance(task["id"], int):
            return False, f"任务索引 {i} 的 id 必须是整数"
        if not isinstance(task["depends_on"], list):
            return False, f"任务索引 {i} 的 depends_on 必须是列表"
        task_ids.add(task["id"])

    for task in dag_tasks:
        for dep_id in task["depends_on"]:
            if dep_id not in task_ids:
                return False, f"任务 {task['id']} 的 depends_on 引用了不存在的任务 {dep_id}"

    # 拓扑排序检测循环依赖
    in_degree = {t["id"]: 0 for t in dag_tasks}
    adjacency: Dict[int, List[int]] = {t["id"]: [] for t in dag_tasks}
    for task in dag_tasks:
        for dep_id in task["depends_on"]:
            adjacency[dep_id].append(task["id"])
            in_degree[task["id"]] += 1

    queue = [tid for tid, deg in in_degree.items() if deg == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited != len(task_ids):
        return False, "DAG 中存在循环依赖"

    return True, None


def launch_subgraphs(state: State) -> List[Send]:
    """
    扫描 DAG 任务列表，找出所有依赖已满足的待执行任务，
    为每个任务创建 Send 对象以并行启动子图。
    """
    completed = state.get("completed_task_ids", set())
    dag_tasks = state.get("dag_tasks", [])

    executable = [t for t in dag_tasks if is_task_ready(t, completed)]
    return [
        Send("atomic_agent", build_send_args(task, state))
        for task in executable
    ]


def dispatch_to_subgraph(state: State) -> Union[List[Send], str]:
    """
    验证 + 调度一体化入口：
    1. 校验 DAG 结构合法性（非法则记录并跳过调度）
    2. 筛选可执行任务并扇出到子图
    3. 无可执行任务时返回汇总节点名
    """
    dag_tasks = state.get("dag_tasks", [])
    valid, error = validate_dag_tasks(dag_tasks)
    if not valid:
        print(f"[Edges] DAG 验证失败: {error}，跳过调度直接汇总")
        return "collect_verify"

    sends = launch_subgraphs(state)
    if sends:
        return sends
    return "collect_verify"


# ============================================================
# 主图路由
# ============================================================

def main_router(state: State) -> Union[List[Send], Literal["collect_verify"]]:
    """
    核心调度路由：基于任务依赖关系 (DAG) 的动态并发发射器。
    检查哪些任务的依赖已满足，通过 Send 并行派发给子图。
    所有任务完成 → 进入汇总验证。
    """
    dag_tasks = state.get("dag_tasks", [])

    # 先校验 DAG 结构合法性
    valid, error = validate_dag_tasks(dag_tasks)
    if not valid:
        print(f"[Edges] DAG 验证失败: {error}，跳过调度直接汇总")
        return "collect_verify"

    completed = state.get("completed_task_ids", set())

    # 找出可立即执行的任务（依赖全部已完成，自身未完成）
    executable_tasks = [t for t in dag_tasks if is_task_ready(t, completed)]

    # 有可执行任务 → Send 扇出并发
    if executable_tasks:
        return [
            Send("atomic_agent", build_send_args(task, state))
            for task in executable_tasks
        ]

    # 全部完成或无任务 → 汇总
    return "collect_verify"


# ============================================================
# 子图路由
# ============================================================

def route_after_reflect(state: AgentState) -> Literal["compress_context_node", "subgraph_final_router"]:
    """子图反思后路由：信息不足 → 压缩重写；信息充足 → 同步结果到主图"""
    if state.get("need_reflect", False):
        return "compress_context_node"
    return "subgraph_final_router"


def subgraph_final_router(state: AgentState) -> dict:
    """
    子图结果同步节点（非路由函数）：构造结果并同步到主图 State。
    作为图节点使用，返回 dict → LangGraph 自动合并到主图状态。
    """
    result = {
        "task_id": state["question_index"],
        "query": state["question"],
        "retrieval_docs": list(state.get("retrieval_keys", set())),
        "sub_answer": state.get("final_answer", ""),
    }

    return {
        "agent_answers": [result],
        "completed_task_ids": {state["question_index"]},
    }


# ============================================================
# 结果聚合
# ============================================================

def aggregate_subgraph_results(agent_answers: List[dict]) -> str:
    """
    将多个子图返回的结果聚合成统一文本。
    按 task_id 排序后拼接，每个结果标注来源任务。
    """
    if not agent_answers:
        return ""

    sorted_answers = sorted(agent_answers, key=lambda x: x.get("task_id", 0))
    parts = []
    for ans in sorted_answers:
        task_id = ans.get("task_id", "?")
        query = ans.get("query", "")
        sub_answer = ans.get("sub_answer", "")
        parts.append(f"[任务 {task_id}] {query}\n{sub_answer}")

    return "\n\n".join(parts)
