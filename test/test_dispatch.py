"""诊断：测试 DAG 调度逻辑，检查 completed_task_ids 是否正确累积"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.nodes.edges import main_router, is_task_ready, validate_dag_tasks

# 模拟状态
state_after_batch1 = {
    "dag_tasks": [
        {"task_id": 1, "query": "获取2025年主要AI公司列表", "depends_on": []},
        {"task_id": 2, "query": "分析各公司发展速度", "depends_on": [1]},
        {"task_id": 3, "query": "比较得出最快发展公司", "depends_on": [1, 2]},
    ],
    "completed_task_ids": {1},  # 假设任务1已完成
    "agent_answers": [
        {"task_id": 1, "query": "获取2025年主要AI公司列表", "sub_answer": "...", "retrieval_docs": []}
    ],
}

print("=== 测试 DAG 调度 ===")
dag_tasks = state_after_batch1["dag_tasks"]
completed = state_after_batch1["completed_task_ids"]

print(f"dag_tasks: {len(dag_tasks)} 个")
print(f"completed_task_ids: {completed}")

# 验证 DAG
valid, error = validate_dag_tasks(dag_tasks)
print(f"DAG 验证: valid={valid}, error={error}")

# 找出可执行任务
executable = [t for t in dag_tasks if is_task_ready(t, completed)]
print(f"可执行任务: {[t['task_id'] for t in executable]}")

# 测试 main_router
result = main_router(state_after_batch1)
print(f"\nmain_router 返回: {type(result).__name__}")
if isinstance(result, list):
    for send in result:
        print(f"  Send -> {send.node}, task_id={send.arg.get('task_id')}, task_query={send.arg.get('task_query', 'N/A')[:50]}")
else:
    print(f"  路由到: {result}")

# 测试空 completed_task_ids 的情况
print("\n=== 测试 completed_task_ids 为空的情况 ===")
state_empty = {**state_after_batch1, "completed_task_ids": set()}
result2 = main_router(state_empty)
print(f"main_router 返回: {type(result2).__name__}")
if isinstance(result2, list):
    for send in result2:
        print(f"  Send -> {send.node}, task_id={send.arg.get('task_id')}")
