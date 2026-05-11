"""诊断：子图结果是否正确合并到主图状态"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.graph_state import State, AgentState

# 模拟子图返回的结果
subgraph_output = {
    "agent_answers": [{"task_id": 1, "query": "test", "sub_answer": "answer", "retrieval_docs": []}],
    "completed_task_ids": {1},
}

# 模拟主图当前状态
main_state = {
    "originalQuery": "test query",
    "dag_tasks": [
        {"task_id": 1, "query": "task 1", "depends_on": []},
        {"task_id": 2, "query": "task 2", "depends_on": [1]},
    ],
    "completed_task_ids": set(),
    "agent_answers": [],
    "final_answer": "",
    "verification_report": {},
    "conversation_summary": "",
}

print("=== 测试 State 合并行为 ===")
print(f"主图状态 completed_task_ids: {main_state['completed_task_ids']}")
print(f"子图返回 completed_task_ids: {subgraph_output['completed_task_ids']}")

# 检查 State 的字段定义
import typing
state_annotations = getattr(State, '__annotations__', {})
print(f"\nState annotations:")
for k, v in state_annotations.items():
    print(f"  {k}: {v}")

# 检查 completed_task_ids 是否有 set_union reducer
completed_field = state_annotations.get('completed_task_ids')
print(f"\ncompleted_task_ids field: {completed_field}")
if completed_field:
    args = getattr(completed_field, '__args__', None)
    metadata = getattr(completed_field, '__metadata__', None)
    print(f"  args: {args}")
    print(f"  metadata: {metadata}")

# 检查 agent_answers 是否有 accumulate_or_reset reducer
answers_field = state_annotations.get('agent_answers')
print(f"\nagent_answers field: {answers_field}")
if answers_field:
    args = getattr(answers_field, '__args__', None)
    metadata = getattr(answers_field, '__metadata__', None)
    print(f"  args: {args}")
    print(f"  metadata: {metadata}")
