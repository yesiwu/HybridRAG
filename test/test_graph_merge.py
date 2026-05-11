"""诊断：测试子图结果是否正确合并到主图状态"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langgraph.graph import START, END, StateGraph
from graph.graph_state import State, AgentState

# 模拟子图：直接返回结果
def mock_subgraph_start(state):
    print(f"  [mock_subgraph] 收到 state: task_id={state.get('task_id')}, task_query={state.get('task_query', 'N/A')[:30]}")
    return {}

def mock_subgraph_sync(state):
    task_id = state.get("task_id", 0)
    print(f"  [mock_subgraph_sync] 同步结果: task_id={task_id}")
    return {
        "agent_answers": [{"task_id": task_id, "query": f"task {task_id}", "sub_answer": "answer", "retrieval_docs": []}],
        "completed_task_ids": {task_id},
    }

# 构建子图
sub_builder = StateGraph(AgentState)
sub_builder.add_node("start", mock_subgraph_start)
sub_builder.add_node("sync", mock_subgraph_sync)
sub_builder.add_edge(START, "start")
sub_builder.add_edge("start", "sync")
sub_builder.add_edge("sync", END)
sub_graph = sub_builder.compile()

# 模拟主图：规划 → 子图 → 检查状态
dag_tasks = [
    {"task_id": 1, "query": "task 1", "depends_on": []},
    {"task_id": 2, "query": "task 2", "depends_on": [1]},
]

def mock_planner(state):
    if state.get("dag_tasks"):
        completed = state.get("completed_task_ids", set())
        print(f"\n[mock_planner] 跳过规划，completed_task_ids={completed}")
        return {}
    print(f"\n[mock_planner] 首次规划")
    return {"dag_tasks": dag_tasks}

from graph.nodes.edges import main_router
from functools import partial

# 构建主图
main_builder = StateGraph(State)
main_builder.add_node("planner", mock_planner)
main_builder.add_node("atomic_agent_subgraph", sub_graph)

main_builder.add_edge(START, "planner")
main_builder.add_conditional_edges("planner", main_router)
main_builder.add_edge("atomic_agent_subgraph", "planner")

main_graph = main_builder.compile()

# 运行
print("=" * 50)
print("测试图状态合并")
print("=" * 50)

initial_state = {
    "originalQuery": "test",
    "dag_tasks": [],
    "agent_answers": [],
    "completed_task_ids": set(),
    "final_answer": "",
    "verification_report": {},
    "conversation_summary": "",
}

step_count = 0
for step in main_graph.stream(initial_state, {"recursion_limit": 20}):
    step_count += 1
    for node_name, node_output in step.items():
        output_desc = "(empty)" if not node_output else f"keys={list(node_output.keys())}"
        print(f"  Step {step_count}: {node_name} -> {output_desc}")
        if node_output:
            if "completed_task_ids" in node_output:
                print(f"    completed_task_ids: {node_output['completed_task_ids']}")
            if "agent_answers" in node_output:
                print(f"    agent_answers count: {len(node_output['agent_answers'])}")

    if step_count > 15:
        print("  ... 超过15步，停止")
        break

print(f"\n总步数: {step_count}")
