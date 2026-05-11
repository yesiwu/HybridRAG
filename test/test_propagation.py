"""测试：子图返回主图字段时的传播行为"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import List, Annotated, Set
from langgraph.graph import START, END, StateGraph, MessagesState


# 主图 state
class MainState(MessagesState):
    results: Annotated[List[str], lambda a, b: a + b] = []
    done_ids: Annotated[Set[int], lambda a, b: a | b] = set()


# 子图 state - 不包含 results 和 done_ids
class SubState(MessagesState):
    task_id: int = 0


# 子图节点
def sub_work(state):
    return {}

def sub_sync(state):
    tid = state.get("task_id", 0)
    # 返回主图的字段，即使 SubState 中没有定义
    return {"results": [f"answer_{tid}"], "done_ids": {tid}}


# 构建子图
sub = StateGraph(SubState)
sub.add_node("work", sub_work)
sub.add_node("sync", sub_sync)
sub.add_edge(START, "work")
sub.add_edge("work", "sync")
sub.add_edge("sync", END)
sub_graph = sub.compile()


# 主图节点
def planner(state):
    if state.get("results"):
        print(f"  [planner] results已有数据: {state['results']}, done_ids: {state.get('done_ids')}")
        return {}
    print(f"  [planner] 首次规划")
    return {}

# 构建主图
main = StateGraph(MainState)
main.add_node("planner", planner)
main.add_node("sub", sub_graph)
main.add_edge(START, "planner")
main.add_edge("planner", "sub")
main.add_edge("sub", "planner")
main_graph = main.compile()

print("=== 测试: SubState 不包含主图字段 ===")
for step in main_graph.stream({"messages": []}, {"recursion_limit": 10}):
    for name, out in step.items():
        keys = list(out.keys()) if out else ["(empty)"]
        print(f"  {name} -> keys={keys}")
        if out and "results" in out:
            print(f"    results: {out['results']}, done_ids: {out.get('done_ids')}")
