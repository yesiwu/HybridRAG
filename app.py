
"""
HybridRAG 完整流程测试入口

流程：
1. main_planner: 任务拆解，生成 DAG
2. sub_planner: 子任务执行（搜索 + RAG）
3. sub_reflector: 反思评估
4. sub_compressor: 上下文压缩（如果需要迭代）
5. subgraph_result_sync: 子图结果同步到主图
6. main_summarizer: 汇总验证 + 冲突消解
"""
import sys
import os
import json

# 修复 Windows 控制台中文/emoji 输出
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(__file__))

from utils.llm_client import LLMClient
from graph.graph import create_agent_graph


def run_full_pipeline(query: str):
    """运行完整的 HybridRAG 流程"""
    print("=" * 70)
    print("HybridRAG 完整流程测试")
    print("=" * 70)
    print(f"\n用户查询: {query}\n")

    # 初始化 LLM
    client = LLMClient()
    llm = client.get_llm()

    # 创建图
    print("[初始化] 创建智能体工作流图...")
    graph = create_agent_graph(llm, [])

    # 初始状态
    initial_state = {
        "originalQuery": query,
        "conversation_summary": "",
        "dag_tasks": [],
        "agent_answers": [],
        "completed_task_ids": set(),
        "final_answer": "",
        "verification_report": {},
    }

    # 运行图（使用 invoke 而非 stream，避免多线程下 Reranker predict 死锁）
    print("\n" + "=" * 70)
    print("开始执行工作流")
    print("=" * 70 + "\n")

    final_state = graph.invoke(initial_state, {"recursion_limit": 50})

    # 输出最终结果
    print("\n" + "=" * 70)
    print("执行完成 - 最终结果")
    print("=" * 70)

    final_answer = final_state.get("final_answer", "")
    verification_report = final_state.get("verification_report", {})
    dag_tasks = final_state.get("dag_tasks", [])
    agent_answers = final_state.get("agent_answers", [])

    if dag_tasks:
        print(f"\nDAG 任务数: {len(dag_tasks)}")
        for task in dag_tasks:
            print(f"  - 任务 {task.get('task_id')}: {task.get('query', '')[:60]}")

    if agent_answers:
        print(f"\n子图回答数: {len(agent_answers)}")
        for ans in agent_answers:
            print(f"  - 任务 {ans.get('task_id')}: 迭代 {ans.get('iteration_count', 0)} 次")

    # 保存结果
    output = {
        "query": query,
        "final_answer": final_answer,
        "verification_report": verification_report,
        "dag_tasks": dag_tasks,
        "agent_answers": [
            {
                "task_id": ans.get("task_id"),
                "query": ans.get("query"),
                "sub_answer": ans.get("sub_answer", "")[:500],
                "iteration_count": ans.get("iteration_count", 0),
            }
            for ans in agent_answers
        ],
    }

    output_file = os.path.join(os.path.dirname(__file__), "test", "pipeline_output.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n最终回答:\n{final_answer}")
    print(f"\n结果已保存到: {output_file}")

    return output


if __name__ == "__main__":
    # 测试查询
    test_query = "2025年发展最快的AI公司"
    run_full_pipeline(test_query)
