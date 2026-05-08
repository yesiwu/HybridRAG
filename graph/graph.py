from langgraph.graph import START, END, StateGraph
from functools import partial

# 导入状态定义
from state.graph_state import State, AgentState
# 导入节点函数
from graph.nodes import (
    main_task_planner,
    subgraph_plan_node,
    reflect_node,
    rewrite_query_node,
    compress_context_node,
    collect_and_verify_answer,
)
# 导入路由函数
from graph.nodes.edges import (
    route_after_reflect,
    subgraph_final_router,
    main_router,
)


def _dummy_tool_node(state):
    """伪代码工具节点：模拟工具调用结果，后续替换为真实 ToolNode"""
    return {
        "tool_call_count": 1,
    }


def create_agent_graph(llm, tools_list):
    """
    最终版 RAG 智能体工作流
    主图：主规划 → DAG 调度 → 子图（并行） → 汇总验证
    子图：规划 → 工具 → 反思 → 压缩 → 重写 → 迭代/完成
    """
    # ==========================
    # 一、子图构建
    # ==========================
    agent_subgraph_builder = StateGraph(AgentState)

    # 子图节点
    agent_subgraph_builder.add_node("subgraph_planner", partial(subgraph_plan_node, llm=llm))
    agent_subgraph_builder.add_node("tools", _dummy_tool_node)
    agent_subgraph_builder.add_node("reflect_node", partial(reflect_node, llm=llm))
    agent_subgraph_builder.add_node("rewrite_query_node", partial(rewrite_query_node, llm=llm))
    agent_subgraph_builder.add_node("compress_context_node", partial(compress_context_node, llm=llm))
    agent_subgraph_builder.add_node("subgraph_final_router", subgraph_final_router)

    # 子图连线
    agent_subgraph_builder.add_edge(START, "subgraph_planner")

    agent_subgraph_builder.add_conditional_edges(
        "subgraph_planner",
        lambda state: "tools" if state.get("need_tool_call", False) else "reflect_node",
        {"tools": "tools", "reflect_node": "reflect_node"},
    )

    agent_subgraph_builder.add_edge("tools", "subgraph_planner")

    agent_subgraph_builder.add_conditional_edges(
        "reflect_node",
        route_after_reflect,
        {
            "compress_context_node": "compress_context_node",
            "subgraph_final_router": "subgraph_final_router",
        },
    )

    agent_subgraph_builder.add_edge("compress_context_node", "rewrite_query_node")
    agent_subgraph_builder.add_edge("rewrite_query_node", "subgraph_planner")
    agent_subgraph_builder.add_edge("subgraph_final_router", END)

    atomic_agent_subgraph = agent_subgraph_builder.compile()

    # ==========================
    # 二、主图构建
    # ==========================
    main_graph_builder = StateGraph(State)

    # 主图节点
    main_graph_builder.add_node("main_task_planner", partial(main_task_planner, llm=llm))
    main_graph_builder.add_node("atomic_agent", atomic_agent_subgraph)
    main_graph_builder.add_node("collect_verify", partial(collect_and_verify_answer, llm=llm))

    # 主图连线
    main_graph_builder.add_edge(START, "main_task_planner")

    main_graph_builder.add_conditional_edges(
        "main_task_planner",
        main_router,
        {
            "atomic_agent": "atomic_agent",
            "collect_verify": "collect_verify",
        },
    )

    main_graph_builder.add_edge("atomic_agent", "main_task_planner")
    main_graph_builder.add_edge("collect_verify", END)

    agent_graph = main_graph_builder.compile()

    print("[Graph] 智能体工作流编译成功")
    return agent_graph
