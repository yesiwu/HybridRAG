from langgraph.graph import START, END, StateGraph
from functools import partial

# 导入状态定义
from state.graph_state import State, AgentState
# 导入节点函数
from graph.nodes import (
    main_planner as main_task_planner,
    main_summarizer,
    sub_planner,
    sub_reflector,
    sub_rewrite,
    sub_compressor,
    subgraph_result_sync
)
# 导入路由函数
from graph.nodes.edges import (
    route_after_reflect,
    main_router,
)
from tools import rag


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

    # 子图节点  partial = 给函数 “预先绑定” 参数，生成一个新函数
    agent_subgraph_builder.add_node("sub_planner", partial(sub_planner, llm=llm))
    agent_subgraph_builder.add_node("sub_reflector", partial(sub_reflector, llm=llm))
    agent_subgraph_builder.add_node("sub_rewrite", partial(sub_rewrite, llm=llm))
    agent_subgraph_builder.add_node("sub_compressor", partial(sub_compressor, llm=llm))
    agent_subgraph_builder.add_node("subgraph_result_sync", subgraph_result_sync)

    # 子图连线
    agent_subgraph_builder.add_edge(START, "sub_planner")
    agent_subgraph_builder.add_edge("sub_planner", "sub_reflector")
    #我回答之后，还没有把结果写入主agent state，这时候应该更新了子agent state，进入反思节点，判断是否需要重写查询，如果需要就进入重写节点，否则直接进入最终路由
    agent_subgraph_builder.add_conditional_edges(
        "sub_reflector",
        route_after_reflect
    )

    #压缩是为了防止上下文过长，除去原来的冗余信息，保留原来关键信息， 同时给新的查询腾出空间。
    agent_subgraph_builder.add_edge("sub_rewrite", "sub_compressor")
    #重写之后直接进入压缩节点，压缩节点完成后直接进入规划节点，进行下一轮迭代
    agent_subgraph_builder.add_edge("sub_compressor", "sub_planner")
    
    agent_subgraph_builder.add_edge("subgraph_result_sync", END)

    atomic_agent_subgraph = agent_subgraph_builder.compile()

    # ==========================
    # 二、主图构建
    # ==========================
    main_graph_builder = StateGraph(State)

    # 主图节点
    main_graph_builder.add_node("main_task_planner", partial(main_task_planner, llm=llm))
    main_graph_builder.add_node("atomic_agent_subgraph", atomic_agent_subgraph)
    main_graph_builder.add_node("main_summarizer", partial(main_summarizer, llm=llm))

    # 主图连线
    main_graph_builder.add_edge(START, "main_task_planner")

    main_graph_builder.add_conditional_edges(
        "main_task_planner",
        main_router,
    )

    main_graph_builder.add_edge("atomic_agent_subgraph", "main_task_planner")
    main_graph_builder.add_edge("main_summarizer", END)

    agent_graph = main_graph_builder.compile()

    print("[Graph] 智能体工作流编译成功")
    return agent_graph
