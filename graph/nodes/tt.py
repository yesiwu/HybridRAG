from typing import List, Annotated, Set, Any
from langgraph.graph import MessagesState
from enum import Enum


def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    """累加或重置回答列表。如果新数据中包含 __reset__ 标识，则清空历史。"""
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new


def set_union(a: Set[Any], b: Set[Any]) -> Set[Any]:
    """对集合执行并集操作，确保并行写入时数据不丢失。"""
    return a | b


class State(MessagesState):
    """主智能体图的状态定义"""
    # --- 基础上下文字段 ---
    conversation_summary: str = ""      # 整个对话的全局摘要（跨轮次的长期记忆）
    originalQuery: str = ""             # 用户的原始输入

    # 核心：通过注解实现「多智能体回答累加」
    # 结构: {task_id: int, query: str, retrieval_docs: List[dict], sub_answer: str}
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []
        # --- DAG 任务编排核心字段 ---
    # 结构示例:
    # {
    #     "task_id": 1,
    #     "query": "2024年全球半导体行业营收数据",
    #     "depends_on": [],
    #     "status": "pending"  # pending, running, completed, failed
    # },
    # {
    #     "task_id": 2,
    #     "query": "基于任务1的营收数据，分析英伟达的市场份额",
    #     "depends_on": [1],
    #     "status": "pending"
    # }
    # --- DAG 任务编排核心字段 ---
    dag_tasks: List[dict[str, Any]] = []

    # 已完成任务的 ID 集合
    completed_task_ids: Annotated[Set[int], set_union] = set()

    # 最终输出
    final_answer: str = ""


class AgentState(MessagesState):
    """单个子任务执行图（子图）的状态定义"""

    # ==================== 基础任务信息 ====================
    task_query: str = ""                # 当前子智能体负责处理的原始查询
    task_id: int = 0                    # 该任务在整个任务序列中的 ID

    # ==================== 上层依赖 ====================
    dependency_context: List[dict] = [] # 接收来自上层依赖的任务结果

    # ==================== 检索相关 ====================
    search_query: str = ""              # 实际搜索的查询词（可能被重写过）
    """
    {
  "query": "2025年发展最快的几家ai公司",
  "results": [
    {
      "title": "2025年度AI创业公司TOP50",
      "url": "http://enet16.com/article/2026/0202/A202602022603.html",
      "content": "DBC德本咨询发布的《2025年度AI创业公司TOP50》榜单显示，2025年中国AI创业公司发展迅猛，涵盖AI大模型、具身智能、AIGC、自动驾驶、AI芯片等多个领域。榜单前十名包括：DeepSeek（AI大模型，2023年成立）、Minimax稀宇科技（AIGC，2021年成立）、月之暗面Kimi（AI大模型，2023年成立）、银河通用机器人（具身智能，2023年成立）、智元机器人（具身智能，2023年成立）、阶跃星辰（通用大模型，2023年成立）、百川智能（AIGC，2023年成立）、九识智能（自动驾驶，2021年成立）、零一万物（AIGC，2023年成立）、众擎机器人（具身智能，2023年成立）。报告指出，2025年头部AI公司融资额达数亿甚至超十亿元，AI应用落地已从概念验证进入商业快速变现期，创业公司从诞生到规模化盈利的周期被极度压缩。"
    },
    {
      "title": "2025年美国获得亿元级融资的55家AI初创公司盘点",
      "url": "https://m.zhiding.cn/article/3177273.htm",
      "content": "至顶网盘点显示，2025年美国AI初创公司融资热潮持续，共有55家公司获得1亿美元以上融资。其中，超过10亿美元的融资轮次有8起，较2024年的3起大幅增加。主要融资事件包括：Anthropic完成130亿美元F轮融资（估值1830亿美元）、OpenAI完成400亿美元融资（估值3000亿美元）、xAI完成200亿美元E轮融资、Reflection AI完成20亿美元B轮融资（估值80亿美元）、Cursor（Anysphere）完成23亿美元融资（估值293亿美元）、Luma AI完成9亿美元C轮融资（估值40亿美元）等。报告指出，AI行业在2025年保持强劲发展势头，融资规模不断扩大，涵盖AI基础设施、编程工具、医疗、法律等多个垂直领域。"
    }
  ]
}
    """
    search_results: List[dict] = []     # Tavily 搜索返回的原始结果
    retrieved_chunks: List[dict] = []   # RAG 精排后的召回块
    # 结构: {rank: int, text: str, metadata: {title, url, chunk_index}, rerank_score: float}

    # ==================== 迭代控制 ====================
    iteration_count: int = 0            # 当前迭代轮次（用于限制最大迭代）
    max_iterations: int = 3             # 最大迭代轮次

    # ==================== 反思与重写 ====================
    need_reflect: bool = False          # 是否需要查询重写
    reflection: str = ""                # 反思结果（评估当前回答质量）

    # ==================== 上下文压缩 ====================
    context_summary: str = ""           # 压缩后的检索上下文摘要
    crawled_urls: List[str] = []        # 已爬取的 URL 列表（避免重复搜索）

    # ==================== 最终输出 ====================
    final_answer: str = ""              # 当前子智能体生成的最终回答


class TaskStatus(str, Enum):
    """任务生命周期状态"""
    PENDING = "pending"       # 等待依赖完成
    READY = "ready"           # 依赖已满足，可被调度
    RUNNING = "running"       # 子图执行中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 执行失败


