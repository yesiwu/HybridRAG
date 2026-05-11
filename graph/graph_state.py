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
    conversation_summary: str = ""
    originalQuery: str = ""
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []
    dag_tasks: List[dict[str, Any]] = []
    completed_task_ids: Annotated[Set[int], set_union] = set()
    final_answer: str = ""


class AgentState(MessagesState):
    """单个子任务执行图（子图）的状态定义"""

    # ==================== 基础任务信息 ====================
    task_query: str = ""                # 原始任务查询（不变）
    task_id: int = 0                    # 任务 ID

    # ==================== 上层依赖 ====================
    dependency_context: List[dict] = [] # 依赖任务的结果

    # ==================== 迭代控制 ====================
    iteration_count: int = 0            # 当前迭代轮次
    max_iterations: int = 3             # 最大迭代轮次

    # ==================== 检索历史（累积，用于避免重复） ====================
    search_query: str = ""              # 当前轮次的搜索词
    crawled_urls: List[str] = []        # 已爬取的 URL（避免重复搜索）

    # ==================== 多轮召回块历史 ====================
    # 结构: [{iteration: int, chunks: [{rank, text, metadata, rerank_score}]}]
    retrieval_history: List[dict] = []

    # ==================== 多轮回答历史 ====================
    # 结构: [{iteration: int, query: str, answer: str, key_points: str}]
    answer_history: List[dict] = []

    # ==================== 当前轮次的结果（最新） ====================
    search_results: List[dict] = []     # 当前轮次 Tavily 搜索结果
    retrieved_chunks: List[dict] = []   # 当前轮次 RAG 精排召回块
    final_answer: str = ""              # 当前轮次的回答

    # ==================== 压缩后的上下文 ====================
    compressed_context: str = ""        # 压缩后的关键信息摘要（跨轮次累积）
    filtered_chunks: List[dict] = []    # 过滤后的高质量召回块（跨轮次累积）

    # ==================== 反思相关 ====================
    need_reflect: bool = False          # 是否需要重写查询
    reflection: str = ""                # 反思结果
    rewritten_query: str = ""           # 重写后的查询


class TaskStatus(str, Enum):
    """任务生命周期状态"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
