from typing import List, Annotated, Set, Any
from langgraph.graph import MessagesState
import operator

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
    agent_answers: Annotated[List[dict], accumulate_or_reset] = [] 

    # --- DAG 任务编排核心字段 ---
    # 结构示例:
    # {
    #     "id": 1,
    #     "query": "2024年全球半导体行业营收数据",
    #     "depends_on": [],
    #     "status": "pending"  # pending, running, completed
    # },
    # {
    #     "id": 2,
    #     "query": "基于任务1的营收数据，分析英伟达的市场份额",
    #     "depends_on": [1],
    #     "status": "pending"
    # }
    dag_tasks: List[dict[str, Any]] = [] 
    
    # 已完成任务的 ID 集合，使用 set_union 确保并行子图返回时 ID 正常合并
    completed_task_ids: Annotated[Set[int], set_union] = set()

    # 图短期记忆：planner 的推理过程、查询改写、关键词等（供后续节点参考）
    plan_memory: List[dict] = []

    # 最终输出
    final_answer: str = ""

class AgentState(MessagesState):
    """单个子任务执行图（子图）的状态定义"""
    # 基础任务信息
    question: str = ""                                # 当前子智能体负责处理的特定查询
    question_index: int = 0                           # 该任务在整个任务序列中的 ID 或索引

    # 上层依赖
    dependency_context: List[dict] = []                # 接收来自上层依赖的任务结果
    
    search_results: List[str] = [] # 搜索到的具体内容（如文本块、URL等），供搜索工具调用后存储，供后续工具调用和回答生成使用
    
    # 检索与工具相关
    retrieval_keys: Annotated[Set[str], set_union] = set()  # 已检索标识集合（去重）
    searched_keywords: Annotated[Set[str], set_union] = set()  # 已搜索关键词（防重复搜索）
    crawled_urls: Annotated[Set[str], set_union] = set()      # 已爬取URL（防重复爬取）
    retrieval_chunks: List[dict] = []                 # 原始召回的文本块
    filtered_high_quality_chunks: List[dict] = []      # 过滤后的高质量块

    # 动态上下文压缩（你的核心安全阀机制）
    context_history: List[str] = []                   # 完整上下文历史
    context_summary: str = ""                         # 压缩后的摘要（用于Token压缩）
    current_context_tokens: int = 0                   # 当前上下文Token数
    max_context_tokens: int = 8000                    # 动态膨胀上限（初始8000）
    tool_call_count: Annotated[int, operator.add] = 0  # 累计工具调用次数
    iteration_count: Annotated[int, operator.add] = 0  # 累计迭代轮次
    
    # 回答与反思
    final_answer: str = ""                            # 当前子智能体生成的最终回答
    agent_answers: List[dict] = []                    # 暂存回答，最终同步到主State
    need_reflect: bool = False                        # 是否需要查询重写
    reflection: str = ""                               # 反思结果（用于压缩总结）

    # 主图同步字段（子图完成后写回主图）
    completed_task_ids: Annotated[Set[int], set_union] = set()


