"""主规划节点 - 任务拆解与 DAG 构建"""
import json


# ── 提示词 ──────────────────────────────────────────────
PLANNER_SYSTEM_PROMPT = """你是一位资深的任务规划专家。

你的职责：将用户的查询拆解为一组具有 DAG 依赖关系的结构化子任务。

规则：
1. 每个子任务必须是独立可执行的原子任务。
2. 使用 `task_id`（从 1 开始）标识任务。
3. 使用 `depends_on` 标识依赖关系。如果任务 B 需要任务 A 的结果，则 B 的 `depends_on` 包含 A 的 `task_id`。
4. 简单查询（单一事实）返回 1 个任务；复杂查询（多方面、对比分析）拆分为 2-5 个任务。

严格按以下 JSON 格式输出，不要输出其他内容：
```json
{
  "reasoning": "拆解思路简述",
  "dag_tasks": [
    {"task_id": 1, "query": "子任务描述", "depends_on": [], "status": "pending"},
    {"task_id": 2, "query": "子任务描述", "depends_on": [1], "status": "pending"}
  ]
}
```"""


# ── 节点函数 ──────────────────────────────────────────────
def main_planner(state, llm) -> dict:
    """
    主规划节点：调用 LLM 拆解任务，生成 DAG 任务列表。
    将推理过程写入 plan_memory 作为图短期记忆。
    """
    # 已有任务时跳过拆解（子图完成后回到此节点检查下一批）
    if state.get("dag_tasks"):
        return {}

    query = state.get("originalQuery", "")
    summary = state.get("conversation_summary", "")

    # 构造用户提示词
    user_prompt = f"用户查询：{query}"
    if summary:
        user_prompt = f"对话摘要：{summary}\n{user_prompt}"

    # 调用 LLM
    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [SystemMessage(content=PLANNER_SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages).content

    # 解析 JSON
    dag_tasks, reasoning = _parse_plan(response)

    print(f"[Main Planner] 任务拆解完成: {len(dag_tasks)} 个子任务, 推理: {reasoning}")

    return {
        "dag_tasks": dag_tasks,
        "conversation_summary": summary + f"\n[规划推理] {reasoning}" if summary else reasoning,
    }




def _parse_plan(response: str) -> tuple:
    """从 LLM 响应中解析任务规划 JSON"""
    try:
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]
        data = json.loads(json_str)

        dag_tasks = data.get("dag_tasks", [])

        reasoning = data.get("reasoning", "")

        # 校验：确保每个任务有 task_id、query、depends_on
        for task in dag_tasks:
            task.setdefault("task_id", 0)
            task.setdefault("query", "")
            task.setdefault("depends_on", [])
            task.setdefault("status", "pending")

        return dag_tasks, reasoning

    except (json.JSONDecodeError, IndexError):
        # 解析失败，回退为单任务
        fallback = [{"task_id": 1, "query": response.strip()[:200], "depends_on": [], "status": "pending"}]
        return fallback, "JSON 解析失败，回退为单任务"
