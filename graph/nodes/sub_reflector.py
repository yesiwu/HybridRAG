"""子图反思节点 - 评估回答质量，决定是否需要补充检索"""
import json
from langchain_core.messages import SystemMessage, HumanMessage

REFLECT_PROMPT = """你是一个严格的任务质量评估专家。

## 任务
根据用户的任务查询和检索到的回答，评估回答是否充分满足了任务需求。

## 评估标准
1. **完整性**：回答是否涵盖了任务查询的所有方面
2. **准确性**：回答内容是否有检索结果支撑，而非编造
3. **时效性**：信息是否是最新的、有效的
4. **深度**：回答是否足够详细，而非泛泛而谈

## 输出格式
请以 JSON 格式输出评估结果：
```json
{
  "need_reflect": true/false,
  "reflection": "评估说明",
  "rewritten_query": "如果 need_reflect=true，填写优化后的搜索查询；否则为空字符串"
}
```

- 如果回答质量足够好，`need_reflect` 设为 false，`reflection` 给出肯定说明
- 如果需要改进，`need_reflect` 设为 true，`reflection` 列出 1-3 个具体改进建议，`rewritten_query` 提供优化后的搜索词

只输出 JSON，不要输出其他内容。"""


def sub_reflector(state, llm) -> dict:
    """反思：评估当前回答质量，决定是否需要补充检索"""
    task_query = state.get("task_query", "")
    final_answer = state.get("final_answer", "")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    # 如果已经达到最大迭代次数，直接结束
    if iteration_count >= max_iterations:
        print(f"[Reflector] 已达到最大迭代次数 {max_iterations}，结束反思")
        return {
            "need_reflect": False,
            "reflection": f"已达到最大迭代次数 {max_iterations}，任务结束。",
            "rewritten_query": "",
        }

    # 如果没有回答，需要重新检索
    if not final_answer:
        return {
            "need_reflect": True,
            "reflection": "未生成有效回答，需要重新检索。",
            "rewritten_query": task_query,
            "iteration_count": iteration_count + 1,
        }

    # 调用 LLM 评估回答质量
    print(f"\n[Reflector] 评估回答质量 (迭代 {iteration_count + 1}/{max_iterations})")

    messages = [
        SystemMessage(content=REFLECT_PROMPT),
        HumanMessage(content=f"## 任务查询\n{task_query}\n\n## 检索回答\n{final_answer}\n\n请评估回答质量。"),
    ]

    resp = llm.invoke(messages)

    # 解析 JSON 响应
    try:
        content = resp.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)
        need_reflect = result.get("need_reflect", False)
        reflection = result.get("reflection", "")
        rewritten_query = result.get("rewritten_query", "")

        print(f"[Reflector] need_reflect={need_reflect}")
        if need_reflect:
            print(f"[Reflector] 改进建议: {reflection[:150]}...")
            print(f"[Reflector] 重写查询: {rewritten_query}")
        else:
            print(f"[Reflector] 评估通过: {reflection[:100]}...")

        return {
            "need_reflect": need_reflect,
            "reflection": reflection,
            "rewritten_query": rewritten_query,
            "iteration_count": iteration_count + 1,
        }

    except (json.JSONDecodeError, IndexError) as e:
        print(f"[Reflector] JSON 解析失败: {e}，默认结束反思")
        return {
            "need_reflect": False,
            "reflection": f"评估结果解析失败，默认通过。原始响应: {resp.content[:200]}",
            "rewritten_query": "",
            "iteration_count": iteration_count + 1,
        }
