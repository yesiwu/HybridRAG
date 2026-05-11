"""动态上下文压缩节点

职责：
1. 过滤掉不相关的召回文档，保留高质量块
2. 提取历史回答的关键内容
3. 生成压缩后的上下文摘要
4. 清理无关的工具调用 message，防止上下文污染
"""
import json
from langchain_core.messages import SystemMessage, HumanMessage

COMPRESS_PROMPT = """你是一个信息提取和压缩专家。

## 任务
从检索到的文档和历史回答中提取关键信息，生成结构化的压缩摘要。

## 要求
1. **过滤无关文档**：只保留与任务查询直接相关的内容
2. **提取关键信息**：从相关文档中提取核心事实、数据、观点
3. **整合历史回答**：将历史回答中有价值的部分融入摘要
4. **保持简洁**：摘要应精炼，去除冗余信息

## 输出格式
```json
{
  "compressed_context": "压缩后的关键信息摘要（200-500字）",
  "relevant_urls": ["相关文档的 URL 列表"],
  "key_facts": ["关键事实列表"]
}
```

只输出 JSON，不要输出其他内容。"""


def sub_compressor(state, llm) -> dict:
    """上下文压缩：过滤无关内容，提取关键信息"""
    task_query = state.get("task_query", "")
    iteration_count = state.get("iteration_count", 0)
    retrieved_chunks = state.get("retrieved_chunks", [])
    retrieval_history = state.get("retrieval_history", [])
    answer_history = state.get("answer_history", [])
    compressed_context = state.get("compressed_context", "")
    filtered_chunks = state.get("filtered_chunks", [])

    print(f"\n[Compressor] 开始压缩 (第{iteration_count}轮后)")

    # ========== 构建待压缩内容 ==========
    # 1. 当前轮次的召回块
    current_chunks_text = ""
    for chunk in retrieved_chunks:
        url = chunk.get("metadata", {}).get("url", "N/A")
        text = chunk.get("text", "")[:500]
        current_chunks_text += f"- [{url}] {text}\n"

    # 2. 历史回答
    history_answers_text = ""
    for hist in answer_history:
        history_answers_text += f"\n### 第{hist['iteration']}轮回答\n查询: {hist['query']}\n{hist['answer'][:500]}\n"

    # 3. 之前的压缩上下文
    prev_context = compressed_context if compressed_context else "无"

    # ========== 调用 LLM 进行压缩 ==========
    messages = [
        SystemMessage(content=COMPRESS_PROMPT),
        HumanMessage(content=f"""## 原始任务查询
{task_query}

## 当前轮次召回块
{current_chunks_text if current_chunks_text else "无"}

## 历史回答
{history_answers_text if history_answers_text else "无"}

## 之前的压缩上下文
{prev_context}

请提取关键信息并压缩。"""),
    ]

    resp = llm.invoke(messages)

    # ========== 解析压缩结果 ==========
    try:
        content = resp.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)
        new_compressed_context = result.get("compressed_context", "")
        relevant_urls = result.get("relevant_urls", [])
        key_facts = result.get("key_facts", [])

        # 合并压缩上下文
        if compressed_context:
            final_context = f"{compressed_context}\n\n## 第{iteration_count}轮新增\n{new_compressed_context}"
        else:
            final_context = new_compressed_context

        # 过滤高质量召回块：只保留与 relevant_urls 匹配的块
        relevant_url_set = set(relevant_urls)
        new_filtered = []
        for chunk in retrieved_chunks:
            chunk_url = chunk.get("metadata", {}).get("url", "")
            if chunk_url in relevant_url_set or not relevant_url_set:
                new_filtered.append(chunk)

        # 合并历史过滤块
        all_filtered = filtered_chunks + new_filtered
        seen_texts = set()
        unique_filtered = []
        for chunk in sorted(all_filtered, key=lambda x: x.get("rerank_score", 0), reverse=True):
            text_prefix = chunk.get("text", "")[:100]
            if text_prefix not in seen_texts:
                seen_texts.add(text_prefix)
                unique_filtered.append(chunk)

        print(f"[Compressor] 压缩完成，保留 {len(unique_filtered)} 个高质量召回块")
        print(f"[Compressor] 关键事实: {len(key_facts)} 条")

        return {
            "compressed_context": final_context,
            "filtered_chunks": unique_filtered[:15],  # 最多保留 15 个高质量块
        }

    except (json.JSONDecodeError, IndexError) as e:
        print(f"[Compressor] JSON 解析失败: {e}，使用默认压缩")
        # 默认保留 rerank_score 较高的块
        high_score_chunks = [c for c in retrieved_chunks if c.get("rerank_score", 0) > 5.0]
        return {
            "compressed_context": f"{compressed_context}\n\n## 第{iteration_count}轮\n{resp.content[:300]}",
            "filtered_chunks": filtered_chunks + high_score_chunks,
        }
