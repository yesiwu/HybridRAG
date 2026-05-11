"""测试 RAG 索引与混合检索功能"""
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.search import search_tavily
from tools.rag import rag_index_and_retrieve

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "rag_output.json")


def test_rag_index_and_retrieve():
    """测试完整的 RAG 流程：搜索 -> 索引 -> 混合召回 -> Rerank"""
    task_id = "test_rag_001"
    query = "2025年发展最快的AI公司有哪些"

    # Step 1: 调用搜索工具获取搜索结果
    print(f"\n[测试] Step 1: 搜索 '{query}'")
    search_result = search_tavily.invoke(query)
    print(f"[测试] 搜索返回 {len(search_result.get('results', []))} 条结果")

    # Step 2: 调用 RAG 工具进行索引与检索
    print(f"\n[测试] Step 2: RAG 索引与检索")
    rag_result = rag_index_and_retrieve.invoke({
        "task_id": task_id,
        "query": query,
        "search_results": search_result,
        "top_k": 5
    })

    # 写入文件方便查看
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(rag_result, f, ensure_ascii=False, indent=2)

    # 断言验证
    assert "task_id" in rag_result, "缺少 task_id"
    assert "retrieved_chunks" in rag_result, "缺少 retrieved_chunks"
    assert rag_result["task_id"] == task_id
    assert rag_result["num_indexed"] > 0, "未索引任何文档"
    assert len(rag_result["retrieved_chunks"]) > 0, "未召回任何结果"
    assert len(rag_result["retrieved_chunks"]) <= 5, "返回结果超过 top_k"

    # 验证返回结构
    for chunk in rag_result["retrieved_chunks"]:
        assert "rank" in chunk, "缺少 rank"
        assert "text" in chunk, "缺少 text"
        assert "rerank_score" in chunk, "缺少 rerank_score"

    print(f"\n[测试] 索引了 {rag_result['num_indexed']} 个文档块")
    print(f"[测试] 精排后返回 {len(rag_result['retrieved_chunks'])} 个结果")

    # 打印 top 3 结果摘要
    print("\n=== Top 3 检索结果 ===")
    for chunk in rag_result["retrieved_chunks"][:3]:
        print(f"\n[Rank {chunk['rank']}] Rerank Score: {chunk['rerank_score']:.4f}")
        print(f"来源: {chunk['metadata'].get('url', 'N/A')}")
        print(f"内容: {chunk['text'][:150]}...")

    print(f"\n完整结果已写入: {OUTPUT_FILE}")
