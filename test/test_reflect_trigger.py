"""测试反思压缩流程触发

策略：构造一个需要多维度信息的复杂问题，或模拟低质量回答来触发反思
"""
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.llm_client import LLMClient
from graph.nodes.sub_planner import sub_planner
from graph.nodes.sub_reflector import sub_reflector
from graph.nodes.sub_compressor import sub_compressor

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "reflect_trigger_output.json")


def build_state(task_query: str, task_id: int = 1, **kwargs):
    state = {
        "task_query": task_query,
        "task_id": task_id,
        "messages": [],
        "iteration_count": 0,
        "max_iterations": 3,
        "crawled_urls": [],
        "retrieval_history": [],
        "answer_history": [],
        "compressed_context": "",
        "filtered_chunks": [],
        "need_reflect": False,
        "reflection": "",
        "rewritten_query": "",
        "search_query": "",
        "search_results": [],
        "retrieved_chunks": [],
        "final_answer": "",
        "dependency_context": [],
    }
    state.update(kwargs)
    return state


def test_with_mock_low_quality_answer():
    """模拟低质量回答，触发反思流程"""
    client = LLMClient()
    llm = client.get_llm()

    task_query = "对比分析2025年中国和美国AI芯片市场规模、主要厂商、技术路线差异，以及未来3年的发展预测"

    # ========== 模拟第一次回答（低质量） ==========
    mock_answer = """AI芯片是人工智能发展的基础。中国和美国都在积极发展AI芯片产业。

美国有NVIDIA、AMD等公司，中国有寒武纪、华为等公司。

AI芯片市场前景广阔。"""

    print("=" * 60)
    print("模拟场景：低质量回答触发反思")
    print("=" * 60)
    print(f"\n任务查询: {task_query}")
    print(f"模拟回答长度: {len(mock_answer)} 字符（明显不足）")

    # 构造状态，假装这是第一轮的结果
    state = build_state(
        task_query,
        final_answer=mock_answer,
        iteration_count=1,
        search_query="2025年中国美国AI芯片市场",
        search_results=[{"title": "测试", "url": "http://test.com", "content": "测试内容"}],
        retrieved_chunks=[{"rank": 1, "text": "测试块", "metadata": {"url": "http://test.com"}, "rerank_score": 5.0}],
        answer_history=[{
            "iteration": 0,
            "query": "2025年中国美国AI芯片市场",
            "answer": mock_answer,
            "key_points": mock_answer[:300],
        }],
        crawled_urls=["http://test.com"],
    )

    # ========== sub_reflector 评估 ==========
    print("\n" + "=" * 60)
    print("sub_reflector 评估")
    print("=" * 60)

    reflect_result = sub_reflector(state, llm)
    state.update(reflect_result)

    print(f"\nneed_reflect: {reflect_result.get('need_reflect')}")
    print(f"reflection: {reflect_result.get('reflection', '')[:200]}")
    print(f"rewritten_query: {reflect_result.get('rewritten_query', '')}")

    # ========== 如果需要反思，进入压缩 ==========
    if reflect_result.get("need_reflect", False):
        print("\n" + "=" * 60)
        print("sub_compressor 压缩")
        print("=" * 60)

        compress_result = sub_compressor(state, llm)
        state.update(compress_result)

        print(f"\n压缩后上下文长度: {len(compress_result.get('compressed_context', ''))}")
        print(f"过滤后召回块数: {len(compress_result.get('filtered_chunks', []))}")

        # ========== 第二轮 sub_planner ==========
        print("\n" + "=" * 60)
        print("第二轮 sub_planner（使用重写查询）")
        print("=" * 60)

        result2 = sub_planner(state, llm)
        state.update(result2)

        print(f"\n第二轮回答长度: {len(result2.get('final_answer', ''))}")

        # ========== 第二轮反思 ==========
        print("\n" + "=" * 60)
        print("第二轮 sub_reflector 评估")
        print("=" * 60)

        reflect_result2 = sub_reflector(state, llm)
        state.update(reflect_result2)

        print(f"\nneed_reflect: {reflect_result2.get('need_reflect')}")

    # ========== 保存结果 ==========
    output = {
        "task_query": task_query,
        "mock_first_answer": mock_answer,
        "final_answer": state.get("final_answer"),
        "iteration_count": state.get("iteration_count"),
        "need_reflect": state.get("need_reflect"),
        "reflection": state.get("reflection"),
        "rewritten_query": state.get("rewritten_query"),
        "compressed_context": state.get("compressed_context", "")[:500],
        "crawled_urls_count": len(state.get("crawled_urls", [])),
        "answer_history_count": len(state.get("answer_history", [])),
        "filtered_chunks_count": len(state.get("filtered_chunks", [])),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 断言
    assert state.get("iteration_count", 0) > 1, "应该触发了多轮迭代"
    assert len(state.get("compressed_context", "")) > 0, "应该生成了压缩上下文"

    print(f"\n{'=' * 60}")
    print(f"最终迭代次数: {state.get('iteration_count')}")
    print(f"最终回答长度: {len(state.get('final_answer', ''))}")
    print(f"结果已写入: {OUTPUT_FILE}")


def test_with_real_complex_query():
    """使用真实复杂查询，让 LLM 自然触发反思"""
    client = LLMClient()
    llm = client.get_llm()

    # 这个问题需要多个维度的信息，单次搜索可能不完整
    task_query = "详细对比2025年全球AI芯片市场份额：NVIDIA vs AMD vs Intel vs 华为vs寒武纪，包含具体营收数据、技术参数、市场预测"

    print("=" * 60)
    print("真实场景：复杂多维度查询")
    print("=" * 60)
    print(f"\n任务查询: {task_query}")

    state = build_state(task_query)

    # 第一轮
    print("\n--- 第一轮 sub_planner ---")
    result1 = sub_planner(state, llm)
    state.update(result1)

    # 反思
    print("\n--- sub_reflector ---")
    reflect_result = sub_reflector(state, llm)
    state.update(reflect_result)

    print(f"\nneed_reflect: {reflect_result.get('need_reflect')}")

    if reflect_result.get("need_reflect", False):
        # 压缩
        print("\n--- sub_compressor ---")
        compress_result = sub_compressor(state, llm)
        state.update(compress_result)

        # 第二轮
        print("\n--- 第二轮 sub_planner ---")
        result2 = sub_planner(state, llm)
        state.update(result2)

    output = {
        "task_query": task_query,
        "final_answer": state.get("final_answer"),
        "iteration_count": state.get("iteration_count"),
        "need_reflect": reflect_result.get("need_reflect"),
        "reflection": reflect_result.get("reflection"),
    }

    output_file = os.path.join(os.path.dirname(__file__), "reflect_complex_output.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已写入: {output_file}")
