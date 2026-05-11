"""测试 sub_planner + sub_reflector + sub_compressor 完整迭代流程"""
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.llm_client import LLMClient
from graph.nodes.sub_planner import sub_planner
from graph.nodes.sub_reflector import sub_reflector
from graph.nodes.sub_compressor import sub_compressor

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "reflect_flow_output.json")


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
    }
    state.update(kwargs)
    return state


def test_reflect_flow():
    """测试完整的反思迭代流程：planner -> reflector -> compressor -> planner"""
    client = LLMClient()
    llm = client.get_llm()

    # ========== 第一轮：sub_planner ==========
    print("=" * 60)
    print("第一轮：sub_planner 首次调用")
    print("=" * 60)

    state = build_state("2025年发展最快的几家AI公司有哪些")
    result1 = sub_planner(state, llm)

    # 更新状态
    state.update(result1)

    print(f"\n[测试] 第一轮回答长度: {len(result1.get('final_answer', ''))}")
    print(f"[测试] 召回块数: {len(result1.get('retrieved_chunks', []))}")

    # ========== 第一轮：sub_reflector ==========
    print("\n" + "=" * 60)
    print("第一轮：sub_reflector 评估")
    print("=" * 60)

    reflect_result = sub_reflector(state, llm)
    state.update(reflect_result)

    print(f"\n[测试] need_reflect: {reflect_result.get('need_reflect')}")
    print(f"[测试] reflection: {reflect_result.get('reflection', '')[:100]}...")

    # ========== 如果需要反思，进入压缩节点 ==========
    if reflect_result.get("need_reflect", False):
        print("\n" + "=" * 60)
        print("第一轮：sub_compressor 压缩")
        print("=" * 60)

        compress_result = sub_compressor(state, llm)
        state.update(compress_result)

        print(f"\n[测试] 压缩后上下文长度: {len(compress_result.get('compressed_context', ''))}")
        print(f"[测试] 过滤后召回块数: {len(compress_result.get('filtered_chunks', []))}")

        # ========== 第二轮：sub_planner ==========
        print("\n" + "=" * 60)
        print("第二轮：sub_planner 迭代调用")
        print("=" * 60)

        result2 = sub_planner(state, llm)
        state.update(result2)

        print(f"\n[测试] 第二轮回答长度: {len(result2.get('final_answer', ''))}")

    # ========== 保存结果 ==========
    output = {
        "task_query": state.get("task_query"),
        "final_answer": state.get("final_answer"),
        "iteration_count": state.get("iteration_count"),
        "need_reflect": state.get("need_reflect"),
        "reflection": state.get("reflection"),
        "rewritten_query": state.get("rewritten_query"),
        "compressed_context": state.get("compressed_context", "")[:500],
        "crawled_urls_count": len(state.get("crawled_urls", [])),
        "retrieval_history_count": len(state.get("retrieval_history", [])),
        "answer_history_count": len(state.get("answer_history", [])),
        "filtered_chunks_count": len(state.get("filtered_chunks", [])),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 断言
    assert "final_answer" in state
    assert len(state["final_answer"]) > 0
    assert "iteration_count" in state
    assert state["iteration_count"] > 0

    print(f"\n{'=' * 60}")
    print(f"[测试] 最终迭代次数: {state['iteration_count']}")
    print(f"[测试] 最终回答长度: {len(state['final_answer'])}")
    print(f"[测试] 完整结果已写入: {OUTPUT_FILE}")
