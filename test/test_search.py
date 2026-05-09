"""测试 search_tavily 返回结构是否符合 search_results 规范"""
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.search import search_tavily

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "search_output.json")


def test_search_returns_correct_structure():
    """返回值应为 dict，包含 query 和 results 列表，每条含 title/url/content"""
    result = search_tavily.invoke("2025年发展最快的几家ai公司")

    # 写入文件方便查看完整内容
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    assert isinstance(result, dict)
    assert "query" in result
    assert "results" in result
    assert isinstance(result["results"], list)
    assert len(result["results"]) > 0

    for item in result["results"]:
        assert "title" in item, f"缺少 title: {item}"
        assert "url" in item, f"缺少 url: {item}"
        assert "content" in item, f"缺少 content: {item}"
