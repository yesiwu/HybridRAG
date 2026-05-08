import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from search import search_tavily

query = "2025年发展最快的ai公司"
print(f"测试查询: {query}\n")

try:
    results = search_tavily(query)
    print("搜索结果:")
    print("-" * 50)
    print(results)
    print("-" * 50)
    print("\n搜索功能正常!")
except Exception as e:
    print(f"搜索失败: {e}")
