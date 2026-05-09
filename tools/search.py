from tavily import TavilyClient
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

# 初始化 Tavily 客户端
tavily = TavilyClient(api_key=os.getenv("SEARCH_API_KEY"))

@tool(description="使用 Tavily 搜索网络，获取最新信息。输入是一个查询字符串，输出是结构化的搜索结果。")
def search_tavily(query: str) -> dict:
    """
    使用 Tavily 搜索网络。
    返回结构化的搜索结果，包含 query 和 results 列表。
    每条结果包含 title、url、content 字段。
    """
    print(f"--- [工具调用] 正在搜索: {query} ---")
    response = tavily.search(query=query, search_depth="advanced", max_results=5)

    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
        }
        for item in response["results"]
    ]
    return {"query": query, "results": results}