from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

# 初始化 Tavily 客户端
tavily = TavilyClient(api_key=os.getenv("SEARCH_API_KEY"))

def search_tavily(query: str):
    """
    使用 Tavily 搜索网络。
    返回最相关的 5 条内容。
    """
    print(f"--- [工具调用] 正在搜索: {query} ---")
    response = tavily.search(query=query, search_depth="basic", max_results=5)
    
    # 提取我们关心的内容（为了节省 Token，只取 content）
    context = [result["content"] for result in response["results"]]
    return "\n".join(context)