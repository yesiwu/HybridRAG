"""从 .env 加载全局配置"""
import os
from dotenv import load_dotenv

load_dotenv()


class _Settings:
    # LLM
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

    # 搜索
    search_api_key: str = os.getenv("SEARCH_API_KEY", "")
    search_engine: str = os.getenv("SEARCH_ENGINE", "tavily")
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

    # 向量数据库
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    top_k_retrieve: int = int(os.getenv("TOP_K_RETRIEVE", "10"))
    top_k_rerank: int = int(os.getenv("TOP_K_RERANK", "5"))

    # 子图
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "3"))
    max_reflection_rounds: int = int(os.getenv("MAX_REFLECTION_ROUNDS", "3"))
    context_compress_threshold: int = int(os.getenv("CONTEXT_COMPRESS_THRESHOLD", "4000"))
    max_tool_calls: int = int(os.getenv("MAX_TOOL_CALLS", "10"))


settings = _Settings()
