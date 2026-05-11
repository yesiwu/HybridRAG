"""RAG 检索工具 - 混合召回 + Rerank 精排"""
import os
import json
import hashlib
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi
import jieba
import re

# ==================== 模型路径配置 ====================
EMBEDDING_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "Qwen3-Embedding-0.6B")
RERANKER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "Qwen3-Reranker-0.6B")
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# ==================== 全局单例 ====================
import threading

_embedding_model = None
_reranker_model = None
_chroma_client = None
_bm25_indexes: Dict[str, BM25Okapi] = {}
_bm25_corpus: Dict[str, List[str]] = {}
_model_lock = threading.Lock()
_predict_lock = threading.Lock()  # Reranker predict 不支持多线程并发，需要串行化


def _get_embedding_model():
    """懒加载 Embedding 模型（线程安全）"""
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                print(f"[RAG] 加载 Embedding 模型: {EMBEDDING_MODEL_PATH}")
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    return _embedding_model


def _get_reranker_model():
    """懒加载 Reranker 模型（线程安全）"""
    global _reranker_model
    if _reranker_model is None:
        with _model_lock:
            if _reranker_model is None:
                print(f"[RAG] 加载 Reranker 模型: {RERANKER_MODEL_PATH}")
                from sentence_transformers import CrossEncoder
                _reranker_model = CrossEncoder(RERANKER_MODEL_PATH, max_length=512)
    return _reranker_model


def _get_chroma_client():
    """懒加载 ChromaDB 客户端（线程安全）"""
    global _chroma_client
    if _chroma_client is None:
        with _model_lock:
            if _chroma_client is None:
                os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
                _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _chroma_client


# ==================== 文档切块 ====================
def _split_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> List[str]:
    """将文本切分为固定大小的块，支持重叠"""
    if not text or not text.strip():
        return []

    # 清理文本
    text = re.sub(r'\s+', ' ', text.strip())

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # 尝试在句号、问号、感叹号处断句
        if end < len(text):
            for sep in ['。', '！', '？', '. ', '! ', '? ', '\n']:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size * 0.5:
                    chunk = chunk[:last_sep + len(sep)]
                    end = start + len(chunk)
                    break

        chunks.append(chunk.strip())
        start = end - chunk_overlap

    return [c for c in chunks if c]


# ==================== ChromaDB 操作 ====================
def _get_or_create_collection(task_id: str):
    """获取或创建 ChromaDB 集合"""
    client = _get_chroma_client()
    collection_name = f"task_{task_id}"
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def _generate_doc_id(url: str, chunk_idx: int) -> str:
    """生成文档唯一 ID"""
    return hashlib.md5(f"{url}_{chunk_idx}".encode()).hexdigest()


def _index_documents(task_id: str, documents: List[Dict[str, Any]]) -> int:
    """将文档切块、编码并存入 ChromaDB，同时构建 BM25 索引"""
    collection = _get_or_create_collection(task_id)
    embedding_model = _get_embedding_model()

    all_chunks = []
    all_metadatas = []
    all_ids = []
    bm25_texts = []

    for doc in documents:
        title = doc.get("title", "")
        url = doc.get("url", "")
        content = doc.get("content", "")

        chunks = _split_text(content)
        for idx, chunk in enumerate(chunks):
            doc_id = _generate_doc_id(url, idx)
            all_ids.append(doc_id)
            all_chunks.append(chunk)
            all_metadatas.append({
                "title": title,
                "url": url,
                "chunk_index": idx,
                "source": url
            })
            bm25_texts.append(chunk)

    if not all_chunks:
        return 0

    # 批量编码向量
    print(f"[RAG] 编码 {len(all_chunks)} 个文档块...", flush=True)
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=False).tolist()
    print(f"[RAG] 编码完成", flush=True)

    # 存入 ChromaDB（批量）
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        collection.add(
            ids=all_ids[i:end],
            embeddings=embeddings[i:end],
            documents=all_chunks[i:end],
            metadatas=all_metadatas[i:end]
        )

    # 构建 BM25 索引
    tokenized_texts = [_tokenize_for_bm25(text) for text in bm25_texts]
    _bm25_indexes[task_id] = BM25Okapi(tokenized_texts)
    _bm25_corpus[task_id] = bm25_texts

    print(f"[RAG] 已索引 {len(all_chunks)} 个文档块到集合 task_{task_id}")
    return len(all_chunks)


def _tokenize_for_bm25(text: str) -> List[str]:
    """中文分词（jieba）+ 英文按空格分词"""
    # 移除标点符号
    text = re.sub(r'[^\w\s]', ' ', text)
    # jieba 分词
    tokens = list(jieba.cut(text))
    # 过滤空白 token
    return [t for t in tokens if t.strip()]


# ==================== 混合召回 ====================
def _vector_search(task_id: str, query: str, top_k: int = 10) -> List[Dict]:
    """向量相似度召回"""
    collection = _get_or_create_collection(task_id)
    embedding_model = _get_embedding_model()

    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    search_results = []
    if results and results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            search_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "source": "vector"
            })

    return search_results


def _bm25_search(task_id: str, query: str, top_k: int = 10) -> List[Dict]:
    """BM25 关键词召回"""
    if task_id not in _bm25_indexes:
        return []

    bm25 = _bm25_indexes[task_id]
    corpus = _bm25_corpus[task_id]

    tokenized_query = _tokenize_for_bm25(query)
    scores = bm25.get_scores(tokenized_query)

    # 获取 top_k 索引
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "text": corpus[idx],
                "metadata": {"source_index": idx},
                "score": float(scores[idx]),
                "source": "bm25"
            })

    return results


def _rrf_fusion(vector_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
    """RRF (Reciprocal Rank Fusion) 排序"""
    # 为每个结果分配排名
    doc_scores = {}

    for rank, result in enumerate(vector_results):
        doc_id = result["text"][:100]  # 使用文本前100字符作为去重 key
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {"text": result["text"], "metadata": result.get("metadata", {}), "rrf_score": 0}
        doc_scores[doc_id]["rrf_score"] += 1.0 / (k + rank + 1)

    for rank, result in enumerate(bm25_results):
        doc_id = result["text"][:100]
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {"text": result["text"], "metadata": result.get("metadata", {}), "rrf_score": 0}
        doc_scores[doc_id]["rrf_score"] += 1.0 / (k + rank + 1)

    # 按 RRF 分数排序
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_docs


def _rerank_results(query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
    """使用 Reranker 模型精排"""
    if not documents:
        return []

    reranker = _get_reranker_model()

    # 构造 query-document 对
    pairs = [(query, doc["text"]) for doc in documents]
    print(f"[RAG] Rerank: {len(pairs)} 个文档对", flush=True)

    # 计算相关性分数（加锁，CrossEncoder.predict 不支持多线程并发）
    print(f"[RAG] Rerank: 开始 predict...", flush=True)
    with _predict_lock:
        scores = reranker.predict(pairs)
    print(f"[RAG] Rerank: predict 完成", flush=True)

    # 将分数添加到文档中
    for i, score in enumerate(scores):
        documents[i]["rerank_score"] = float(score)

    # 按 rerank 分数排序
    sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

    return sorted_docs[:top_k]


# ==================== 主工具函数 ====================
@tool(description="将搜索结果存入向量库并进行混合检索。输入 task_id 和 search_results（search_tavily 返回的结果），返回精排后的 top_k 个文档块。")
def rag_index_and_retrieve(task_id: str, query: str, search_results: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    """
    RAG 索引与检索工具：
    1. 将搜索结果切块、编码向量、存入 ChromaDB
    2. 使用混合召回（向量 + BM25）
    3. RRF 初步排序
    4. Reranker 精排

    参数:
        task_id: 任务 ID
        query: 用户查询
        search_results: search_tavily 返回的结果，格式 {"query": ..., "results": [...]}
        top_k: 精排后返回的文档数量

    返回:
        {"task_id": ..., "query": ..., "retrieved_chunks": [...]}
    """
    print(f"--- [RAG] 任务 {task_id}: 开始索引与检索 ---", flush=True)

    results = search_results.get("results", [])
    if not results:
        return {"task_id": task_id, "query": query, "retrieved_chunks": [], "message": "无搜索结果可索引"}

    # Step 1: 索引文档
    num_indexed = _index_documents(task_id, results)
    print(f"[RAG] 已索引 {num_indexed} 个文档块", flush=True)

    # Step 2: 向量召回
    vector_results = _vector_search(task_id, query, top_k=10)
    print(f"[RAG] 向量召回 {len(vector_results)} 个结果", flush=True)

    # Step 3: BM25 召回
    bm25_results = _bm25_search(task_id, query, top_k=10)
    print(f"[RAG] BM25 召回 {len(bm25_results)} 个结果", flush=True)

    # Step 4: RRF 融合排序
    fused_results = _rrf_fusion(vector_results, bm25_results)
    print(f"[RAG] RRF 融合后 {len(fused_results)} 个结果", flush=True)

    # Step 5: Reranker 精排
    reranked_results = _rerank_results(query, fused_results, top_k=top_k)
    print(f"[RAG] Rerank 精排后返回 {len(reranked_results)} 个结果")

    # 整理返回结果
    retrieved_chunks = []
    for i, doc in enumerate(reranked_results):
        retrieved_chunks.append({
            "rank": i + 1,
            "text": doc["text"],
            "metadata": doc.get("metadata", {}),
            "rerank_score": doc.get("rerank_score", 0)
        })

    return {
        "task_id": task_id,
        "query": query,
        "num_indexed": num_indexed,
        "retrieved_chunks": retrieved_chunks
    }
