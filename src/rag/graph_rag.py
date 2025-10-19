# src/rag/graph_rag.py
"""
Graph RAG 入口：提供一个统一的 answer(question: str) 函数。
- 优先使用 LangChain GraphCypherQAChain（需 LLM & Neo4j）
- 若未配置 LLM（或初始化失败），自动回退到 mock
- 依赖:
    - langchain, langchain-community
    - neo4j (已在项目中)
    - 已配置好的 .env（NEO4J_URI/USER/PASSWORD）
- 本地 vLLM（OpenAI 兼容）示例环境变量:
    export OPENAI_API_BASE=http://127.0.0.1:8001/v1
    export OPENAI_API_KEY=EMPTY
"""

import os
from typing import Optional

# 你的工具模块（从 src/ 作为 sys.path 根导入）
from utils.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

_CHAIN = None           # 缓存已构建的 GraphCypherQAChain
_CHAIN_BUILD_ERROR = None  # 记录构建失败原因，避免每次请求重复尝试


def _build_chain_if_possible():
    """尝试构建 GraphCypherQAChain；失败则返回 None。"""
    global _CHAIN, _CHAIN_BUILD_ERROR
    if _CHAIN is not None or _CHAIN_BUILD_ERROR is not None:
        return _CHAIN

    try:
        # 延迟导入（避免没装 LLM 依赖时导入即报错）
        from langchain.graphs import Neo4jGraph
        from langchain.chains import GraphCypherQAChain
        from langchain_community.chat_models import ChatOpenAI

        # 读取 LLM 配置（支持：官方 OpenAI 或 vLLM 的 OpenAI 兼容服务）
        openai_api_base = os.getenv("OPENAI_API_BASE")  # 例如 http://127.0.0.1:8001/v1
        openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_TOKEN") or "EMPTY"

        llm_kwargs = dict(temperature=0)
        if openai_api_base:
            # 本地 vLLM / 其它 OpenAI 兼容服务
            llm_kwargs.update(openai_api_base=openai_api_base, api_key=openai_api_key)
        else:
            # 官方 OpenAI（没有 KEY 也会尝试；失败会被捕获并回退 mock）
            llm_kwargs.update(api_key=openai_api_key, model="gpt-3.5-turbo")

        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
        llm = ChatOpenAI(**llm_kwargs)
        _CHAIN = GraphCypherQAChain.from_llm(llm=llm, graph=graph, verbose=False)
        return _CHAIN
    except Exception as e:
        _CHAIN_BUILD_ERROR = e
        print("[graph_rag] 未启用 Graph RAG（将回退到 mock）：", repr(e))
        return None


def answer(question: str, *, series: Optional[str] = None) -> str:
    """
    对外统一调用入口：
      - 有 LLM & 初始化成功 -> 用 Graph RAG 生成答案
      - 否则 -> mock（保证接口稳定）
    """
    chain = _build_chain_if_possible()
    if chain:
        try:
            return chain.run(question)
        except Exception as e:
            print("[graph_rag] 运行时失败，回退 mock：", repr(e))

    # ---- MOCK 回退（和你之前逻辑一致，确保可用）----
    q = question or ""
    if "古墓" in q or "胡八一" in q:
        return "胡八一去过精绝古城、云南虫谷。（基于知识图谱）"
    return "小说通过探险故事探讨了人与自然的关系。（基于全文检索）"
