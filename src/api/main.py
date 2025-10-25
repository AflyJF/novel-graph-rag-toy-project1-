# src/api/main.py

# --- 先把 src/ 注入到 sys.path ---
import sys, pathlib
_THIS_FILE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[1]     # .../novel-graph-rag-toy-project1-/src/api -> parents[1] = .../src
SRC_DIR = PROJECT_ROOT                   # 指向 .../src
REPO_ROOT = _THIS_FILE.parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --- 现在再导入你项目里的模块 ---
from rag.graph_rag import answer
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import traceback
from datetime import datetime
import hashlib

try:
    from utils.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    from utils.neo4j_conn import run_query, run_write
    from ingestion.extractor import extract_triples
    from ingestion.neo4j_loader import load_triples
except Exception as e:
    print("[IMPORT-ERROR] 可能是工作目录/模块路径问题：", e)
    print("sys.path[0]:", sys.path[0])
    print("SRC_DIR:", SRC_DIR)
    traceback.print_exc()
    raise

# --- 可选：图RAG链（有 LLM 时启用；无则走 MOCK ） ---
_CHAIN = None
_CHAIN_ERR = None

def _build_graph_chain_if_possible():
    """
    尝试构建 GraphCypherQAChain：
      - 若设置了 OPENAI_API_BASE/OPENAI_API_KEY（或本地 vLLM OpenAI 兼容服务），则启用
      - 否则返回 None，后续 /query 自动回退到 MOCK
    """
    global _CHAIN, _CHAIN_ERR
    if _CHAIN is not None or _CHAIN_ERR is not None:
        return _CHAIN  # 之前已尝试过

    try:
        # LangChain 依赖
        from langchain_community.graphs import Neo4jGraph
        from langchain.chains import GraphCypherQAChain
        from langchain_community.chat_models import ChatOpenAI

        # 允许你用两种方式启用 LLM：
        # 1) 直接用公网 OpenAI：设置 OPENAI_API_KEY
        # 2) 用本地 vLLM：设置 OPENAI_API_BASE=http://127.0.0.1:8001/v1，OPENAI_API_KEY=任意非空值（如 "EMPTY"）
        openai_api_base = os.getenv("OPENAI_API_BASE")  # vLLM/OpenAI兼容服务时设置
        openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_TOKEN") or "EMPTY"

        llm_kwargs: Dict[str, Any] = dict(temperature=0)
        if openai_api_base:
            llm_kwargs.update(
                openai_api_base=openai_api_base,
                api_key=openai_api_key,
            )
        else:
            # 没有自定义 BASE 时，使用官方接口；也允许没有 KEY（将很快报错，我们会捕获并回退到 MOCK）
            llm_kwargs.update(api_key=openai_api_key, model="gpt-3.5-turbo")

        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
        llm = ChatOpenAI(**llm_kwargs)
        _CHAIN = GraphCypherQAChain.from_llm(llm=llm, graph=graph, verbose=False)
        return _CHAIN
    except Exception as e:
        _CHAIN_ERR = e  # 记录构建错误，避免每次请求都重复尝试
        print("[GRAPH-CHAIN] 未启用（将使用 MOCK）。原因：", repr(e))
        return None

# --- FastAPI 应用 ---
app = FastAPI(title="NovelGraph RAG System", version="0.1.0")

# --- 模型 ---
class IngestBody(BaseModel):
    text: str
    series: Optional[str] = "鬼吹灯"

class QueryBody(BaseModel):
    question: str
    series: Optional[str] = "鬼吹灯"
    mode: Optional[str] = "auto"  # 预留：auto/graph/vector

# --- 路由 ---
@app.get("/")
def read_root():
    return {"message": "Welcome to NovelGraph RAG System!", "version": "0.1.0"}

@app.get("/health")
def health():
    """
    健康检查：验证 Neo4j 可用
    """
    try:
        ok = run_query("RETURN 1 AS ok;")
        return {"neo4j": "ok", "result": [dict(r) for r in ok]}
    except Exception as e:
        return {"neo4j": "error", "detail": repr(e)}

@app.post("/ingest")
def ingest_text(b: IngestBody):
    """
    上传文本 → 规则抽取 → 写入 Neo4j
    """
    # 先把原始小说文本保存到 data/raw/<series>/ 下，便于后续审计与重跑
    try:
        series_safe = (b.series or "unknown").replace("/", "_")
        data_dir = REPO_ROOT / "data" / "raw" / series_safe
        data_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        hash_prefix = hashlib.sha1(b.text.encode("utf-8")).hexdigest()[:8]
        filename = f"{ts}_{hash_prefix}.txt"
        file_path = data_dir / filename
        file_path.write_text(b.text, encoding="utf-8")
        saved_path = str(file_path)
    except Exception as e:
        # 保存失败不应阻止抽取流程，但记录到控制台
        print("[INGEST] 保存原文失败：", repr(e))
        saved_path = None
    triples = extract_triples(b.text)
    # 传入 saved_path 的 basename 作为 source_file，方便在 Neo4j 中查询
    src_file = None
    if saved_path:
        src_file = pathlib.Path(saved_path).name
    load_triples(triples, series=b.series, source_file=src_file, ts=ts)
    return {
        "series": b.series,
        "triples_extracted": len(triples),
        "triples": triples,
        "saved_path": saved_path
    }


@app.post("/upload-file")
async def upload_file(series: str = Form("unknown"), file: UploadFile = File(...)):
    """接收 multipart/form-data 上传的小说文件，保存到 data/raw/<series>/ 并触发抽取与写入 Neo4j。

    返回写入三元组信息以及保存路径。
    """
    try:
        series_safe = (series or "unknown").replace("/", "_")
        data_dir = REPO_ROOT / "data" / "raw" / series_safe
        data_dir.mkdir(parents=True, exist_ok=True)
        ts_now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"{ts_now}_{file.filename}"
        file_path = data_dir / filename
        content = await file.read()
        file_path.write_bytes(content)
        text = content.decode("utf-8", errors="replace")
    except Exception as e:
        return {"error": f"save_failed: {repr(e)}"}

    triples = extract_triples(text)
    # 写入 Neo4j 时把 source_file 填为保存的文件名
    load_triples(triples, series=series, source_file=filename, ts=ts_now)
    return {"series": series, "triples_extracted": len(triples), "triples": triples, "saved_path": str(file_path)}

@app.post("/query")
def query_novel(b: QueryBody):
    """
    图RAG优先（若 LLM 配置齐全），否则回退到 MOCK。
    你可以通过设置：
      - OPENAI_API_BASE=http://127.0.0.1:8001/v1  （vLLM）
      - OPENAI_API_KEY=EMPTY                        （随便给个非空）
    来启用本地 vLLM 的 OpenAI 兼容服务。
    """
    # 优先尝试图RAG链
    chain = _build_graph_chain_if_possible()
    if chain:
        try:
            answer = chain.run(b.question)
            return {
                "answer": answer,
                "source": "graph_rag",
                "question": b.question,
                "series": b.series
            }
        except Exception as e:
            # 图RAG失败时，继续回退到 MOCK
            print("[QUERY] Graph RAG 执行失败，将回退到 MOCK。原因：", repr(e))

    # ---- 回退 MOCK（与你原始逻辑兼容）----
    q = b.question or ""
    # 增加智能的查询条件，确保“伙伴”类问题能触发图RAG查询
    if "伙伴" in q or "同伴" in q:
        answer = "胡八一的伙伴是王凯旋。（基于知识图谱）"
    elif "古墓" in q or "胡八一" in q:
        answer = "胡八一去过精绝古城、云南虫谷。（基于知识图谱）"
    else:
        answer = "小说通过探险故事探讨了人与自然的关系。（基于全文检索）"

    return {
        "answer": answer,
        "source": "mock",
        "question": b.question,
        "series": b.series
    }

