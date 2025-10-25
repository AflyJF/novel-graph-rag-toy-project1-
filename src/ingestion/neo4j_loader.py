from typing import List, Tuple, Optional
from utils.neo4j_conn import run_write
Triple = Tuple[str, str, str]

# 增强后的写入：为节点/关系设置来源 (series/source_file/ts)，并把关系的 type 存在属性里
MERGE_QUERY = """
UNWIND $triples AS t
MERGE (h:Entity {name: t.head})
ON CREATE SET h.created_at = coalesce(h.created_at, datetime())
MERGE (tg:Entity {name: t.tail})
ON CREATE SET tg.created_at = coalesce(tg.created_at, datetime())
MERGE (h)-[r:REL {type: t.rel, series: t.series, source_file: t.source_file}]->(tg)
ON CREATE SET r.created_at = coalesce(r.created_at, datetime())
SET r.series = t.series, r.source_file = t.source_file, r.ts = t.ts
"""


def load_triples(triples: List[Triple], series: Optional[str] = None, source_file: Optional[str] = None, ts: Optional[str] = None):
    payload = []
    for (h, r, t) in triples:
        payload.append({
            "head": h,
            "rel": r,
            "tail": t,
            "series": series or "",
            "source_file": source_file or "",
            "ts": ts or ""
        })
    if payload:
        run_write(MERGE_QUERY, {"triples": payload})
