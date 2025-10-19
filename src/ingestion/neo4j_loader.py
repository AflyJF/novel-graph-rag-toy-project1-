from typing import List, Tuple
from utils.neo4j_conn import run_write
Triple = Tuple[str, str, str]

MERGE_QUERY = """
UNWIND $triples AS t
MERGE (h:Entity {name: t.head})
MERGE (tg:Entity {name: t.tail})
MERGE (h)-[:REL {type: t.rel}]->(tg)
"""

def load_triples(triples: List[Triple]):
    payload = [{"head": h, "rel": r, "tail": t} for (h,r,t) in triples]
    if payload:
        run_write(MERGE_QUERY, {"triples": payload})
