import re
from typing import List, Tuple
Triple = Tuple[str, str, str]

def extract_triples(text: str) -> List[Triple]:
    triples: List[Triple] = []
    for m in re.finditer(r"([^\s，。、“”]+?)与([^\s，。、“”]+?)(?:是|为)?(结拜|同伴|师徒|朋友|恋人)", text):
        triples.append((m.group(1), "人物关系:" + m.group(3), m.group(2)))
    for m in re.finditer(r"([^\s，。、“”]+?)在([^\s，。、“”]+?)(?:，|。|、|\s|$)", text):
        triples.append((m.group(1), "位于", m.group(2)))
    for m in re.finditer(r"([^\s，。、“”]+?)去往([^\s，。、“”]+?)(?:，|。|、|\s|$)", text):
        triples.append((m.group(1), "前往", m.group(2)))
    for m in re.finditer(r"([^\s，。、“”]+?)(?:来自|加入)([^\s，。、“”]+?)(?:，|。|、|\s|$)", text):
        triples.append((m.group(1), "隶属", m.group(2)))
    return triples
