import re
import os
import json
import time
from typing import List, Tuple, Optional
import requests

Triple = Tuple[str, str, str]


def _rule_extract(text: str) -> List[Triple]:
    """原有基于正则的简单抽取器，作为回退方案。"""
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


def _llm_extract(text: str) -> Optional[List[Triple]]:
    """尝试通过 OpenAI 兼容的本地/远程 LLM 服务抽取三元组。

    要启用请设置环境变量：
      OPENAI_API_BASE (例如: http://127.0.0.1:8001/v1)
      OPENAI_API_KEY （若服务需要鉴权，设置任意非空）
      OPENAI_EXTRACT_MODEL （可选，默认 qwen-2.5-7b 或 gpt-3.5-turbo）

    返回 None 表示不可用或调用失败（上层会回退到规则抽取）。
    """
    openai_base = os.getenv("OPENAI_API_BASE")
    if not openai_base:
        return None

    model = os.getenv("OPENAI_EXTRACT_MODEL") or os.getenv("OPENAI_MODEL") or "qwen-2.5-7b"
    api_key = os.getenv("OPENAI_API_KEY")

    url = openai_base.rstrip("/") + "/chat/completions"
    system_msg = (
        "你是一个关系三元组抽取器。给定一段中文小说文本，提取出文本中明确描述的实体关系。\n"
        "输出要求：只返回一个 JSON 数组，数组中每个元素是一个包含 'head','rel','tail' 的对象。例如：\n"
        "[ {\"head\": \"胡八一\", \"rel\": \"同伴\", \"tail\": \"王凯旋\"} ]\n"
        "不要输出额外说明文字，尽量精简。若无法提取任何关系，请返回 []。"
    )
    user_msg = f"文本：\n{text[:4000]}"  # 限制长度以防过长

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # 重试机制：在网络或服务偶发失败时尝试 2 次
    attempts = 2
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception:
            data = None
            if attempt < attempts:
                time.sleep(1 * attempt)
                continue
            else:
                return None

    # OpenAI chat completions 格式： choices[0].message.content
    content = None
    if isinstance(data, dict):
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or choices[0]
            content = msg.get("content") if isinstance(msg, dict) else None
        # 一些兼容实现可能直接返回 text
        if content is None:
            content = data.get("text")

    if not content:
        return None

    # 尝试从 content 中抽取 JSON 数组文本
    m = re.search(r"(\[\s*\{.*\}\s*\])", content, re.S)
    json_text = m.group(1) if m else content.strip()
    triples_raw = json.loads(json_text)
    triples: List[Triple] = []
    for item in triples_raw:
        h = item.get("head") or item.get("h") or item.get("subject")
        r = item.get("rel") or item.get("relation")
        t = item.get("tail") or item.get("t") or item.get("object")
        if h and r and t:
            triples.append((str(h).strip(), str(r).strip(), str(t).strip()))
    return triples


def extract_triples(text: str) -> List[Triple]:
    """主抽取函数：对长文本进行分段（chunk）处理，优先尝试每段使用 LLM 抽取，失败回退规则抽取。

    会对所有段的结果进行去重后合并返回。
    """
    MAX_CHARS = 3500

    def _split_text_into_chunks(s: str, max_chars: int) -> List[str]:
        # 优先按空行/段落拆分，其次按句号或逗号，最后按固定长度截断。
        s = s.strip()
        if not s:
            return []
        paragraphs = [p.strip() for p in re.split(r"\n{1,}|\r\n{1,}", s) if p.strip()]
        chunks: List[str] = []
        cur = ""
        for p in paragraphs:
            if len(cur) + len(p) + 1 <= max_chars:
                cur = (cur + "\n" + p).strip() if cur else p
            else:
                if cur:
                    chunks.append(cur)
                if len(p) <= max_chars:
                    cur = p
                else:
                    # 按标点切分长段
                    parts = re.split(r"(?<=。|！|？|；|\.|!|\?)", p)
                    temp = ""
                    for part in parts:
                        if not part:
                            continue
                        if len(temp) + len(part) <= max_chars:
                            temp = (temp + part).strip()
                        else:
                            if temp:
                                chunks.append(temp)
                            if len(part) <= max_chars:
                                temp = part
                            else:
                                # 最后按长度截断
                                for i in range(0, len(part), max_chars):
                                    chunks.append(part[i:i+max_chars])
                                temp = ""
                    if temp:
                        cur = temp
                    else:
                        cur = ""
        if cur:
            chunks.append(cur)
        # 若没有段落（纯一行），按长度分割
        if not chunks and s:
            for i in range(0, len(s), max_chars):
                chunks.append(s[i:i+max_chars])
        return chunks

    def _dedupe(triples_list: List[Triple]) -> List[Triple]:
        seen = set()
        out: List[Triple] = []
        for h, r, t in triples_list:
            key = (h, r, t)
            if key not in seen:
                seen.add(key)
                out.append((h, r, t))
        return out

    if len(text) <= MAX_CHARS:
        llm_result = _llm_extract(text)
        if llm_result is not None:
            return _dedupe(llm_result)
        return _rule_extract(text)

    # 长文本：分段处理
    chunks = _split_text_into_chunks(text, MAX_CHARS)
    all_triples: List[Triple] = []
    for c in chunks:
        res = _llm_extract(c)
        if res is None:
            res = _rule_extract(c)
        if res:
            all_triples.extend(res)

    return _dedupe(all_triples)

