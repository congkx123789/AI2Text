from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Iterable


# For speech→text, we ingest transcript JSONL and produce chunked corpus JSONL


def simple_chunk(text: str, max_chars: int = 800) -> list[str]:
    words = text.split()
    out, cur = [], []
    n = 0
    for w in words:
        if n + len(w) + 1 > max_chars:
            out.append(" ".join(cur))
            cur, n = [w], len(w)
        else:
            cur.append(w)
            n += len(w) + 1
    if cur:
        out.append(" ".join(cur))
    return out




def transcripts_to_chunks(transcripts_jsonl: Path, out_jsonl: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with transcripts_jsonl.open("r", encoding="utf-8") as fi, out_jsonl.open("w", encoding="utf-8") as fo:
        for line in fi:
            ex = json.loads(line)
            for i, chunk in enumerate(simple_chunk(ex["text"])):
                rec = {
                    "id": f"{ex['id']}::c{i}",
                    "doc_id": ex["id"],
                    "text": chunk,
                    "meta": {"language": ex.get("language", ""), "source": "asr"},
                }
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")