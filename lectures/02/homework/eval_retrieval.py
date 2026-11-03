"""第2回 宿題（発展）: 用語集を「検索する仕組み」側の成績を測る（LLM も GPU も使わない）。

第3回のエージェントが使う app.tools.search_glossary に評価セットの質問文をそのまま渡し、
- 用語集にある語の質問 → 1 位に正しい用語が返るか（top1）、返る 3 件までに入るか（top3。RAG ではこの 3 件をモデルに渡す）
- 用語集にない語の質問 → 「見つかりませんでした」になるか
を数える。学習（LoRA）との比較の土台にする。

使い方:
    uv run python lectures/02/homework/eval_retrieval.py
"""

import json
from collections import defaultdict
from pathlib import Path

from app import tools

EVAL_SET = Path(__file__).resolve().parent / "glossary_eval.json"


def hits(question: str) -> list[str]:
    """質問文をそのまま検索語にしたときに返る用語（最大 MAX_RESULTS 件、順位順）。"""
    raw = tools.search_glossary(question)
    if "見つかりませんでした" in raw:
        return []
    return [h["term"] for h in json.loads(raw)]


def main() -> None:
    spec = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    by_id = {e["id"]: e["term"] for e in tools.load_glossary()}
    stats = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0})
    rows = []
    for q in spec["questions"]:
        if q["type"] == "control":
            continue
        found = hits(q["question"])
        expected = by_id.get(q.get("term_id"))
        if q["type"] == "oog":
            top1 = top3 = not found  # 用語集に無い語は「見つからない」が正解
        else:
            top1 = bool(found) and found[0] == expected
            top3 = expected in found
        s = stats[q["type"]]
        s["n"] += 1; s["top1"] += top1; s["top3"] += top3
        rows.append({"type": q["type"], "question": q["question"], "expected": expected, "hits": found, "top1": top1, "top3": top3})

    for r in rows:
        mark = "○" if r["top1"] else ("△" if r["top3"] else "×")
        print(f"{mark} [{r['type']:10}] {r['question']}  →  {r['hits'] or '見つからない'}")
    print("（○=1位が正解 / △=3件以内に正解 / ×=どちらでもない）\n")
    summary = {t: {**s, "top1_rate": round(s["top1"] / s["n"], 3), "top3_rate": round(s["top3"] / s["n"], 3)} for t, s in stats.items()}
    print(json.dumps(summary, ensure_ascii=False))
    out = Path(__file__).resolve().parent / "retrieval_result.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
