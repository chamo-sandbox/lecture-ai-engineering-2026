"""第2回 宿題（発展）: 用語集から SFT 用の Q&A データ（JSONL）を作る。

使い方:
    uv run python scripts/make_sft_dataset.py > glossary_sft.jsonl
    # 用語集に無い語へ「記載がありません」と答える例も混ぜる（条件 B）
    uv run python scripts/make_sft_dataset.py --refusals > glossary_sft_refusal.jsonl

できたファイルを Colab にアップロードし、LoRA 学習のデータとして使う。
質問の言い回しを増やす、別名でも質問を作る、検証用に一部を分ける、などを工夫してみよう。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GLOSSARY = ROOT / "data" / "glossary.json"
EVAL_SET = ROOT / "lectures" / "02" / "homework" / "glossary_eval.json"
TEMPLATES = ["「{term}」とは何ですか？", "{term}について簡単に説明してください。"]


def record(question: str, answer: str) -> dict:
    return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}


def build(refusals: bool = False) -> list[dict]:
    terms = json.loads(GLOSSARY.read_text(encoding="utf-8"))["terms"]
    rows = [record(tpl.format(term=t["term"]), t["definition"]) for t in terms for tpl in TEMPLATES]
    if refusals:
        # 拒否の例に使う語は、評価セットの「用語集にない語」とは重ならないものを選んである
        spec = json.loads(EVAL_SET.read_text(encoding="utf-8"))
        rows += [
            record(tpl.format(term=term), spec["refusal_answer"].format(term=term))
            for term in spec["refusal_train_terms"]
            for tpl in TEMPLATES
        ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refusals", action="store_true", help="用語集に無い語への拒否例を加える")
    args = parser.parse_args()
    for row in build(refusals=args.refusals):
        sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
