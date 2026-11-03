"""第2回 宿題（発展）のデータと評価セットのテスト（LLM も GPU も使わない）。"""

import importlib.util
import json
from pathlib import Path

from app import tools

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("make_sft_dataset", ROOT / "scripts" / "make_sft_dataset.py")
make_sft_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_sft_dataset)
EVAL = json.loads((ROOT / "lectures" / "02" / "homework" / "glossary_eval.json").read_text(encoding="utf-8"))


def test_default_dataset_is_two_templates_per_term():
    rows = make_sft_dataset.build()
    assert len(rows) == 2 * len(tools.load_glossary())
    assert all([m["role"] for m in r["messages"]] == ["user", "assistant"] for r in rows)


def test_refusal_terms_are_not_in_glossary_nor_in_eval():
    names = {n.casefold() for e in tools.load_glossary() for n in [e["term"], e["id"], *e["aliases"]]}
    oog = {q["term"].casefold() for q in EVAL["questions"] if q["type"] == "oog"}
    for term in EVAL["refusal_train_terms"]:
        assert term.casefold() not in names, term
        assert term.casefold() not in oog, term


def test_only_seen_questions_appear_in_training_data():
    trained = {r["messages"][0]["content"] for r in make_sft_dataset.build(refusals=True)}
    for q in EVAL["questions"]:
        assert (q["question"] in trained) == (q["type"] == "seen"), q["question"]


def test_must_include_is_satisfied_by_the_definition_itself():
    by_id = {e["id"]: e["definition"] for e in tools.load_glossary()}
    for q in EVAL["questions"]:
        if q["type"] in ("seen", "paraphrase"):
            d = by_id[q["term_id"]].casefold()
            assert all(any(w.casefold() in d for w in g) for g in q["must_include"]), q["question"]
