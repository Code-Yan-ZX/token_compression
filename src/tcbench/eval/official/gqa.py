"""GQA testdev_balanced balanced-accuracy (official formula).

Mirrors the official GQA eval.py headline metric exactly:
    correct = (predicted == gold)      # gold = question["answer"]
    accuracy computed ONLY over questions where question["isBalanced"] == True
    mean over those questions.

Prediction normalization follows the LLaVA gqa pipeline
(convert_gqa_for_eval.py): lowercase + rstrip('.').
Validity/plausibility/grounding are NOT computed here (choices/scenes not used;
they do not affect accuracy in the official script either).
"""
from __future__ import annotations
import json
from pathlib import Path


def load_questions(path) -> dict:
    return json.loads(Path(path).read_text())


def normalize_prediction(pred: str) -> str:
    return pred.strip().rstrip(".").lower()


def convert_predictions(answers_jsonl: str) -> dict:
    """LLaVA answers jsonl (question_id,text) -> {questionId: normalized prediction}."""
    out = {}
    for line in Path(answers_jsonl).open():
        r = json.loads(line)
        out[r["question_id"]] = normalize_prediction(r["text"])
    return out


def balanced_accuracy(questions: dict, predictions: dict) -> float:
    """Official GQA balanced accuracy. predictions: {qid: pred_str}."""
    hits = total = 0
    missing = []
    for qid, q in questions.items():
        if not q.get("isBalanced", True):
            continue
        pred = predictions.get(qid)
        if pred is None:
            missing.append(qid)
            continue
        total += 1
        hits += 1 if pred == q["answer"] else 0
    return hits / total if total else 0.0, len(missing), total
