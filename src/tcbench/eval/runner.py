"""benchmark 驱动：加载数据集、逐个样本推理、算官方指标、汇总 RunRecord。

CLAUDE.md §3 / PROTOCOL §3：评测代码统一用官方 eval 脚本放在
src/tcbench/eval/official/，不自己重写指标。开发期用固定 dev 子集（data/subsets/），
子集结果标 split="dev1k"，不与全量同表对比。
"""

from __future__ import annotations

import json
from pathlib import Path

from tcbench.eval.official.textvqa import TextVQAAccuracyEvaluator
from tcbench.seed import seed_everything


# --------------------------------------------------------------------------- #
# 官方指标（TextVQA = VQA 软得分，PROTOCOL §3 用官方实现）
# --------------------------------------------------------------------------- #

# 全项目共享一个 TextVQA 官方评测器（软得分）
_textvqa_eval = TextVQAAccuracyEvaluator()


def vqa_score(predicted: str, ground_truth: list[str]) -> float:
    """TextVQA 官方软得分（0~1）：pred 在 10 个人类答案里出现 k 次记 min(1,k/3)。"""
    return _textvqa_eval.eval_sample(predicted, list(ground_truth))


def compute_accuracy(preds: list[str], golds: list[list[str]]) -> float:
    if not preds:
        return 0.0
    return sum(vqa_score(p, g) for p, g in zip(preds, golds)) / len(preds)


# --------------------------------------------------------------------------- #
# dev 子集（固定、落盘）
# --------------------------------------------------------------------------- #


def load_or_create_dev_subset(
    ds,
    bench: str,
    n: int = 100,
    seed: int = 0,
    subsets_dir: Path = Path("data/subsets"),
) -> list[int]:
    """取固定 n 个样本的下标，落盘 data/subsets/<bench>_dev{n}.json。幂等（不重抽）。"""
    subsets_dir.mkdir(parents=True, exist_ok=True)
    out = subsets_dir / f"{bench}_dev{n}.json"
    if out.exists():
        with out.open() as f:
            return json.load(f)
    seed_everything(seed)
    total = len(ds)
    if total <= n:
        idxs = list(range(total))
    else:
        idxs = sorted(range(total), key=lambda _: __import__("random").random())[:n]
    with out.open("w") as f:
        json.dump(idxs, f)
    return idxs


# --------------------------------------------------------------------------- #
# 单样本跑一个方法
# --------------------------------------------------------------------------- #


def build_textvqa_prompt(question: str, ocr_tokens: list[str]) -> str:
    """构造官方 TextVQA prompt 的「问题正文」。

    格式与 LLaVA 官方 `llava_textvqa_val_v051_ocr.jsonl` 的 `text` 字段一致：
        OCR tokens: <tok1> <tok2> ...
        Question: <q>  Short answer:
    （注意 "Short answer:" 前是两个空格，与官方 eval_textvqa.py 的 regex 对应。）
    完整对话由 wrapper 用 LLaVA vicuna 模板包裹。
    """
    ocr = " ".join(str(t) for t in (ocr_tokens or []))
    return f"OCR tokens: {ocr}\nQuestion: {question}  Short answer:"


def run_single_sample(
    wrapper,
    sample,
    bench_suffix: str,
    max_new_tokens: int = 64,
) -> tuple[str, list[int]]:
    """跑单样本，返回 (predicted_str, token_schedule)。

    TextVQA 用官方 OCR prompt 构造问题正文；其余 bench 退化为 `question + bench_suffix`。
    """
    image = sample["image"]
    if "ocr_tokens" in sample:
        q_body = build_textvqa_prompt(str(sample["question"]), sample["ocr_tokens"])
    else:
        q_body = str(sample["question"])
        if bench_suffix:
            q_body = f"{q_body}\n{bench_suffix}"
    pred = wrapper.generate(
        [image], q_body, bench_suffix="", max_new_tokens=max_new_tokens
    )
    return pred, list(wrapper.token_schedule)


def gold_answers(sample) -> list[str]:
    if "answers" in sample:
        return [str(a) for a in sample["answers"]]
    if "answer" in sample:
        return [str(sample["answer"])]
    raise KeyError(f"样本无 answers/answer 字段，字段为 {list(sample.keys())}")