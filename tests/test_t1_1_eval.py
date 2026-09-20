"""T1.1 eval 组件的离线测试（不加载 GPU/模型）。

度量测试用官方 TextVQA 软得分（TextVQAAccuracyEvaluator），验证与官方口径一致。
"""

from __future__ import annotations

from pathlib import Path

from tcbench.eval.official.textvqa import (
    EvalAIAnswerProcessor,
    TextVQAAccuracyEvaluator,
    prompt_processor,
)
from tcbench.eval.runner import compute_accuracy, load_or_create_dev_subset


def test_answer_processor_basic():
    p = EvalAIAnswerProcessor()
    # 小写 + 去冠词
    assert p("  The Hello ") == "hello"
    # 数字映射
    assert p("two") == "2"


def test_answer_processor_number_and_article():
    p = EvalAIAnswerProcessor()
    assert p("One apple") == "1 apple"
    # 去标点
    assert p("blue.") == "blue"


def test_soft_score_partial_credit():
    ev = TextVQAAccuracyEvaluator()
    # 10 个答案里 "blue" 出现 8 次 → 记 1.0
    gt = ["blue"] * 6 + ["red"] * 4
    assert ev.eval_sample("blue", gt) == 1.0
    # 未出现 → 0
    assert ev.eval_sample("green", gt) == 0.0
    # 部分：red 出现 2 次 → 软得分 0.6（官方逻辑）
    assert ev.eval_sample("red", ["red"] * 2 + ["blue"] * 8) == 0.6


def test_compute_accuracy_soft():
    # 全对 → 1.0
    preds = ["a", "b"]
    golds = [["a"] * 10, ["b"] * 10]
    assert compute_accuracy(preds, golds) == 1.0


def test_prompt_processor():
    # 官方 prompt 格式（eval_textvqa.py）：OCR 行 + Question: ...  Short answer:
    p = "OCR tokens: DAKOTA DIGITAL\nQuestion: What brand is the camera?  Short answer:"
    assert prompt_processor(p) == "what brand is the camera?"


def test_dev_subset_is_fixed_and_idempotent(tmp_path: Path):
    class FakeDS:
        def __init__(self):
            self.n = 500

        def __len__(self):
            return self.n

    ds = FakeDS()
    a = load_or_create_dev_subset(ds, "textvqa", n=100, seed=0, subsets_dir=tmp_path)
    b = load_or_create_dev_subset(ds, "textvqa", n=100, seed=0, subsets_dir=tmp_path)
    # 幂等：同 seed 同文件，第二次直接读
    assert a == b and len(a) == 100
    assert (tmp_path / "textvqa_dev100.json").exists()