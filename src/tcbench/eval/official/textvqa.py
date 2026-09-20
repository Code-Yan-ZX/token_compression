"""TextVQA 官方评测指标实现。

来源:LLaVA 官方评测脚本
  - 评分: `third_party/SparseVLMs/llava/eval/m4c_evaluator.py`
           `TextVQAAccuracyEvaluator` + `EvalAIAnswerProcessor`
  - 题干处理: `third_party/SparseVLMs/llava/eval/eval_textvqa.py` `prompt_processor`

PROTOCOL §3：评测指标用官方实现，不自造。这里把官方逻辑独立成干净实现（不 import
第三方模型代码），保证与 LLaVA/SparseVLM repo 报告的 TextVQA 数字同口径。
"""

from __future__ import annotations

import re


class EvalAIAnswerProcessor:
    """VQA/EvalAI 官方答案规范化。抄自 SparseVLMs/llava/eval/m4c_evaluator.py#L7。

    关键点：TextVQA 官方用【软得分】——答案在一个样本 10 个人类答案里被提到 k 次，
    记 min(1, k/3) 分，而非简单的对/错。
    """

    CONTRACTIONS = {
        "aint": "ain't", "arent": "aren't", "cant": "can't",
        "couldve": "could've", "couldnt": "couldn't", "dont": "don't",
        "didnt": "didn't", "doesnt": "doesn't", "isnt": "isn't",
        "itll": "it'll", "thats": "that's", "theres": "there's",
        "theyre": "they're", "theyve": "they've", "youre": "you're",
        "youve": "you've", "wont": "won't", "wouldnt": "wouldn't",
    }
    NUMBER_MAP = {
        "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
        "nine": "9", "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_",
        "-", ">", "<", "@", "`", ",", "?", "!",
    ]

    def word_tokenize(self, word: str) -> str:
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text: str) -> str:
        out = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text) or (" " + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out = out.replace(p, "")
            else:
                out = out.replace(p, " ")
        out = self.PERIOD_STRIP.sub("", out, re.UNICODE)
        return out

    def process_digit_article(self, in_text: str) -> str:
        temp = in_text.lower().split()
        out = [self.NUMBER_MAP.get(w, w) for w in temp]
        out = [w for w in out if w not in self.ARTICLES]
        out = [self.CONTRACTIONS.get(w, w) for w in out]
        return " ".join(out)

    def __call__(self, item: str) -> str:
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


class TextVQAAccuracyEvaluator:
    """官方软得分 VQA accuracy。抄自 SparseVLMs L221。"""

    def __init__(self) -> None:
        self.answer_processor = EvalAIAnswerProcessor()

    def _compute_answer_scores(self, raw_answers: list[str]) -> dict[str, float]:
        answers = [self.answer_processor(a) for a in raw_answers]
        assert len(answers) == 10, f"TextVQA 每个样本应有 10 个答案，得到 {len(answers)}"
        gt_answers = list(enumerate(answers))
        unique_answers = set(answers)
        scores: dict[str, float] = {}
        for unique in unique_answers:
            accs = []
            for gt in gt_answers:
                others = [item for item in gt_answers if item != gt]
                matching = [item for item in others if item[1] == unique]
                accs.append(min(1, float(len(matching)) / 3))
            scores[unique] = sum(accs) / len(accs)
        return scores

    def eval_sample(self, pred_answer: str, gt_answers: list[str]) -> float:
        """单个样本的官方分数（0~1，软得分）。"""
        pred = self.answer_processor(pred_answer)
        scores = self._compute_answer_scores(gt_answers)
        return scores.get(pred, 0.0)

    def eval_pred_list(self, preds: list[str], golds: list[list[str]]) -> float:
        accs = [self.eval_sample(p, g) for p, g in zip(preds, golds)]
        return sum(accs) / len(accs)


def prompt_processor(prompt: str) -> str:
    """从官方 prompt 字符串里提取 question 小写形式（SparseVLMs eval_textvqa.py）。"""
    if prompt.startswith("OCR tokens: "):
        m = re.search(r"Question: (.*?) Short answer:", prompt, re.DOTALL)
        if m:
            return m.group(1).strip().lower()
    elif "Reference OCR token: " in prompt and len(prompt.split("\n")) == 3:
        question = prompt.split("\n")[1] if prompt.startswith("Reference OCR token:") else prompt.split("\n")[0]
        return question.strip().lower()
    elif len(prompt.split("\n")) == 2:
        return prompt.split("\n")[0].strip().lower()
    raise ValueError(f"无法解析官方 prompt 格式: {prompt[:80]!r}")