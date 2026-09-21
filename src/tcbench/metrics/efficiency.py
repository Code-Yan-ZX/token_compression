"""EFFICIENCY_PROTOCOL e1.0 的纯计算辅助函数。

GPU 计时边界由各方法 runner 负责；本模块只做无副作用的汇总和派生指标，
确保所有 runner 使用相同公式。
"""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite
from statistics import median


def distribution_summary(values: Iterable[float]) -> dict[str, float]:
    """返回 median/p25/p75/p95；使用线性插值分位数。"""
    xs = sorted(float(v) for v in values)
    if not xs:
        raise ValueError("不能汇总空序列")
    if not all(isfinite(v) and v >= 0 for v in xs):
        raise ValueError("效率指标必须是有限非负数")

    def quantile(q: float) -> float:
        pos = (len(xs) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        frac = pos - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac

    return {
        "median": median(xs),
        "p25": quantile(0.25),
        "p75": quantile(0.75),
        "p95": quantile(0.95),
    }


def quality_metrics(score: float, noop_score: float) -> dict[str, float]:
    """计算绝对掉点（百分点）与质量保持率。输入分数范围为 [0, 1]。"""
    if not (0 <= score <= 1 and 0 < noop_score <= 1):
        raise ValueError("score 必须在 [0,1]，noop_score 必须在 (0,1]")
    return {
        "quality_drop_pp": 100.0 * (noop_score - score),
        "quality_retention": score / noop_score,
    }


def speedup(noop_latency_ms: float, method_latency_ms: float) -> float:
    """相对 NoOp 的时延加速比。"""
    if noop_latency_ms <= 0 or method_latency_ms <= 0:
        raise ValueError("latency 必须为正数")
    return noop_latency_ms / method_latency_ms


def relative_saving(noop_cost: float, method_cost: float) -> float:
    """相对 NoOp 的成本节省比例；负值表示方法更贵。"""
    if noop_cost <= 0 or method_cost < 0:
        raise ValueError("noop_cost 必须为正，method_cost 必须非负")
    return 1.0 - method_cost / noop_cost
