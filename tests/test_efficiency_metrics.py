import pytest

from tcbench.metrics.efficiency import (
    distribution_summary,
    quality_metrics,
    relative_saving,
    speedup,
)


def test_distribution_summary_uses_protocol_quantiles() -> None:
    out = distribution_summary([1, 2, 3, 4, 5])
    assert out == {"median": 3.0, "p25": 2.0, "p75": 4.0, "p95": 4.8}


def test_distribution_summary_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        distribution_summary([])
    with pytest.raises(ValueError):
        distribution_summary([-1])


def test_quality_and_cost_derivations() -> None:
    quality = quality_metrics(score=0.58, noop_score=0.60)
    assert quality["quality_drop_pp"] == pytest.approx(2.0)
    assert quality["quality_retention"] == pytest.approx(0.58 / 0.60)
    assert speedup(100, 50) == 2.0
    assert relative_saving(100, 75) == 0.25
