"""T1.0 基建测试:config_hash 稳定性、run schema、seed、环境指纹。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcbench.config import (
    BudgetConfig,
    MethodConfig,
    BaseConfig,
    RunConfig,
    _canonical,
    config_hash,
)
from tcbench.results import RunRecord, dump_run, gen_run_id, validate_run
from tcbench.seed import EnvFingerprint, seed_everything


# --------------------------------------------------------------------------- #
# config_hash
# --------------------------------------------------------------------------- #


def _mk_cfg(prompt: str = "Answer.") -> RunConfig:
    return RunConfig(
        base=BaseConfig(
            model_id="llava-hf/llava-1.5-7b-hf",
            revision="a" * 40,
            bench_suffix="Answer the question using a single word or phrase.",
        ),
        method=MethodConfig(name="NoOp", stage="pre_llm"),
        budget=BudgetConfig(kind="tlb", value=0.111),
        bench="textvqa",
        split="val",
        seed=0,
        prompt=prompt,
    )


def test_config_hash_stable_across_calls():
    a = config_hash(_mk_cfg())
    b = config_hash(_mk_cfg())
    assert a == b and len(a) == 64


def test_config_hash_sensitive_to_semantics():
    # method 变了,hash 必须变
    c1 = _mk_cfg()
    c2 = _mk_cfg()
    c2.method.name = "VisPruner"
    assert config_hash(c1) != config_hash(c2)
    # budget 变了,hash 必须变
    c3 = _mk_cfg()
    c3.budget.value = 0.333
    assert config_hash(c1) != config_hash(c3)


def test_canonical_ignores_key_order():
    d1 = {"a": 1, "b": {"y": 2, "x": 3}}
    d2 = {"b": {"x": 3, "y": 2}, "a": 1}
    assert _canonical(d1) == _canonical(d2)


# --------------------------------------------------------------------------- #
# run schema
# --------------------------------------------------------------------------- #


def _mk_run() -> RunRecord:
    return RunRecord(
        run_id=gen_run_id(),
        timestamp="2026-09-20T00:00:00+00:00",
        git_commit=None,
        protocol_version="v0.1",
        config_hash="0" * 64,
        full_config_dump={},
        base_model_id="llava-hf/llava-1.5-7b-hf",
        base_model_revision="a" * 40,
        method="NoOp",
        budget_spec={"kind": "tlb", "value": 1.0},
        seed=0,
        hardware={},
        metrics={"accuracy": "NOT_RUN"},
        artifacts_path="",
        bench="textvqa",
        split="val",
        token_schedule=[576] * 32,
        tlb=1.0,
    )


def test_validate_run_ok():
    rec = _mk_run()
    assert validate_run(rec) == []


def test_validate_run_missing_field():
    d = _mk_run().to_dict()
    del d["seed"]
    errs = validate_run(d)
    assert any("seed" in e for e in errs)


def test_tlb_out_of_range():
    rec = _mk_run()
    rec.tlb = 1.5
    errs = validate_run(rec)
    assert any("tlb" in e for e in errs)


def test_dump_run_append_only(tmp_path: Path):
    r1 = _mk_run()
    r1.run_id = "testrun000001"
    p1 = dump_run(r1, runs_dir=tmp_path)
    assert p1.name == "testrun000001.json"
    # 同 id 第二次必须拒绝(append-only)
    with pytest.raises(FileExistsError):
        dump_run(r1, runs_dir=tmp_path)
    # 内容可回读
    with p1.open() as f:
        d = json.load(f)
    assert d["run_id"] == "testrun000001"
    assert d["metrics"]["accuracy"] == "NOT_RUN"


def test_dump_run_illegal_id(tmp_path: Path):
    r = _mk_run()
    r.run_id = "../escape"
    with pytest.raises(ValueError):
        dump_run(r, runs_dir=tmp_path)


# --------------------------------------------------------------------------- #
# seed / env
# --------------------------------------------------------------------------- #


def test_seed_everything_runs():
    # 不抛异常即可
    seed_everything(0)


def test_env_fingerprint_has_required_keys():
    env = EnvFingerprint.capture()
    d = env.to_dict()
    for key in ("python_version", "torch_version", "gpu"):
        assert key in d


# --------------------------------------------------------------------------- #
# load_config 从样例 YAML
# --------------------------------------------------------------------------- #


def test_load_config_from_yaml(tmp_path: Path):
    (tmp_path / "base.yaml").write_text(
        "model_id: llava-hf/llava-1.5-7b-hf\nrevision: 'aaaa'\nbench_suffix: 'Answer.'\n"
    )
    (tmp_path / "method.yaml").write_text("name: NoOp\nstage: pre_llm\n")
    (tmp_path / "budget.yaml").write_text("kind: tlb\nvalue: 0.111\n")
    # load_config 相对 configs/ 找 ref;这里直接传路径
    m = pytest.importorskip("tcbench.config")
    cfg = m.load_config(tmp_path / "base.yaml", tmp_path / "method.yaml", tmp_path / "budget.yaml", "textvqa", 0)
    assert cfg.bench == "textvqa"
    assert cfg.seed == 0
    assert cfg.budget.value == 0.111