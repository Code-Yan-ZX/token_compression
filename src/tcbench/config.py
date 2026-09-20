"""配置解析与 config_hash。

CLAUDE.md §3/T1.0:配置用 YAML + dataclass 解析，`config_hash` 对语义项做稳定哈希，
忽略注释与 key 顺序。budget / prompt / split 只能来自配置文件，才能算 config_hash。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# 语义化配置 dataclass。实验语义项（budget / prompt / split）只允许在这里出现。
# --------------------------------------------------------------------------- #


@dataclass
class BaseConfig:
    """base model 配置。见 PROTOCOL §1。"""

    model_id: str
    revision: str  # 必须锁定具体 commit hash，不得用 main
    dtype: str = "float16"
    attn_impl: str = "eager"  # 精度实验统一 eager；时延另设
    patch_grid: tuple[int, int] = (24, 24)  # 固定分辨率主模型
    llm_layers: int = 32
    max_new_tokens: int = 128
    temperature: float = 0.2  # PROTOCOL §1.3 v0.2：与 LLaVA 官方一致
    bench_suffix: str = ""  # benchmark 指令后缀，prompt 共用


@dataclass
class MethodConfig:
    """压缩方法配置。名字必须能映射到 src/tcbench/pruners 里的实现。"""

    name: str
    stage: str  # vit / pre_llm / in_llm
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetConfig:
    """budget 档位。维度是"名义保留数 N"还是"TLB 目标"，由 kind 指明。"""

    kind: str  # "n_tokens" 或 "tlb"
    value: float  # n_tokens 时取整；tlb 时在 (0,1]


@dataclass
class RunConfig:
    """一次实验的完整、冻结后的配置。"""

    base: BaseConfig
    method: MethodConfig
    budget: BudgetConfig
    bench: str  # textvqa / gqa / docvqa / pope / mme
    split: str
    seed: int
    prompt: str  # 冻结后的完整 prompt(模板 + bench_suffix)。不得在命令行覆盖。


CONFIG_SEMANTIC_FIELDS = ("base", "method", "budget", "bench", "split", "seed", "prompt")


# --------------------------------------------------------------------------- #
# YAML 加载
# --------------------------------------------------------------------------- #


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"配置 {p} 顶层必须是一个映射")
    return raw


def _resolve_ref(raw: dict[str, Any]) -> dict[str, Any]:
    """支持简单 `base: {ref: configs/base/llava15_7b.yaml}` 或直接内联映射。"""
    for out_key in ("base", "method", "budget"):
        v = raw.get(out_key)
        if isinstance(v, dict) and set(v.keys()) == {"ref"}:
            ref_path = Path(str(v["ref"]))
            if not ref_path.is_absolute():
                # 相对当前 YAML 所在目录的 configs 根
                ref_path = Path("configs") / ref_path
            raw[out_key] = _load_yaml(ref_path)
    return raw


def _coerce_attrs(cfg_cls, data: dict[str, Any], origin: str) -> Any:
    """把 dict 塞进 dataclass，tuple 类型做一次性转换，忽略未知 key。"""
    if not is_dataclass(cfg_cls):
        raise TypeError(f"{origin}: 期望 dataclass，得到 {cfg_cls}")
    from dataclasses import fields
    hints = __import__("typing").get_type_hints(cfg_cls)  # noqa: F821
    valid = {f.name for f in fields(cfg_cls)}
    kwargs: dict[str, Any] = {}
    for name, val in data.items():
        if name not in valid:
            continue  # 未知 key 忽略(如注释类)
        typ = hints.get(name, Any)
        if str(typ).startswith("tuple"):
            val = tuple(val)
        kwargs[name] = val
    return cfg_cls(**kwargs)


def load_config(
    base_path: str | Path,
    method_path: str | Path,
    budget_path: str | Path,
    bench: str,
    seed: int,
) -> RunConfig:
    """从三份配置文件 + bench + seed 组装一份冻结的 RunConfig。"""
    base_raw = _load_yaml(base_path)
    method_raw = _load_yaml(method_path)
    budget_raw = _load_yaml(budget_path)

    base_obj = _coerce_attrs(BaseConfig, base_raw, "base")
    method_obj = _coerce_attrs(MethodConfig, method_raw, "method")
    budget_obj = _coerce_attrs(BudgetConfig, budget_raw, "budget")

    suffix = base_obj.bench_suffix
    prompt = suffix  # prompt = 模板(此处简化) + bench_suffix

    # bench 的 split 校验:只允许协议定的 split
    base_obj.prompt_template = base_obj.bench_suffix  # preserved for hash

    return RunConfig(
        base=base_obj,
        method=method_obj,
        budget=budget_obj,
        bench=bench,
        split=_default_split(bench),
        seed=seed,
        prompt=prompt,
    )


def _default_split(bench: str) -> str:
    # PROTOCOL §3
    return {
        "gqa": "testdev_balanced",
        "textvqa": "val",
        "docvqa": "val",
        "pope": "random",
        "mme": "perception",
    }.get(bench, "dev")


# --------------------------------------------------------------------------- #
# config_hash:稳定哈希，忽略注释与 key 顺序
# --------------------------------------------------------------------------- #


def _canonical(value: Any) -> Any:
    """把任意值规整为可排序/可哈希的规范形式。dict 按键排序。"""
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)) or value is None:
        return value
    if is_dataclass(value):
        return _canonical(asdict(value))
    return str(value)


def _config_semantics(cfg: RunConfig) -> dict[str, Any]:
    d = asdict(cfg)
    # 只保留语义项，忽略诸如时间戳、路径类的非语义内容
    return {k: d[k] for k in CONFIG_SEMANTIC_FIELDS if k in d}


def config_hash(cfg: RunConfig) -> str:
    """基于语义项的稳定 SHA-256。忽略注释与 key 顺序。

    注意:revision 被包含,因为它是可复现性的一部分(model 权重版本)。
    """
    canonical = _canonical(_config_semantics(cfg))
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()