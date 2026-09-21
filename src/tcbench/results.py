"""结果落盘:run JSON 的 schema、写入工具与校验。

CLAUDE.md §2.3:`results/runs/*.json` append-only,每个数字必须可追溯到 run_id。
T1.1 起每个 run 要把 keep_index / merge_map 落盘到 artifacts_path(供 T1.5 补算 ER)。
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results")
RUNS_DIR = RESULTS_DIR / "runs"

RUN_ID_RE = re.compile(r"^[0-9a-z_-]{1,64}$")


def gen_run_id() -> str:
    """短可读 run id，避免撞到。用时间戳 + 4 hex。全小写无大写字母。"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


def git_commit() -> str | None:
    try:
        r = __import__("subprocess").run(
            ["git", "-C", str(Path.cwd()), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


_REQUIRED_TOP_LEVEL = (
    "run_id", "timestamp", "git_commit", "protocol_version",
    "config_hash", "full_config_dump", "base_model_id", "base_model_revision",
    "method", "budget_spec", "seed", "hardware", "metrics", "artifacts_path",
)

# run_kind 三字段对 2026-09-21 之后的新 run 必填。校验器仍把缺失字段的历史 JSON
# 视为 legacy quality run，避免为了 schema 升级改写 append-only 结果。

_EFFICIENCY_REQUIRED = (
    "attention_track",
    "attn_impl",
    "batch_size",
    "warmup_iterations",
    "profile_samples",
    "passes",
    "raw_timings_path",
    "summary",
)


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    git_commit: str | None
    protocol_version: str
    config_hash: str
    full_config_dump: dict[str, Any]
    base_model_id: str
    base_model_revision: str
    method: str
    budget_spec: dict[str, Any]
    seed: int
    hardware: dict[str, Any]
    metrics: dict[str, Any]
    artifacts_path: str
    bench: str = ""
    split: str = ""
    token_schedule: list[int] = field(default_factory=list)  # 长度 L,LLM 各层输入端 vtok 数
    tlb: float | None = None  # 由 token_schedule 实测算出,不允许从配置反推
    run_kind: str = "quality"  # quality | efficiency
    measurement_protocol_version: str = ""
    linked_run_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        # 保留全部字段(含 None),确保 schema 必填项不因 None 而消失
        return dict(self.__dict__)


def dump_run(rec: RunRecord, runs_dir: Path = RUNS_DIR) -> Path:
    """写入 run JSON。append-only:文件名即 run_id,绝不覆盖。"""
    if not RUN_ID_RE.match(rec.run_id):
        raise ValueError(f"非法 run_id: {rec.run_id!r}")
    runs_dir.mkdir(parents=True, exist_ok=True)
    out = runs_dir / f"{rec.run_id}.json"
    if out.exists():
        raise FileExistsError(f"拒绝覆盖已有 run 文件: {out} (append-only)")
    with out.open("w", encoding="utf-8") as f:
        json.dump(rec.to_dict(), f, indent=2, ensure_ascii=False)
    return out


def validate_run(rec: RunRecord | dict[str, Any]) -> list[str]:
    """schema 校验,返回缺失/非法项清单(空表=通过)。"""
    d = rec.to_dict() if isinstance(rec, RunRecord) else rec
    errs: list[str] = []
    for key in _REQUIRED_TOP_LEVEL:
        if key not in d:
            errs.append(f"缺少必填字段: {key}")
    if "metrics" in d and not isinstance(d["metrics"], dict):
        errs.append("metrics 必须是 dict")
    if "token_schedule" in d and not isinstance(d["token_schedule"], list):
        errs.append("token_schedule 必须是 list")
    if "tlb" in d and d["tlb"] is not None and not (0 <= d["tlb"] <= 1.0001):
        errs.append(f"tlb 越界: {d.get('tlb')}")
    run_kind = d.get("run_kind", "quality")
    if run_kind not in {"quality", "efficiency"}:
        errs.append(f"run_kind 非法: {run_kind!r}")
    if run_kind == "efficiency":
        if not d.get("measurement_protocol_version"):
            errs.append("efficiency run 缺少 measurement_protocol_version")
        if not d.get("linked_run_ids"):
            errs.append("efficiency run 缺少 linked_run_ids")
        metrics = d.get("metrics", {})
        efficiency = metrics.get("efficiency") if isinstance(metrics, dict) else None
        if not isinstance(efficiency, dict):
            errs.append("efficiency run 的 metrics.efficiency 必须是 dict")
        else:
            for key in _EFFICIENCY_REQUIRED:
                if key not in efficiency:
                    errs.append(f"efficiency run 缺少 metrics.efficiency.{key}")
    return errs


def load_run(run_id: str, runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    p = runs_dir / f"{run_id}.json"
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
