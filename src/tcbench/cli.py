"""tcbench 命令行入口。

主要子命令:
- `run`        执行一次实验(Phase 1 阶段先支持 dummy run 验证 pipeline)
- `table`      从 results/runs/*.json 生成汇总表(脚本生成,不手写)
- `verify-protocol`  协议自检:检查配置 schema、run schema、铁律约束
"""

from __future__ import annotations

import argparse
import json
import sys

from tcbench.config import RunConfig, config_hash, load_config
from tcbench.results import RunRecord, dump_run, gen_run_id, validate_run
from tcbench.seed import EnvFingerprint, seed_everything


PROTOCOL_VERSION = "v0.3"


def cmd_verify_protocol(args: argparse.Namespace) -> int:
    """协议自检:加载默认配置、生成 dummy run 并校验 schema。"""

    # 构造一份最小的合法配置并跑通 schema,确认协议定义可落地
    from tcbench.config import BaseConfig, BudgetConfig, MethodConfig
    from tcbench.results import gen_run_id

    try:
        _base = BaseConfig(
            model_id="dummy/test",
            revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        _method = MethodConfig(name="NoOp", stage="pre_llm")
        _budget = BudgetConfig(kind="tlb", value=1.0)
        _cfg = RunConfig(
            base=_base, method=_method, budget=_budget,
            bench="textvqa", split="val", seed=0, prompt="",
        )
        _h = config_hash(_cfg)
        if len(_h) != 64:
            raise RuntimeError(f"config_hash 长度异常: {len(_h)}")
        rec = RunRecord(
            run_id=gen_run_id(), timestamp="", git_commit=None,
            protocol_version=PROTOCOL_VERSION, config_hash=_h,
            full_config_dump={}, base_model_id=_base.model_id,
            base_model_revision=_base.revision, method=_method.name,
            budget_spec={"kind": _budget.kind, "value": _budget.value},
            seed=0, hardware={}, metrics={}, artifacts_path="",
            bench="textvqa", split="val",
        )
        errs = validate_run(rec)
        if errs:
            for e in errs:
                print(f"[FAIL] {e}")
            return 1
        print("[OK] config_hash / run schema 校验通过")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {type(e).__name__}: {e}")
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """执行一次实验。当前为 dummy 版:验证 pipeline 能生成合法 run JSON。

    Phase 1 T1.1 之后接入真实 model wrapper / pruner 推理。
    """
    seed_everything(args.seed)
    env = EnvFingerprint.capture()

    cfg = load_config(
        args.base, args.method, args.budget,
        args.bench, args.seed,
    )

    # --- 模拟指标:直到接上真实推理前,用占位 NOT_RUN,绝不写编造数字 ---
    metrics: dict = {"accuracy": "NOT_RUN"}
    token_schedule: list[int] = []

    from dataclasses import asdict
    rec = RunRecord(
        run_id=gen_run_id(),
        timestamp="",
        git_commit=None,
        protocol_version=PROTOCOL_VERSION,
        config_hash=config_hash(cfg),
        full_config_dump=asdict(cfg),
        base_model_id=cfg.base.model_id,
        base_model_revision=cfg.base.revision,
        method=cfg.method.name,
        budget_spec={"kind": cfg.budget.kind, "value": cfg.budget.value},
        seed=cfg.seed,
        hardware=env.to_dict(),
        metrics=metrics,
        artifacts_path="",
        bench=cfg.bench,
        split=cfg.split,
        token_schedule=token_schedule,
    )

    # 自动填时间戳与 git commit
    from datetime import datetime, timezone
    rec.timestamp = datetime.now(timezone.utc).isoformat()
    rec.git_commit = __import__("tcbench.results", fromlist=["git_commit"]).git_commit()

    errs = validate_run(rec)
    if errs:
        print("schema 校验失败:", *errs, sep="\n  ")
        return 1

    path = dump_run(rec)
    print(f"已写入 run: {path}")
    print(f"  run_id       = {rec.run_id}")
    print(f"  config_hash  = {rec.config_hash}")
    print(f"  base         = {rec.base_model_id}")
    print(f"  method       = {rec.method}  budget={cfg.budget.kind}={cfg.budget.value}")
    print(f"  bench        = {rec.bench} / {rec.split}  seed={rec.seed}")
    print(f"  metrics      = {metrics}")
    return 0


def cmd_table(args: argparse.Namespace) -> int:
    """从 results/runs/*.json 生成 markdown 汇总表。"""
    from pathlib import Path
    from tcbench.results import RUNS_DIR

    bench = args.bench
    rows = []
    for f in sorted((RUNS_DIR or Path("results/runs")).glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("stale"):  # v0.1 旧口径，排除（协议已升 v0.2）
            continue
        if bench and d.get("bench") != bench:
            continue
        rows.append(d)
    if not rows:
        print(f"没有 [{bench or 'all'}] 下的 run", file=sys.stderr)
        return 1
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {bench} 汇总表", "", "| run_id | method | budget | seed | metric |", "|---|---|---|---|---|"]
    for d in rows:
        m = d.get("metrics", {})
        lines.append(
            f"| {d['run_id']} | {d['method']} | {d['budget_spec']} | {d['seed']} | {m} |"
        )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"汇总表已生成: {out_path} ({len(rows)} runs)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tcbench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="执行一次实验")
    r.add_argument("--base", required=True, help="base 配置文件路径")
    r.add_argument("--method", required=True, help="method 配置文件路径")
    r.add_argument("--budget", required=True, help="budget 配置文件路径")
    r.add_argument("--bench", required=True, help="benchmark: textvqa/gqa/docvqa/pope/mme")
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("table", help="生成汇总表")
    t.add_argument("--bench", default="")
    t.add_argument("--out", default="results/tables/table.md")
    t.set_defaults(func=cmd_table)

    v = sub.add_parser("verify-protocol", help="协议自检")
    v.set_defaults(func=cmd_verify_protocol)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
