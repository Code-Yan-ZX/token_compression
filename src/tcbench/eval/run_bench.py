"""对给定数组样本跑无压缩 (NoOp) 评测，生成可追溯的 run JSON。

用法:
    PYTHONPATH=src python -m tcbench.eval.run_bench \
        --bench textvqa --split dev100 --seed 0

产出:
    - results/runs/<run_id>.json   (含 token_schedule、TLB、metrics)
    - 控制台打印 accuracy 与关键字段
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from tcbench import env
from tcbench.config import config_hash, load_config
from tcbench.eval.runner import compute_accuracy, gold_answers, load_or_create_dev_subset
from tcbench.models.llava_wrapper import LLaVAWrapper
from tcbench.results import RunRecord, dump_run, gen_run_id, git_commit, validate_run
from tcbench.seed import EnvFingerprint, seed_everything


PROTOCOL_VERSION = "v0.2"


def tlb_from_schedule(n0: int, schedule: list[int]) -> float:
    """由逐层 n_ℓ 实测计算 TLB（PROTOCOL §2.1）。不允许从配置反推。"""
    if not schedule:
        return None
    L = len(schedule)
    return sum(schedule) / (L * n0)


def main() -> int:
    env.configure()  # 缓存重定向（先于任何 HF/transformers import）
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--budget", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-n", type=int, default=100, help="dev 子集大小")
    ap.add_argument("--full-val", action="store_true",
                    help="跑整个验证集（split=val，T1.1 定稿用）。与 --dev-n 二选一，优先。")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 个（调试用）")
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = load_config(args.base, args.method, args.budget, args.bench, args.seed)
    n0 = cfg.base.patch_grid[0] * cfg.base.patch_grid[1]  # 576

    import torch
    from datasets import load_dataset

    try:
        ds = load_dataset("lmms-lab/TextVQA", split="validation")
    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] 数据集加载失败，先运行数据准备: {e}", file=sys.stderr)
        return 2

    # split 选择：--full-val 跑整个验证集（T1.1 定稿）；否则用固定 dev 子集。
    is_full_val = args.full_val
    if is_full_val:
        idxs = list(range(len(ds)))
    else:
        idxs = load_or_create_dev_subset(ds, args.bench, n=args.dev_n, seed=args.seed)
    if args.limit:
        idxs = idxs[: args.limit]
    samples_idx = idxs  # 每个样本在原始数据集里的下标（供预测落盘回溯）
    samples = [ds[i] for i in idxs]

    # wrapper
    wrap = LLaVAWrapper(model_id="llava-hf/llava-1.5-7b-hf", device="cuda")
    wrap.model_id = env.model_dir("llava-hf/llava-1.5-7b-hf") or "llava-hf/llava-1.5-7b-hf"
    wrap.dtype = torch.float16
    wrap.load()

    bench_suffix = cfg.base.bench_suffix
    preds = []
    golds = []
    all_schedules = []
    for row in samples:
        pred, sched = run_one(wrap, row, bench_suffix, cfg.base.max_new_tokens)
        preds.append(pred)
        golds.append(gold_answers(row))
        all_schedules.append(sched)
    acc = compute_accuracy(preds, golds)

    # 首个样本的 schedule 作为该 run 的 token_schedule（TLB 预算在 prefill 恒定）
    token_schedule = all_schedules[0] if all_schedules else []
    tlb = tlb_from_schedule(n0, token_schedule)

    run_id = gen_run_id()
    # 逐样本预测落盘（artifacts），供离线重算指标（如换官方 metric），不重跑推理。
    run_dir = Path("results/artifacts") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    preds_path = run_dir / "predictions.jsonl"
    with preds_path.open("w", encoding="utf-8") as f:
        for i, (orig_idx, sched) in enumerate(zip(samples_idx, all_schedules, strict=True)):
            f.write(json.dumps({"idx": orig_idx, "sample_no": i, "pred": preds[i], "token_schedule": sched}) + "\n")

    rec = RunRecord(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit(),
        protocol_version=PROTOCOL_VERSION,
        config_hash=config_hash(cfg),
        full_config_dump=asdict(cfg),
        base_model_id=cfg.base.model_id,
        base_model_revision=cfg.base.revision,
        method=cfg.method.name,
        budget_spec={"kind": cfg.budget.kind, "value": cfg.budget.value},
        seed=cfg.seed,
        hardware=EnvFingerprint.capture().to_dict(),
        metrics={"accuracy": acc, "n_samples": len(samples)},
        artifacts_path=str(preds_path),
        bench=cfg.bench,
        split="val" if is_full_val else f"dev{args.dev_n}",
        token_schedule=token_schedule,
        tlb=tlb,
    )
    errs = validate_run(rec)
    if errs:
        print("schema 校验失败:", *errs, sep="\n  ")
        return 1
    path = dump_run(rec)
    print(f"已写入 run: {path}")
    print(f"  bench={rec.bench}/{rec.split}  method={rec.method}  seed={rec.seed}")
    print(f"  samples={len(samples)}  accuracy={acc:.4f}")
    print(f"  token_schedule len={len(token_schedule)}  first3={token_schedule[:3]}  tlb={tlb:.4f}")
    return 0


def run_one(wrap, row, bench_suffix: str, max_new_tokens: int) -> tuple[str, list[int]]:
    from tcbench.eval.runner import run_single_sample
    return run_single_sample(wrap, row, bench_suffix, max_new_tokens)


if __name__ == "__main__":
    raise SystemExit(main())