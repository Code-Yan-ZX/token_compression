"""官方 TextVQA 评测（prompt 100% 官方；图像与 gold 来自 TextVQA 数据集本身）。

对齐全链 LLaVA-1.5 官方 textvqa：
  - 问题正文 prompt : {ocr_jsonl}  官方 llava_textvqa_val_v051_ocr.jsonl 的 text 字段（逐字）
  - 标准答案 gold   : TextVQA 数据集 answers（== TextVQA_0.5.1 官方答案，按 image_id+question 匹配）
  - 图像            : TextVQA 数据集 image 字段（同一批标准图）
不再使用任何 lmms 的 ocr_tokens / prompt 逻辑。

用法:
  PYTHONPATH=src python -m tcbench.eval.run_official_textvqa \
      --ocr-jsonl data/ocr/llava_textvqa_val_v051_ocr.jsonl \
      --dev-n 1000     # 固定 dev 子集；或 --full-val 全量 5000
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone

import torch

from tcbench import env
from tcbench.config import config_hash, load_config
from tcbench.eval.official.textvqa import TextVQAAccuracyEvaluator
from tcbench.models.llava_wrapper import LLaVAWrapper
from tcbench.results import RunRecord, dump_run, gen_run_id, git_commit, validate_run
from tcbench.seed import EnvFingerprint, seed_everything

PROTOCOL_VERSION = "v0.3"
_eval = TextVQAAccuracyEvaluator()


def main() -> int:
    env.configure()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-jsonl", required=True)
    ap.add_argument("--base", default="configs/base/llava15_7b.yaml")
    ap.add_argument("--method", default="configs/method/noop.yaml")
    ap.add_argument("--budget", default="configs/budget/tlb100.yaml")
    ap.add_argument("--bench", default="textvqa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-n", type=int, default=1000)
    ap.add_argument("--full-val", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = load_config(args.base, args.method, args.budget, args.bench, args.seed)
    n0 = cfg.base.patch_grid[0] * cfg.base.patch_grid[1]

    # 官方 jsonl 顺序驱动
    official = []
    for line in open(args.ocr_jsonl):
        o = json.loads(line)
        official.append({"image_id": o["question_id"], "question": o["text"].split("\n")[0].strip(),
                         "text": o["text"]})

    # 从本地 TextVQA 数据集建立 (image_id -> question -> (answers, image))
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/TextVQA", split="validation")
    idx = {}
    for s in ds:
        idx.setdefault(s["image_id"], {})[str(s["question"]).strip().lower()] = \
            {"answers": list(s["answers"]), "image": s["image"]}

    rows = []
    for o in official:
        q = o["question"].lower()
        hit = idx.get(o["image_id"], {}).get(q)
        assert hit is not None, f"official ({o['image_id']},{q}) 在 TextVQA 数据集缺"
        rows.append({**o, "gold": hit["answers"], "image": hit["image"]})
    print(f"官方条目 {len(official)} / 全部配上数据集 gold+image {len(rows)}")

    if args.full_val:
        idxs = list(range(len(rows))); split = "val"
    else:
        sub = Path("data/subsets") / f"textvqa_off_dev{args.dev_n}.json"
        if sub.exists():
            idxs = json.loads(sub.read_text())
        else:
            seed_everything(args.seed)
            idxs = sorted(random.sample(range(len(rows)), min(args.dev_n, len(rows))))
            sub.write_text(json.dumps(idxs))
        split = f"dev{args.dev_n}"
    selected = [rows[i] for i in idxs]

    wrap = LLaVAWrapper(model_id="llava-hf/llava-1.5-7b-hf", device="cuda")
    wrap.model_id = env.model_dir("llava-hf/llava-1.5-7b-hf") or "llava-hf/llava-1.5-7b-hf"
    wrap.dtype = torch.float16
    wrap.load()

    preds, golds, scheds = [], [], []
    for row in selected:
        img = row["image"].convert("RGB")
        pred, sched = run_one(wrap, img, row["text"])
        preds.append(pred); golds.append(row["gold"]); scheds.append(sched)

    acc = sum(_eval.eval_sample(p, g) for p, g in zip(preds, golds)) / len(preds)
    schedule = scheds[0] if scheds else []
    tlb = sum(schedule) / (len(schedule) * n0) if schedule else None

    rid = gen_run_id()
    run_dir = Path("results/artifacts") / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for i in range(len(selected)):
            f.write(json.dumps({"idx": idxs[i], "image_id": selected[i]["image_id"],
                                "pred": preds[i], "token_schedule": scheds[i]}) + "\n")

    rec = RunRecord(
        run_id=rid, timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit(), protocol_version=PROTOCOL_VERSION,
        config_hash=config_hash(cfg), full_config_dump=asdict(cfg),
        base_model_id=cfg.base.model_id, base_model_revision=cfg.base.revision,
        method=cfg.method.name, budget_spec={"kind": cfg.budget.kind, "value": cfg.budget.value},
        seed=cfg.seed, hardware=EnvFingerprint.capture().to_dict(),
        metrics={"accuracy": acc, "n_samples": len(preds)},
        artifacts_path=str(run_dir / "predictions.jsonl"),
        bench=cfg.bench, split=split, token_schedule=schedule, tlb=tlb,
        run_kind="quality",
    )
    errs = validate_run(rec)
    if errs:
        print("schema 校验失败:", *errs, sep="\n  "); return 1
    path = dump_run(rec)
    print(f"已写入 run: {path}")
    print(f"  bench={cfg.bench}/{split}  method={cfg.method.name}  OCR=OFFICIAL-v051  (无 lmms ocr/prompt)")
    print(f"  samples={len(preds)}  accuracy={acc:.4f}")
    print(f"  token_schedule len={len(schedule)}  first3={schedule[:3]}  tlb={tlb}")
    print(f"  (目标: 官方 LLaVA-1.5-7B TextVQA val ≈ 0.582)")
    return 0


def run_one(wrap, img, text):
    pred = wrap.generate([img], text, bench_suffix="", max_new_tokens=128)
    return pred, list(wrap.token_schedule)


if __name__ == "__main__":
    raise SystemExit(main())
