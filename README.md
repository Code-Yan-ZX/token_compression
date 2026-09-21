# tcbench — Visual Token Compression, Measured Honestly

一个 VLM 视觉 token 压缩的**统一实验台**与研究仓库。

## 这个仓库在解决什么

现有 token 压缩论文之间的对比大多不可比：budget 定义不一致、base model 不一致、prompt 不一致、random baseline 缺失或过弱。

最典型的问题：同样是"保留 64 个 visual token"，pre-LLM 剪枝方法的 32 层都只跑 64 个，而 FastV 前 K 层跑满 576 个——实际视觉 token 计算量相差 50% 以上，却被放在同一列里比较。

本仓库引入 **Token-Layer Budget (TLB)** 作为统一口径，在完全相同的条件下重测代表性方法，并提供 **Evidence Recall (ER)** 诊断指标。系统效率单独实测 TTFT、端到端时延、吞吐、显存、KV cache 与方法自身开销，最终用质量—成本 Pareto 曲线比较方法。

## 文档

| 文件 | 读它如果你想… |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | 了解协作规则（Claude Code 必读） |
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | **理解 TLB / ER 的定义，以及为什么必须这么定**（核心） |
| [`docs/EFFICIENCY_PROTOCOL.md`](docs/EFFICIENCY_PROTOCOL.md) | **理解真实时延、吞吐和显存怎样公平测量**（核心） |
| [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) | 了解研究判断与 go/no-go 决策点 |
| [`docs/PHASE1_TASKS.md`](docs/PHASE1_TASKS.md) | 知道现在该做什么 |
| [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) | 看实验记录与观察 |

## 当前状态

**Phase 1：统一实验台 + 复现。** 尚无结论性结果。

覆盖方法：`NoOp` / `Random` / `SpatialUniform` / `VisPruner` / `SparseVLM` / `PruMerge`
主模型：LLaVA-1.5-7B（迁移验证：Qwen3-VL-8B）

## 快速开始

```bash
make setup

python -m tcbench.cli run \
  --base configs/base/llava15_7b.yaml \
  --method configs/method/random.yaml \
  --budget configs/budget/tlb_0111.yaml \
  --bench textvqa --seed 0

python -m tcbench.cli table --bench textvqa --out results/tables/textvqa.md
```

## 原则

1. 没跑过的数字不写进表里。
2. `results/` append-only，每个数字可追溯到 run_id。
3. 协议变更走文档流程，不在代码里悄悄改。
4. 复现失败如实报告，不调参"修"到对为止。
