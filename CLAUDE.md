# CLAUDE.md

本文件是本仓库对 Claude Code 的常驻指令。**每次开始任务前先读本文件和 `docs/PROTOCOL.md`；涉及时延、显存、吞吐或 FLOPs 时还必须读 `docs/EFFICIENCY_PROTOCOL.md`。**

---

## 1. 这个项目在做什么

研究 VLM 视觉 token 压缩（token pruning / compression）。

当前的科研判断是：现有方法（FastV、VisionZip、SparseVLM、PruMerge、VisPruner、MMTok 等）优化的都是**代理指标**——attention 分数、视觉显著性、token 多样性、集合覆盖度——而不是"答对这道题所必需的证据"。不同论文之间的对比又普遍不公平（budget 定义不一致、base model 不一致、prompt 不一致）。

所以项目的第一阶段目标不是提新方法，而是：

> 建一个**统一、可信、可复现**的实验台，在完全相同的 budget / base model / prompt 下重新测量若干代表性方法，并搞清楚它们各自到底在什么样本上失败。

**当前阶段：Phase 1（统一实验台 + 复现）。见 `docs/PHASE1_TASKS.md`。**

---

## 2. 铁律（violate 这些比不完成任务更严重）

### 2.1 研究诚信

1. **绝不编造、估计、或"合理推测"任何实验数字。** 没跑过就是没跑过。在表格里写 `NOT_RUN`，不要填一个看起来合理的数。
2. **绝不把论文里报告的数字当作本仓库的实验结果。** 原论文数字只能出现在 `paper_reported` 字段里，且必须标注来源（论文名 + 表号）。
3. **复现失败是一个有价值的结果。** 如果复现结果和原论文差距很大，如实记录差距和你的排查过程，不要反复调参"修"到对为止。反复调参本身就是在污染结论。
4. **绝不为了让结果好看而改动协议**：不改 budget 定义、不改 prompt、不换 split、不挑 seed、不挑 subset。
5. 报告结果时**同时报告不利证据**。如果 random baseline 打平了某个方法，这件事必须写在最显眼的位置，而不是藏在附录。

### 2.2 协议不可私自变更

`docs/PROTOCOL.md` 定义的质量与 budget 口径、`docs/EFFICIENCY_PROTOCOL.md` 定义的系统效率口径都是**冻结的**。

- 如果你认为协议有问题（很可能确实有），**先停下来，在回复里说明问题，等我确认**。
- 确认后：先改对应协议，在 `docs/EXPERIMENT_LOG.md` 记录变更原因，**然后**才改代码。
- 修改 base model、数据、prompt、解码、评分或 budget 语义时 bump `PROTOCOL` 版本；之前的质量结果全部标记为 `stale`。
- 新增不改变模型输出的效率测量项时 bump `EFFICIENCY_PROTOCOL` 版本；已有质量 run 继续有效，旧 profile run 只对其原效率协议有效。禁止为了补效率字段重跑或覆盖已有质量 run。

### 2.3 可复现性

每一次实验运行必须写入 `results/runs/<run_id>.json`，且包含：

```
run_id, timestamp, git_commit, protocol_version,
run_kind, measurement_protocol_version, linked_run_ids,
config_hash, full_config_dump,
base_model_id, base_model_revision,
method, budget_spec, seed,
hardware (gpu type, count, driver, torch/transformers/flash-attn version),
metrics {...}, artifacts_path
```

- `run_kind=quality` 时，`measurement_protocol_version` 可为空。
- `run_kind=efficiency` 时，必须按 `docs/EFFICIENCY_PROTOCOL.md` 写入原始逐样本计时，并用 `linked_run_ids` 关联质量结果。

- `results/` 目录 **append-only**。不要覆盖、不要删除、不要"清理"旧结果。
- 任何进入论文或汇总表的数字，必须能追溯到一个 `run_id`。
- 汇总表（`results/tables/*.md`）由脚本从 JSON 生成，**不要手写**。

### 2.4 不要跑在我前面

Phase 1 阶段：

- ❌ 不要发明新的压缩方法
- ❌ 不要发明新的评测指标（`docs/PROTOCOL.md` 里已定义的除外）
- ❌ 不要做"顺手优化一下效果"的改动
- ✅ 只做：基础设施、忠实复现、测量、诊断分析

如果你在实验中发现了有意思的现象，**写进 `docs/EXPERIMENT_LOG.md` 的 Observations 区并告诉我**，不要自己顺着做下去。

---

## 3. 仓库结构

```
.
├── CLAUDE.md                  # 本文件
├── README.md
├── docs/
│   ├── PROTOCOL.md            # 【核心】统一实验协议，冻结
│   ├── EFFICIENCY_PROTOCOL.md # 【核心】端到端效率测量协议，冻结
│   ├── RESEARCH_PLAN.md       # 研究路线与 go/no-go 判断点
│   ├── PHASE1_TASKS.md        # 当前阶段任务分解
│   └── EXPERIMENT_LOG.md      # 实验日志（人工 + Claude 共同维护）
├── configs/
│   ├── base/                  # base model 配置
│   ├── method/                # 各压缩方法配置
│   └── budget/                # budget 档位
├── src/tcbench/
│   ├── models/                # base model wrapper，统一 hook 点
│   ├── pruners/               # 统一 Pruner 接口 + 各方法实现
│   ├── eval/                  # benchmark runner
│   ├── metrics/               # TLB / latency / evidence recall
│   └── cli.py
├── scripts/                   # 一次性脚本、数据准备
├── results/
│   ├── runs/                  # 每次运行的 JSON，append-only
│   └── tables/                # 脚本生成的汇总表
└── third_party/               # 原始实现，只读参考，不直接 import
```

---

## 4. 统一 Pruner 接口（Phase 1 最重要的工程决策）

所有方法必须实现同一个接口。不允许为某个方法在 runner 里开特例分支。

```python
from typing import Literal, Protocol, Optional
import torch

class PruneResult:
    keep_index: torch.LongTensor      # [N_keep]，在原始 token 序列中的下标
    merge_map: Optional[torch.LongTensor]  # [N_orig]，每个原 token 被并入哪个 keep 位置；
                                           # 纯删除方法此处为 -1
    new_features: Optional[torch.Tensor]   # 若方法会修改 token 表示（如 merge/aggregate）

class Pruner(Protocol):
    stage: Literal["vit", "pre_llm", "in_llm"]

    def on_vit_layer(self, layer_idx, feats, meta) -> Optional[PruneResult]: ...
    def on_vit_output(self, feats, meta) -> Optional[PruneResult]: ...
    def on_llm_layer(self, layer_idx, hidden, attn, meta) -> Optional[PruneResult]: ...
```

**`merge_map` 是硬性要求**，不是可选装饰。Phase 2 的 evidence recall 指标需要知道"某个原始 patch 的信息是否还以某种形式存在"，纯删除方法和 merge 方法在这一点上必须可比。实现任何方法时如果发现 `merge_map` 难以给出，停下来问我，不要留空。

`meta` 里必须携带：`patch_grid (h, w)`、`image_token_span (start, end)`、`text_token_ids`、`question_token_span`。

---

## 5. 常用命令

```bash
# 环境
make setup

# 单次运行
python -m tcbench.cli run \
  --base configs/base/llava15_7b.yaml \
  --method configs/method/vispruner.yaml \
  --budget configs/budget/b128.yaml \
  --bench textvqa --seed 0

# 批量（budget 扫描）
bash scripts/sweep_budget.sh <method> <bench>

# 从 results/runs/*.json 生成汇总表
python -m tcbench.cli table --bench textvqa --out results/tables/textvqa.md

# 协议自检（改动代码后必跑）
pytest tests/ -v && python -m tcbench.cli verify-protocol
```

---

## 6. 代码约定

- Python 3.10+，type hints 必写，`ruff` + `black` 默认配置。
- 配置用 YAML + dataclass 解析，**不要用命令行参数覆盖配置里的实验语义项**（budget、prompt、split 只能来自配置文件，这样才能算 config_hash）。
- 随机性：`seed_everything(seed)` 统一入口，解码固定 `do_sample=False`。注意 kernel 层面仍可能有 ~0.1 分的非确定性，所以 random baseline 必须跑 3 个 seed 并报告均值 ± 标准差。
- `third_party/` 里的原始实现只作参考阅读。**不要直接 import 或魔改后 import**——所有方法在 `src/tcbench/pruners/` 里按统一接口重写，并在 docstring 里写明参照的原始文件与行号。
- 长实验用 `scripts/` 下的脚本提交，不要在交互式会话里跑超过 10 分钟的东西。

---

## 7. 沟通约定

- 每完成一个 `docs/PHASE1_TASKS.md` 里的任务，在该任务下勾选并追加一行结果摘要（含 run_id）。
- 遇到需要判断的岔路口（协议冲突、复现对不上、显存不够要改 batch），**停下来问**，给我 2–3 个选项和你的推荐，不要自己选。
- 报告实验结果时，格式：先给结论一句话，再给表，再给你对异常点的怀疑。不要先铺垫过程。
