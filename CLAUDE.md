# CLAUDE.md

这是本仓库对服务器 Claude Code 的常驻指令。开始任何工作前，按顺序阅读：

1. 本文件；
2. `docs/RESEARCH_PLAN.md`；
3. `docs/CURRENT_TASKS.md`；
4. 与本次实验对应的协议文件。

> **方向已于 2026-09-21 调整。** 旧的“横向复现 VisPruner / SparseVLM / PruMerge，做统一分析论文”路线已经终止。不要从旧日志、旧配置或历史协议中自动恢复那条任务链。`docs/CURRENT_TASKS.md` 是唯一有效任务队列。

---

## 1. 项目目标

本项目研究 VLM 视觉 token 压缩。论文目标是：

> 以 SCoRe（CVPR 2026）为强基线和方法起点，找到它尚未解决、且能被实验验证的问题，提出一个机制清楚的新方法，并推进质量—真实成本 Pareto 前沿。

不是当前目标：

- 写一篇只比较旧方法优缺点的分析论文；
- 为了表格完整而继续大规模扫旧 baseline；
- 把 TLB、ER 或 benchmark 基础设施本身包装成主要创新；
- 在未复现 SCoRe、未证明 gap 前直接堆模块。

当前阶段：

- **S0：SCoRe 忠实复现**；
- **S1：假设审计与 Oracle 判定**；
- S1 通过 go/no-go 后，才进入 **S2：新方法实现**。

当前首要假设：SCoRe 的固定 token 数 `K` 可能不能适配样本级视觉证据需求。候选方向是基于边际 salience–coverage 收益的自适应停止。它只是待验证假设，不是既定结论。

---

## 2. 执行边界

### 2.1 现在应该做

- 核验论文、补充材料、官方代码、commit、模型和评测配置；
- 在论文原设置下忠实复现 SCoRe；
- 把 SCoRe 接入 tcbench，在固定的受控条件下复测；
- 检查 fixed-`K`、表示空间、query 条件和 selector 开销等假设；
- 做每样本最小可接受 `K` 的 Oracle，判断自适应预算是否真有空间；
- 记录负结果，并据此停止或换问题。

### 2.2 现在不要做

- 不要自动继续旧的 T1.2–T1.8 或旧 sweep；
- 不要把 SCoRe 论文数字写成本仓库结果；
- 不要在 S1 阶段提前实现完整新方法；
- 不要把“query-aware”本身当创新：同期已有 CoverPruner 等相关路线，必须做新颖性核验；
- 不要声称“SCoRe 在所有设置下都是绝对 SOTA”。准确表述是“当前已核验、覆盖本项目所选旧基线的最强公开起点”；
- 不要只和 2024 年方法比较。开发阶段可用少量旧 baseline 做 sanity check，最终论文必须包含 SCoRe 和提交时最近的直接相关工作。

### 2.3 阶段门

S0 未通过复现验收，不进入 gap 结论。

S1 的 fixed-`K` Oracle 在**相同平均 token 或相同真实成本**下，若不能满足以下任一条件，则否决该方向：

- 相对最强固定 `K` 提升至少 `1.0–1.5` 个百分点；或
- 在质量下降不超过 `0.5` 点时，降低至少 `15%` 的平均视觉 token / TTFT。

阈值必须在看完整结果前冻结。Oracle 通过后，先在 `RESEARCH_PLAN.md` 写清方法目标、公式、超参和消融，再改方法代码。

---

## 3. 两条实验轨必须分开

### Track A：论文忠实复现

严格沿用 SCoRe 论文设置，用来回答“我们是否复现成功”。模型、数据、prompt、解码、token 数和指标以论文/官方实现为准。

### Track B：tcbench 受控比较

在本仓库统一设置下比较质量、actual TLB 和真实效率，用来回答“方法在同条件下是否更好”。历史 `PROTOCOL v0.3` 可解释旧 run，但新实验以 `docs/SCORE_REPRO_PROTOCOL.md` 冻结的设置为准。

两个 Track 的表格、run_kind 和结论必须显式区分。不得用 Track B 的结果宣称忠实复现，也不得用 Track A 的异构配置做公平排名。

---

## 4. 研究诚信与可复现性

1. 绝不编造、估计或合理推测实验数字；未运行写 `NOT_RUN`。
2. 论文值只能放在 `paper_reported`，并记录论文名、表号和版本。
3. 复现失败如实记录，不通过反复调参把数字“修”到目标附近。
4. 不因结果不好改 prompt、split、seed、解码或 budget 语义。
5. 有不利证据必须显著报告，包括 random 打平、selector 开销过大和跨模型失效。
6. `results/` append-only；不要覆盖、删除或清理历史 run。
7. 每次实验写入 `results/runs/<run_id>.json`，至少包含：

```text
run_id, timestamp, git_commit, protocol_version,
track, run_kind, measurement_protocol_version, linked_run_ids,
config_hash, full_config_dump,
base_model_id, base_model_revision,
method, budget_spec, seed,
hardware, metrics, artifacts_path
```

8. 汇总表由脚本从 run JSON 生成，不手写结果数字。
9. 长实验用 `scripts/` 提交；不要在交互会话里启动超过 10 分钟且无法追踪的任务。

---

## 5. 协议变更

- 运行前先确认使用 Track A 还是 Track B，以及对应协议版本。
- 如果模型、数据、prompt、解码、评分或 budget 语义要变，先停下说明；经确认后先改协议和实验日志，再改代码。
- 涉及时延、吞吐、显存、FLOPs 或 selector overhead，必须遵守 `docs/EFFICIENCY_PROTOCOL.md`。
- 质量 run 与效率 run 分开，通过 `linked_run_ids` 关联。
- 历史 `docs/PROTOCOL.md v0.3` 已冻结，仅用于解释旧阶段结果；不要据此自动发起旧实验矩阵。

---

## 6. 工程约定

所有方法继续使用统一 `Pruner` 接口；不要为 SCoRe 在 runner 中写不可审计的特殊分支。

```python
from typing import Literal, Optional, Protocol
import torch

class PruneResult:
    keep_index: torch.LongTensor
    merge_map: Optional[torch.LongTensor]
    new_features: Optional[torch.Tensor]

class Pruner(Protocol):
    stage: Literal["vit", "pre_llm", "in_llm"]

    def on_vit_layer(self, layer_idx, feats, meta) -> Optional[PruneResult]: ...
    def on_vit_output(self, feats, meta) -> Optional[PruneResult]: ...
    def on_llm_layer(self, layer_idx, hidden, attn, meta) -> Optional[PruneResult]: ...
```

- `keep_index` 必须保存，且能映射回原始 patch。
- `merge_map` 对纯删除方法统一为 `-1`；若方法聚合特征，必须给出真实映射。
- `meta` 至少包含 `patch_grid`、`image_token_span`、`text_token_ids`、`question_token_span`。
- Python 3.10+，type hints 必写；使用 `ruff` 和 `black`。
- YAML 语义配置进入 `config_hash`，不要用临时 CLI 参数偷改实验语义。
- 随机性统一由 `seed_everything(seed)` 管理；需要随机的 baseline 至少三 seed。
- `third_party/` 只读参考；实现放在 `src/tcbench/pruners/`，docstring 标明来源 commit 与关键文件。

---

## 7. 历史资产如何处理

- 旧的 VisPruner、SparseVLM、Random、SpatialUniform 结果是研究证据，保留并标注 `historical`。
- `docs/PROTOCOL.md v0.3` 和旧 run 不删除、不覆盖。
- 旧路线不再拥有执行优先级，也不能从 `EXPERIMENT_LOG.md` 的“下一步”恢复。
- 如果旧结果能低成本充当 sanity baseline，可以复用；如需新跑，必须由 `CURRENT_TASKS.md` 明确列出。

---

## 8. 沟通与验收

- 只执行 `docs/CURRENT_TASKS.md` 中的未完成项。
- 每完成一项，在该文件中勾选，并在 `EXPERIMENT_LOG.md` 追加摘要和 run_id。
- 遇到协议冲突、官方代码缺失、复现偏差超过阈值、显存不足或研究假设不成立时，停止扩展实验并报告。
- 报告顺序：一句话结论 → 关键数字/表 → 异常与风险 → 推荐下一步。
- 分析发现只服务于选出可做的方法；未经 go/no-go 不写“我们提出……”。
