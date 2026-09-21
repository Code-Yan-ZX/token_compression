# PROTOCOL.md — 统一实验协议

**协议版本：`v0.3`**
**状态：冻结。变更流程见 `CLAUDE.md` §2.2。**

本文件存在的唯一理由：现有 token 压缩论文之间的对比大多不可比。本项目所有结论的可信度都建立在本文件上。

---

## 0. 为什么需要这份文件

同样一句"保留 128 个 visual token"，在不同论文里含义完全不同：

- VisPruner / PruMerge：在 LLM 第 0 层之前就只剩 128 个，**全部 32 层**都只跑 128 个。
- FastV：前 K 层跑满 576 个，第 K 层之后才剩 128 个。
- SparseVLM：逐层递减，每层数量都不同。
- DyVTE：visual token 在某层整体退出，之后为 0。

把它们放在"128"这一列里横向比较，是**错误的**。下面 §2 给出统一口径。

---

## 1. Base model

### 1.1 主模型（Phase 1 全部实验）

| 项 | 值 |
|---|---|
| model_id | `llava-hf/llava-1.5-7b-hf` |
| revision | 必须在 config 里锁定具体 commit hash，不得用 `main` |
| vision encoder | CLIP ViT-L/14-336 |
| 视觉 token 数 `N₀` | **576**（固定分辨率，无 anyres） |
| LLM 层数 `L` | **32** |
| dtype | `float16` |
| attn 实现 | `eager`（**见下方警告**） |

> ⚠️ **注意力实现的坑**：FastV / SparseVLM 这类方法需要读取 attention map，而 FlashAttention 默认不返回 attention weights。为保证所有方法在同一实现下测**精度**，Phase 1 精度实验统一用 `eager`。
> 但 `eager` 会让**时延**测量失真（VisPruner 的一个卖点正是兼容高效 attention）。因此：
> - 精度实验：`eager`，所有方法一致。
> - 时延实验：单独一组，每个方法用它**实际能用的最快实现**，并在结果里记录用的是哪种。两组结果不得混在同一张表。

选 LLaVA-1.5 作为主模型的理由：表格里几乎所有方法都在它上面报过数，复现失败时能最快定位是我们的问题还是协议的问题。它**不是**最终目标模型。

### 1.2 迁移验证模型（Phase 1 末尾）

`Qwen2.5-VL-7B-Instruct` — 动态分辨率，`N₀` 随图像变化。

> 动态分辨率会让下面所有"固定 N"的 budget 定义失效，需要改成按比例定义。**这部分协议留到 Phase 1 T1.7 再定，现在不要提前实现。**

### 1.3 Prompt 与解码

- Prompt 模板：严格使用 base model 官方 chat template，**不做任何针对方法的调整**。
- 每个 benchmark 的 instruction 后缀（如 `Answer the question using a single word or phrase.`）统一写在 `configs/base/*.yaml` 的 `bench_suffix` 字段，所有方法共用。
- 解码：主实验统一 **贪心** `do_sample=False`，`temperature=0`，`num_beams=1`，`max_new_tokens=128`。这是三方法原仓库 eval 脚本（均 `--temperature 0`）与 LLaVA 官方 58.2 的同一口径，唯一变量是各方法算法本身。
- `seed=0`（random baseline 额外跑 `seed ∈ {0,1,2}`）。

---

## 2. Budget 的统一定义

### 2.1 主口径：Token-Layer Budget (TLB)

设 `n_ℓ` 为 LLM 第 ℓ 层**输入端**存在的 visual token 数量，`N₀` 为未压缩时的数量，`L` 为 LLM 层数：

```
TLB = (1 / (L · N₀)) · Σ_{ℓ=1..L} n_ℓ
```

TLB ∈ (0, 1]，含义是"相对于不压缩，LLM 在 visual token 上花掉的线性计算比例"。

**为什么必须用它 —— 一个具体例子（LLaVA-1.5，N₀=576，L=32）：**

| 方法 | 名义 budget | 实际 TLB |
|---|---|---|
| VisPruner，pre-LLM 保留 64 | "64" | 64/576 = **0.111** |
| FastV，K=2 层后剪到 64 | "64" | (2·576 + 30·64) / (32·576) = **0.167** |
| FastV，K=5 层后剪到 64 | "64" | (5·576 + 27·64) / (32·576) = **0.180** |

也就是说，**在"同样保留 64 个 token"的表述下，FastV 实际多花了 50%–60% 的 visual token 计算量。** 任何把这两者并列的表格都在高估 FastV。这是 Phase 1 要纠正的第一件事。

### 2.2 Budget 档位（所有方法共用）

TLB 目标档位：`{1.000 (无压缩), 0.333, 0.222, 0.111, 0.056}`

对应 pre-LLM 方法的保留数：`{576, 192, 128, 64, 32}`。

对 in-LLM 方法（FastV、SparseVLM）：给定目标 TLB 和该方法自己的剪枝层 K，**反解**出剪枝后应保留的 token 数 `n_after`：

```
n_after = (TLB · L · N₀ − K · N₀) / (L − K)
```

如果反解出 `n_after ≤ 0`，说明该方法**在这个 TLB 档位下物理上不可达**，在表格中记为 `INFEASIBLE`，不得用最接近的档位替代。这件事本身就是一个结论。

### 2.3 TLB 的已知局限（必须在论文里写明）

1. TLB 是**线性**近似。attention 的代价对 token 数是二次的，FFN 是线性的。TLB 低估了在深层保留大量 token 的代价。
2. TLB **不计入 vision encoder 的开销**。ViT 内部剪枝的方法（QuietPrune 类）省的是 encoder 的钱，TLB 看不见。Phase 1 不涉及这类方法；若后续涉及，须单独报告 `encoder_flops`。
3. TLB 不等于时延。**时延必须实测，见 §4.2。** 不允许用 TLB 或 FLOPs 代替时延下结论。

### 2.4 记录要求

runner 必须逐层记录 `n_ℓ` 并存入结果 JSON 的 `token_schedule` 字段（长度 L 的数组）。TLB 由此数组计算得出，**不允许从配置里"应该是多少"反推**。这是防止实现 bug 的主要手段。

---

## 3. Benchmark 与 split

| bench | split | 指标 | 作用 |
|---|---|---|---|
| GQA | `testdev_balanced` | accuracy | 通用推理，低分辨率需求 |
| TextVQA | `val` | VQA accuracy | **场景文字，对剪枝最敏感** |
| DocVQA | `val` | ANLS | **文档，证据稀疏** |
| POPE | `random/popular/adversarial` | F1 | 幻觉，检查压缩是否放大幻觉 |
| MME | `perception` | score | 覆盖面 |

- Phase 1 必跑：**TextVQA + GQA**（一个敏感、一个不敏感，对比最有信息量）。其余在 T1.6 补齐。
- 开发期可用固定子集加速，但**子集必须是预先随机抽样并落盘的固定 id 列表**（`data/subsets/<bench>_dev1k.json`），不得每次重抽。子集结果在 JSON 里标 `split: "dev1k"`，**不得与全量结果同表对比**。
- 评测代码统一用官方 eval 脚本，放在 `src/tcbench/eval/official/`，不要自己重写指标。
- **TextVQA 的 prompt 必须用官方 `data/ocr/llava_textvqa_val_v051_ocr.jsonl` 每条的 `text` 字段**（含 `Reference OCR token:`），入口 `src/tcbench/eval/run_official_textvqa.py`。不得用 lmms-lab `ocr_tokens` 等替代源——那会导致基线偏离官方 ~17 点（T1.1 已证实）。

---

## 4. 必须报告的指标

质量 run 至少报告 §4.1；独立的效率 profile run 按 §4.2 报告系统指标。诊断指标按
任务进度离线补算，不得把未测量字段填成估计值。

### 4.1 精度
- benchmark 主指标
- `token_schedule`、实测 `TLB`

### 4.2 效率（独立 profile run，非估算）

完整定义见 [`EFFICIENCY_PROTOCOL.md`](EFFICIENCY_PROTOCOL.md) `e1.0`。必须报告分阶段时延、
TTFT、总时延、吞吐、峰值 allocated/reserved 显存、KV cache、选择器开销和 attention
实现，并分别运行 `matched_eager` 与 `best_compatible` 两条轨道。

> TLB 是统一 token 计算口径，不是真实系统性能。很多方法 prefill 快了但总时间没变；
> 排序、聚类、merge/recycle 也可能抵消节省。本项目不允许只报 TLB、FLOPs 或 prefill。

效率协议是加法测量扩展，不改变本文件 v0.3 的模型输出语义。已有和正在运行的 v0.3
质量实验继续有效，效率 run 通过 `linked_run_ids` 与之关联。

### 4.3 诊断指标（Phase 1 建基础设施，Phase 2 重用）

**Evidence Recall (ER)** — 保留下来的 token 里，有多少覆盖了答案所需的证据区域：

```
ER = |{原始 token t : t 与 evidence 区域有重叠，且 t 的信息被保留}| / |{原始 token t : t 与 evidence 区域有重叠}|
```

"信息被保留"的定义（这就是为什么 `merge_map` 是硬性要求）：
- 纯删除方法：`t ∈ keep_index`
- merge 方法：`t ∈ keep_index` **或** `merge_map[t] ≥ 0`（被并入某个保留 token）

evidence 区域的来源：
- TextVQA：数据集自带的 OCR token bbox；取其中文本与 ground-truth answer 匹配上的那些。
- DocVQA：需要用 OCR 结果定位 answer 字符串得到 bbox。

> ⚠️ 以上两条数据来源**必须先实际验证可行性再实现**（T1.5 的第一步就是验证）。如果某个数据集拿不到可靠的 evidence bbox，如实记录，不要用启发式凑一个出来。

ER 是本项目后续论证的关键：假设是"coverage 高但 ER 低"能解释掉分。**Phase 1 只负责把 ER 算准，不负责下结论。**

---

## 5. 复现验收标准

某方法视为"复现成功"，需同时满足：

1. 在该方法原论文报告过的 (base model, bench, budget) 组合上，本仓库结果与论文数字差距 **≤ 1.0 个点**；
2. `token_schedule` 与该方法的设计描述一致（逐层数量对得上）；
3. 无压缩（TLB=1.0）时，本仓库跑出的 base model 分数与官方报告差距 **≤ 0.5 个点**（这一条先过，否则问题在实验台不在方法）。

差距在 1.0–2.0 点之间：标 `PARTIAL`，记录已排查项，继续推进，不要卡住。
差距 > 2.0 点：标 `FAILED`，**停下来报告**，不要自行调参。

---

## 变更记录

| 版本 | 日期 | 变更 | 原因 |
|---|---|---|---|
| v0.1 | (初始) | 建立协议 | — |
| v0.1 | 2026-09-20 | 方法 C 命名纠错：DART → **PruMerge**（arXiv 2403.15388）。不改任何协议规则语义。 | `docs/EXPERIMENT_LOG.md` [DEC] 2026-09-20 |
| v0.2 | 2026-09-20 | 解码口径与 LLaVA 官方对齐：`do_sample=False` → `do_sample=True, temperature=0.2`；`max_new_tokens` 64 → 128。目的：让无压缩 baseline 与三方法论文（均基于 LLaVA 官方评测）完全同口径，唯一变量是各方法算法。 | PROTOCOL §1.3 |
| v0.3 | 2026-09-21 | 解码口径回落 **贪心**：`do_sample=True, temperature=0.2` → `do_sample=False, temperature=0, num_beams=1, max_new_tokens=128`。原因：三篇方法原仓库 eval 脚本均为 `--temperature 0`（贪心），LLaVA 官方 58.2 亦为贪心；v0.2 的采样口径与它们不同。v0.2 采样口径下所有结果（含 57.85% 无压缩基线）标 `stale`，不再与 v0.3 结果同表对比。 | PROTOCOL §1.3；[DEC] 2026-09-21 |

> 2026-09-21：端到端效率测量独立版本化为 `EFFICIENCY_PROTOCOL e1.0`。这是加法扩展，
> 不 bump 本质量协议，也不使 v0.3 质量结果 stale。
