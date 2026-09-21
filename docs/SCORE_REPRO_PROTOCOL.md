# SCoRe 复现与受控比较协议

**版本：`draft-0.1`**
**状态：DRAFT — S0.0 完成前不得启动结论性长实验**
**适用方法：SCoRe 及其后续候选方法**

---

## 1. 目的

本协议把两个容易混淆的问题分开：

- **Track A / faithful reproduction**：我们是否忠实复现了 SCoRe 论文？
- **Track B / controlled comparison**：在本仓库相同条件下，SCoRe 和候选方法谁更好？

任何 run 必须声明 `track=A` 或 `track=B`。两条轨道的结果不得混在一列，也不得互相替代结论。

---

## 2. 论文身份

| 项 | 值 |
|---|---|
| title | SCoRe: Salience-Coverage Reduction for Vision Token Pruning in Vision-Language Models |
| authors | Tong Xu, Hailong Shi, Xingyu Gao |
| venue | CVPR 2026 |
| official paper | https://openaccess.thecvf.com/content/CVPR2026/html/Xu_SCoRe_Salience-Coverage_Reduction_for_Vision_Token_Pruning_in_Vision-Language_Models_CVPR_2026_paper.html |
| paper version/hash | `TBD` |
| supplementary version/hash | `TBD` |
| verified author code | `TBD — must verify, do not assume` |
| code commit/license | `TBD` |

如果没有找到可验证的作者官方代码，记录 `AUTHOR_CODE_NOT_VERIFIED`，并把实现称为 clean-room reproduction，不得称“官方实现”。

---

## 3. Track A：忠实复现

S0.0 必须从论文、补充材料或官方实现逐项填完下表。无法确认的字段保持 `TBD`，不能用经验猜测。

| 字段 | 冻结值 | 证据位置 |
|---|---|---|
| base model / revision | `TBD` | `TBD` |
| vision tower / input resolution | `TBD` | `TBD` |
| benchmark / split | `TBD` | `TBD` |
| prompt / conversation template | `TBD` | `TBD` |
| decoding | `TBD` | `TBD` |
| evaluator / metric | `TBD` | `TBD` |
| token budgets | `{128, 64, 32}` initially; verify | `TBD` |
| salience definition | `TBD` | `TBD` |
| coverage distance / feature normalization | `TBD` | `TBD` |
| initialization / tie-breaking | `TBD` | `TBD` |
| batch size / dtype / attention backend | `TBD` | `TBD` |

### Track A 验收

- 无压缩 baseline 先与论文/官方管线对齐；
- 关键设置差距 `≤1.0` 点：`OK`；
- `1.0–2.0` 点：`PARTIAL`，列出排查项；
- `>2.0` 点：`FAILED`，停止扩展，不通过搜索超参追数；
- 同时核对每样本保留 token 数、token 位置与算法中间量，而不只看总分。

---

## 4. Track B：tcbench 受控比较

Track B 在启动前另行冻结下列配置：

| 字段 | 冻结值 |
|---|---|
| base model / revision | `TBD` |
| benchmark / split | `TBD` |
| prompt / decoding / evaluator | `TBD` |
| nominal K | `{32, 64, 128, 192}` initially |
| seeds | deterministic methods: 1; random baseline: `{0,1,2}` |
| quality protocol version | `score-b1.0` after freeze |
| efficiency protocol | `EFFICIENCY_PROTOCOL e1.0` or later frozen version |

受控比较要求：

1. 同一模型、权重 revision、输入、prompt、解码和评分器；
2. 同时报告 nominal `K`、逐层 `token_schedule`、actual TLB；
3. 动态预算方法报告 mean / median / P10 / P90 token 数；
4. 质量按相同平均 token、相同 actual TLB、相同真实 TTFT 三种口径比较；
5. 真实效率拆分 vision、selector/compression、LLM prefill、decode 和 total；
6. 动态长度 batching 的 padding 和调度损失必须计入端到端结果。

---

## 5. 必存产物

每个样本至少保存：

```text
sample_id, prediction, score,
nominal_k, actual_k, keep_index,
token_schedule, actual_tlb,
selector_time_ms, prefill_time_ms, total_time_ms
```

每个 run 还必须保存完整配置、代码 commit、环境指纹、原始逐样本输出和聚合脚本版本。`results/` append-only。

---

## 6. fixed-K Oracle 协议

Oracle 只用于判断研究空间，不是可部署方法。

1. 在看结果前冻结样本集、`K` 集合、质量容差和 go/no-go 阈值；
2. 对每个样本在 `{32,64,128,192}` 中找满足质量容差的最小 `K`；
3. 与最强 fixed-K 在相同平均 token 或相同真实成本下比较；
4. 人工审查一批 Oracle 决策，排除开放式评分噪声；
5. 分普通 VQA、OCR/文档、细粒度和复杂度子组报告，但不靠事后挑子组决定 Go。

Go 条件：同成本提升 `≥1.0–1.5` 点，或质量下降 `≤0.5` 点时节省 `≥15%` 平均 token / TTFT。

---

## 7. 协议冻结清单

从 `draft-0.1` 升为可执行版本前，必须完成：

- [ ] §2 所有来源和代码身份已核验
- [ ] Track A 所有 `TBD` 已填
- [ ] Track B 所有 `TBD` 已填
- [ ] paper_reported 值已附表号/页码
- [ ] 实现单元测试已通过
- [ ] Oracle 阈值和样本集已预先冻结
- [ ] 在 `EXPERIMENT_LOG.md` 记录冻结决策
