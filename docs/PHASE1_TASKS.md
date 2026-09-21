# PHASE 1 — 统一实验台与复现

**目标**：在完全相同的 budget / base model / prompt 下，得到 3 个代表性方法 + random baseline 的可信数字，并具备算 evidence recall 的能力。

**不是目标**：提出新方法、刷点、写论文。

**预计周期**：4–6 周。

任务按依赖顺序排列。**不要跳序做**——T1.2 的 random baseline 如果打平了所有方法，后面的任务需要重新规划。

---

## T1.0 · 仓库与环境骨架

- [x] 按 `CLAUDE.md` §3 建目录结构
- [x] `pyproject.toml` + `Makefile`（`make setup` / `make test` / `make lint`）
- [x] 配置解析：YAML → dataclass，实现 `config_hash`（对语义项做稳定哈希，忽略注释和 key 顺序）
- [x] `results/runs/*.json` 的 schema 定义 + 写入工具 + schema 校验测试
- [x] `seed_everything`，环境指纹采集（torch / transformers / flash-attn / driver / GPU 型号）

**结果**：`pytest` 11 passed；dummy run 生成合法 JSON（含 git_commit、config_hash、A40 指纹）。见 `[RUN] T1.0`。

**验收**：`pytest` 全绿；跑一个 dummy run 能生成合法 JSON。

---

## T1.1 · Base model wrapper 与 hook 点

- [x] 封装 LLaVA-1.5-7B，暴露三个 hook：ViT 逐层 / ViT 输出 / LLM 逐层
- [ ] `meta` 正确携带 `patch_grid=(24,24)`、`image_token_span`、`question_token_span`（span 已粗算，逐字段精确验证在进行）
- [x] 实现 `NoOpPruner`，验证挂上 hook 后输出与原生 pipeline **逐 token 完全一致**（实测最大绝对差 0.0 < 1e-3）
- [x] 逐层记录 `n_ℓ` → `token_schedule`（32 层，每层 576，TLB=1.0）
- [x] 跑通 TextVQA 官方 eval，得到无压缩 baseline（dev 子集；评测口径已对齐官方软评分/模板/OCR prompt）

**验收**：
1. ✅ `NoOpPruner` 与原生 pipeline 输出 logits 最大绝对差 0.0 < 1e-3；
2. ✅ 无压缩 baseline 与 LLaVA-1.5-7B 官方报告差距 ≤ 0.5 点：**全量 TextVQA val（官方 v051 prompt）= 57.85%**（run `20260920_151736_87c203`），官方 58.2%，差距 **-0.35 点**。T1.1 baseline 对齐通过。

> 进度注意（2026-09-20）：T1.1 定稿。**关键修复**：此前用 lmms-lab `ocr_tokens` 拼 prompt 得 40.87%（距官方 17 点）；改用官方 `llava_textvqa_val_v051_ocr.jsonl` 的 `text` 字段逐字做 prompt 后，全量 val = 57.85%（-0.35 点）。新评测入口 `src/tcbench/eval/run_official_textvqa.py`（纯官方 prompt，gold/图像取自 TextVQA 数据集，不用 lmms）。v0.1 贪心口径旧结果已标 `stale`。下一窗口：GQA → T1.2 Random baseline。

> ⚠️ 这一条不过就不要往下做。整个项目的可信度从这里开始。

---

## T1.2 · Random baseline（先于任何方法）

- [ ] `RandomPruner`（pre-LLM 阶段随机保留 N 个）
- [ ] `SpatialUniformPruner`（按 2D 网格均匀采样，比纯随机更强的"笨"baseline）
- [ ] 在全部 TLB 档位 × {TextVQA, GQA} × seed ∈ {0,1,2} 上跑完
- [ ] 输出 mean ± std

**验收**：得到一张完整的 random baseline 曲线，写入 `results/tables/random_baseline.md`。

> **为什么先做这个**："PruMerge / 重要性打分"类工作已指出"重要性打分有时不如随机"。如果我们的 random baseline 在某些档位就已经逼近了各方法的论文数字，那说明这个领域的比较基准本身有问题——**这件事比任何新方法都值钱**，且会直接改变后续研究方向。做完立刻报告给我。

---

## T1.3 · 复现方法 A：VisPruner（纯视觉早剪）

- [ ] 阅读 `third_party/vispruner/`，在 `src/tcbench/pruners/vispruner.py` 按统一接口重写
- [ ] docstring 注明参照的原始文件与关键行
- [ ] 正确实现 `merge_map`（若为纯删除，全部置 -1 并在 docstring 说明）
- [ ] 全 budget 档位 × {TextVQA, GQA}
- [ ] 对照 `PROTOCOL.md` §5 判定 OK / PARTIAL / FAILED

**代表性**：视觉显著性 + 去重，pre-LLM，免训练，不看 query。

---

## T1.4 · 复现方法 B：SparseVLM（query 引导，in-LLM）

- [ ] 同上流程
- [ ] **重点**：按 `PROTOCOL.md` §2.2 反解 `n_after` 以对齐 TLB，而非照搬原论文的名义 budget
- [ ] 逐层 `token_schedule` 与论文描述核对
- [ ] 记录 `INFEASIBLE` 的档位

**代表性**：用文本信号引导，必须先跑若干 LLM 层。与 A 构成"看不看 query"的对照。

---

## T1.5 · 复现方法 C：PruMerge（去冗余）+ Evidence Recall 基础设施

- [ ] 复现 PruMerge，流程同上
- [ ] **先验证 evidence bbox 数据可得性**（`PROTOCOL.md` §4.3 的警告），验证结论写进 `EXPERIMENT_LOG.md`
- [ ] 实现 `ER` 计算，含 merge 情形
- [ ] 为 T1.2–T1.5 的所有已有 run **补算 ER**（离线，不重跑推理：保存 `keep_index` / `merge_map` 即可）

**代表性**：按 token 间重复度剪枝，不依赖 attention 打分。与 A、B 构成三种不同的信号来源。

> 为了能补算 ER，T1.1 起每个 run 就要把 `keep_index` / `merge_map` 落盘到 `artifacts_path`。**这一点现在就要实现，不要等到 T1.5。**

---

## T1.6 · 统一对比表与 sanity check

- [ ] 补齐 DocVQA / POPE
- [ ] 生成主表：行 = {NoOp, Random, SpatialUniform, VisPruner, SparseVLM, PruMerge}，列 = TLB 档位 × bench
- [ ] 同时生成"名义 budget 表"作为对照，直观展示 §2.1 那个例子造成的差异有多大
- [ ] 质量表完成后，按 `EFFICIENCY_PROTOCOL e1.0` 建立独立 profile run；不重跑全量准确率
- [ ] 两条 attention 轨道分别成表：`matched_eager` / `best_compatible`
- [ ] 实测分阶段时延：vision encoder / compression / LLM prefill / decode / total
- [ ] 实测吞吐、peak allocated/reserved 显存、KV cache；FLOPs 无可靠计数器则标 `NOT_MEASURED`
- [ ] 生成质量—成本 Pareto 图，以及掉分 ≤1.0 / ≤2.0 点时的最省成本 operating point

**验收问题（必须在 `EXPERIMENT_LOG.md` 里逐条回答）**：
1. TLB 对齐后，各方法的相对排名和原论文一致吗？哪些变了？
2. 有没有方法在某些档位输给 random / spatial-uniform？
3. TextVQA 和 GQA 上的排名一致吗？不一致说明什么？
4. prefill 加速和端到端加速的差距有多大？
5. TLB / FLOPs 的排序是否与真实 TTFT、总时延和吞吐排序一致？
6. Random/Spatial 的低算法复杂度是否转化为更低的 selector overhead？

---

## T1.7 · 互斥错误集与 Oracle 上界【本阶段最重要】

这是决定整个项目走向的任务。

- [ ] 在固定档位（建议 TLB≈0.111）上，对每个样本记录三个方法的对错
- [ ] 统计互斥错误集：`A对B错` / `B对A错` / `全错` 的样本数
- [ ] 计算 **Oracle 上界**：每个样本取三个方法里最好的那个（用 ground truth 作弊）
- [ ] 抽 50 个互斥样本，导出图 + 问题 + 各方法保留的 token 可视化，**供人工检查**
- [ ] 各方法在这些样本上的 ER 分布对比

**Go / No-Go 判断点**：

| Oracle − 最强单一方法 | 结论 |
|---|---|
| < 1.0 点 | 样本级融合/路由这条路**不成立**，停止，回到诊断方向 |
| 1.0 – 3.0 点 | 空间有限，需要极低开销的 router 才划算 |
| > 3.0 点 | 存在真实的样本级异质性，值得继续 |

**这一步的可视化产物请交给我人工看，不要替我总结成一句话就过去了。** 创新点大概率藏在这 50 个样本里。

---

## T1.8 · 迁移验证（Qwen2.5-VL）

- [ ] 与我确认动态分辨率下的 budget 定义（协议 v0.2）
- [ ] 至少把 Random + 2 个方法迁过去
- [ ] 回答：LLaVA 上的结论在动态分辨率模型上还成立吗？

> 很多方法的结论在换 base model 后失效。如果这里出现不一致，**这本身可能就是论文的核心**。

---

## Phase 1 完成标志

1. 一张可信的、TLB 对齐的主表，每个数字可追溯到 run_id
2. random baseline 的位置清楚
3. Oracle 上界数字出来了，方向决策做完了
4. ER 能算，且已有初步分布观察
5. `EXPERIMENT_LOG.md` 里有 ≥ 5 条真实的 Observation
6. 一组符合 `EFFICIENCY_PROTOCOL e1.0` 的可追溯 profile run 与质量—成本 Pareto 图

**Phase 1 结束后再讨论方法设计。在此之前不要写任何 "我们提出……"。**
