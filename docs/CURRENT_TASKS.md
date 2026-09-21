# 当前任务：SCoRe 复现 → 缺口验证 → 方法立项

**状态：ACTIVE**
**本文件是唯一有效任务队列。** 旧 `PHASE1_TASKS.md` 已删除；不要继续其中的 T1.2–T1.8。

任务按依赖顺序执行。S0 未完成前，不进入 S1 结论；S1 未过阶段门前，不实现完整新方法。

---

## S0.0 · 论文与实现审计

- [ ] 保存 SCoRe 论文与补充材料的准确版本、发布日期和表格设置
- [ ] 核验是否存在作者官方代码；记录 URL、commit、license 与下载日期
- [ ] 若无可验证官方实现，明确标记 `AUTHOR_CODE_NOT_VERIFIED`，再制定 clean-room 实现清单
- [ ] 列出每个复现表格对应的模型、数据集、token budget、prompt、解码与评分器
- [ ] 把论文报告值录入独立 `paper_reported` 清单，附表号，禁止混入实测表
- [ ] 检索并记录最近直接相关工作，尤其是 adaptive budget、coverage、query-aware 和真实效率方向

**验收**：`docs/SCORE_REPRO_PROTOCOL.md` 中所有 `TBD` 被证据填充，并冻结一个协议版本。

---

## S0.1 · SCoRe clean-room 单元验证

- [ ] 写出论文目标函数与算法伪代码的逐项对应表
- [ ] 用小张量验证 salience、coverage、距离、初始化、tie-breaking 和停止条件
- [ ] 验证 `K=N` 时与 NoOp 特征/输出一致
- [ ] 验证每个样本恰好保留目标 `K`，`keep_index` 唯一、合法、可映射到 patch 网格
- [ ] 固定随机性并增加回归测试

**验收**：单元测试全绿；每个公式项都能追溯到论文或官方实现位置。

---

## S0.2 · Track A：论文忠实复现

- [ ] 先跑论文使用的最小代表子集，验证配置与指标管线
- [ ] 复现 `K ∈ {128, 64, 32}` 的关键结果
- [ ] 至少包含论文中与 VisPruner / SparseVLM 的直接比较设置
- [ ] 保存逐样本输出、keep_index、token 数、耗时和完整环境指纹
- [ ] 按协议判定 `OK / PARTIAL / FAILED`，不得靠反复调参追数

**验收**：关键设置与论文报告值差距 `≤1.0` 点为 `OK`；`1.0–2.0` 为 `PARTIAL`；`>2.0` 为 `FAILED` 并暂停后续扩展。

---

## S0.3 · Track B：接入 tcbench

- [ ] 实现统一 `Pruner` 接口，不在 runner 写方法特例
- [ ] 保存 `keep_index` / `merge_map` / 原始 token 映射
- [ ] 与 NoOp、Random、SpatialUniform 做小规模 sanity check
- [ ] 固定同模型、同 prompt、同解码、同 split 和同 token 数
- [ ] 记录 actual TLB，不把名义 `K` 当作真实成本
- [ ] 建立可链接的 quality run 与 efficiency run

**验收**：Track A 与 Track B 的配置和结果表明确分离；SCoRe 在 tcbench 中可复现运行且结果可追溯。

---

## S1.0 · 先修历史正确性风险

- [ ] 审计旧实验中 random 三个 seed 与 spatial 逐题答案 100% 相同的原因
- [ ] 直接比较各方法/seed 的 `keep_index`，确认剪枝子集确实不同
- [ ] 检查 gather、position id、image span、缓存和 kernel 路径是否忽略了 token 身份
- [ ] 将审计结论写入实验日志；未排除实现错误前，不引用旧 random 观察支撑论文假设

**验收**：能用可复现测试证明“相同答案”来自模型鲁棒性，或定位并修复实现错误。

---

## S1.1 · fixed-K Oracle

- [ ] 预先冻结样本集、质量容差和 go/no-go 阈值
- [ ] 在 `K ∈ {32, 64, 128, 192}` 上生成逐样本结果
- [ ] 定义每样本最小可接受 `K`，报告定义对答案评分的敏感性
- [ ] 构造仅用于上界的 Oracle，并匹配最强 fixed-K 的平均 token / actual TLB
- [ ] 在普通 VQA、OCR/文档、细粒度和复杂图像子组分别统计
- [ ] 抽样人工检查 Oracle 选择，排除评分噪声造成的虚假 gap

**Go**：同成本提升 `≥1.0–1.5` 点，或质量下降 `≤0.5` 点时节省 `≥15%` 平均 token / TTFT。
**No-Go**：否则否决自适应 `K`，转 S1.2 的其他缺口，不降低门槛。

---

## S1.2 · 竞争性缺口压力测试

- [ ] 表示空间：检查 coverage 与 OCR/细粒度证据保留的关系
- [ ] query 条件：统计同图不同问题的最优 token 集是否显著变化
- [ ] 真实开销：分辨率扩大时测 selector overhead、TTFT、总时延与显存
- [ ] 跨模型：在一个较新动态分辨率 VLM 上做最小迁移验证
- [ ] 对每条缺口写“证据强度 / 已有工作 / 可形成的单一机制 / 风险”四栏决策表

**验收**：只推荐一条最有证据、且与最近工作有明确差异的方向。

---

## S1.3 · 方法立项评审

- [ ] 重新检索提交时最近工作，确认不是 CoverPruner、VisPCO、StepPrune 等路线的直接重复
- [ ] 写出问题定义、核心公式、计算复杂度和不超过 3 个超参
- [ ] 明确最强 baseline、同成本口径和必要消融
- [ ] 明确 kill criterion：什么结果出现就停止该方法
- [ ] 经人工确认后，把状态从 `HYPOTHESIS` 改为 `APPROVED_METHOD`

**未完成本节前，不开始 S2。**

---

## S2 · 候选方法（当前仅为 HYPOTHESIS）

候选名：**Adaptive Marginal-Coverage Stopping for SCoRe**。

- [ ] 把 fixed-cardinality 目标改为 `f(S)-λ|S|` 或等价约束形式
- [ ] 使用归一化边际收益决定停止，避免额外训练 router
- [ ] 保证 batch 推理可实现，记录动态长度带来的 padding/调度成本
- [ ] 与 fixed-K SCoRe 做相同平均 token、相同 actual TLB、相同 TTFT 三组对照
- [ ] 消融停止信号、阈值归一化、最小/最大 K 与 overhead
- [ ] 在至少两个模型家族和多类任务上验证

只有 S1.3 标记 `APPROVED_METHOD` 后，这些项才变为可执行任务。

---

## 暂停项

以下内容不再自动执行：

- VisPruner / SparseVLM / PruMerge 全预算、全数据集 sweep；
- 为形成“统一横向分析论文”而补表；
- 把 TLB、ER 或 random baseline 观察作为最终论文主贡献；
- 在没有 SCoRe 的情况下只对比 2024/2025 方法并宣称 SOTA。

旧结果保留为历史资产，需要时按明确任务复用。
