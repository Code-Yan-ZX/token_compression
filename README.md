# tcbench — SCoRe-first Visual Token Compression Research

这个仓库现在服务于一个明确目标：**以当前已确认的强基线 SCoRe 为起点，提出并验证一个新的视觉 token 压缩方法。**

本项目不再以“横向分析过去若干方法”或“统一 benchmark 本身”作为论文目标。已有的 TLB、效率测量和历史复现结果继续作为可信实验基础，但它们只为方法研究服务。

## 当前研究锚点

- **SCoRe: Salience-Coverage Reduction for Vision Token Pruning in Vision-Language Models**
- 发表：**CVPR 2026**
- 定位：training-free、pre-LLM，通过统一的 salience–coverage 目标选择视觉 token
- 选择理由：在其论文的 LLaVA-1.5 设置中，它对比并优于 SparseVLM、VisionZip、VisPruner 等方法，是本项目当前最合适的公开强基线。
- 官方论文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_SCoRe_Salience-Coverage_Reduction_for_Vision_Token_Pruning_in_Vision-Language_Models_CVPR_2026_paper.html)

“SOTA”在本仓库中不是永久标签。开始写论文前必须重新检索同期工作；目前应把 SCoRe 称为**已核验的最强公开起点**，而不是声称它在所有模型、任务和成本口径下都绝对最强。

## 当前阶段

**S0：忠实复现 SCoRe；S1：检验其未解决问题是否足以支撑新方法。**

优先检验的研究假设是：SCoRe 使用固定保留数量 `K`，但不同图像与问题所需的视觉证据量不同。固定 `K` 可能造成简单样本浪费、复杂/OCR/细粒度样本证据不足。候选方法方向是把固定基数选择改成**由边际覆盖收益决定的自适应停止**，但只有在 Oracle 试验通过后才立项。

```text
忠实复现 SCoRe
      ↓
固定 K / 表示空间 / query 条件 / 真实开销压力测试
      ↓
Oracle 证明“自适应预算”存在可利用空间？
   否 → 换问题，不硬做方法
   是 → 冻结方法定义并实现
      ↓
与 SCoRe 及最近同期工作做同协议比较
```

## 当前文档

| 文件 | 作用 |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Claude Code 必读；当前执行边界与研究纪律 |
| [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) | 当前问题、假设、阶段门和论文目标 |
| [`docs/CURRENT_TASKS.md`](docs/CURRENT_TASKS.md) | 唯一有效的任务队列 |
| [`docs/SCORE_REPRO_PROTOCOL.md`](docs/SCORE_REPRO_PROTOCOL.md) | SCoRe 复现与受控比较协议 |
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | 历史 tcbench v0.3 质量协议，仅用于解释旧 run |
| [`docs/EFFICIENCY_PROTOCOL.md`](docs/EFFICIENCY_PROTOCOL.md) | 真实时延、吞吐、显存与 selector 开销协议 |
| [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) | append-only 的实验记录与决策 |

## 给执行者的最短指令

1. 不要继续旧的 VisPruner / SparseVLM / PruMerge 全矩阵扫描。
2. 先核验 SCoRe 的官方实现与论文配置，再做忠实复现。
3. “论文复现轨”和“tcbench 受控比较轨”必须分开，不能混用数字。
4. S0/S1 的分析是内部选题工具；最终目标是**方法论文**，不是分析论文。
5. 已有 `results/`、日志和历史协议不得删除或覆盖。

## 研究诚信

- 没跑过的数字写 `NOT_RUN`，不估计、不补齐。
- 论文报告值与本仓库实测值分字段记录。
- 每个结果必须追溯到 run_id、配置、代码版本和环境。
- 不因结果不好而改 prompt、split、seed 或 budget 语义。
