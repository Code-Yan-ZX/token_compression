# EXPERIMENT_LOG.md

人工与 Claude Code 共同维护。仅记录正确的结论；误导性的排查过程不保留。

三类条目：`[RUN]` 实验记录 / `[OBS]` 观察 / `[DEC]` 决策。

---

## 日志

### [DEC] 统一 budget 用 TLB（Token-Layer Budget）
- 各论文 "N tokens" 含义不一，pre-LLM 与 in-LLM 方法不可比。以 `token_schedule` 实测验算 TLB，不可作弊。PROTOCOL v0.1。

### [DEC] 主模型定 LLaVA-1.5-7B，三方法复现优先
- 三方法（VisPruner / SparseVLM / PruMerge）原论文均在 LLaVA-1.5 上报数，选它才能对齐原论文。权重经 modelscope 下载到本机。方法 C 顺带正名：DART → **PruMerge**（403 2403.15388）。

### [DEC] 评测口径逐项对齐 LLaVA 官方（PROTOCOL v0.2）
- 软评分 / vicuna_v1 模板 / 官方 TextVQA prompt / 解码 temp=0.2。v0.1 贪心口径旧结果标 `stale`。

### [DEC] 评测口径改回贪心 + 三方法审计（PROTOCOL v0.3，2026-09-21）
- **审计结论（读完三个原仓库后）**：
  - VisPruner（`vispruner`@`aefa01a`）：预 LLM 剪枝，全量 `llava-v1.5-7b` 无 LoRA，`--visual_token_num T` 任意对齐，eager。
  - SparseVLM（`SparseVLMs`，默认 v1.0 非 v2.0 的 SparseVLM+）：in-LLM 剪枝，`RETAIN_TOKN∈{192,128,96,64}`，全量 `llava-v1.5-7b` 无 LoRA。
  - PruMerge（`LLaVA-PruMerge`@`8989c43`）：默认激活 **PruMerge+**（自适应），**必须**用 `vicuna-7b-v1.5`+LoRA，独立 base，**不可严格同 base/同预算对齐** → 单独报告。
  - 三方法 eval 脚本均为 `--temperature 0`（贪心）；官方 LLaVA 58.2 亦贪心。
- **决策**：PROTOCOL bump 到 v0.3，主实验统一贪心 `do_sample=False,temp=0,num_beams=1,max_new_tokens=128`。基础权重用本地已校验的 HF `llava-hf/llava-1.5-7b-hf`，转成三 repo 所需遗留格式 `llava-v1.5-7b`（同一套权重）。
- v0.2 采样口径所有结果（含 57.85% 无压缩基线）标 `stale`，需在 v0.3 下重跑无压缩基线。PruMerge 单独报告其自适应 TLB。

### [RUN] T1.0 · 环境骨架
- pytest 全绿；dummy run 生成合法 JSON（含 git_commit / config_hash / A40 指纹）。

### [RUN] ./T1.1 · 复现 LLaVA-1.5-7B 官方 TextVQA 基线（**v0.2 采样口径，已标 STALE***）
- **关键坑**：TextVQA 的 prompt 必须用官方 `llava_textvqa_val_v051_ocr.jsonl` 每条的 `text` 字段（`问题\nReference OCR token: <逗号分隔>\nAnswer the question...`）。曾用 lmms-lab 的 `ocr_tokens` 拼 prompt，得全量 val 40.87%，比官方 58.2 差 17 点（OCR 太稀疏模型读不出文字）。
- **修复**：改用官方 v051 prompt，gold/图像取自 TextVQA 数据集本身，彻底不用 lmms 的 ocr 逻辑。评测入口 `src/tcbench/eval/run_official_textvqa.py`。
- **结果**：全量 val `20260920_151736_87c203` = **57.85%**（v0.2 采样 temp=0.2）。官方 58.2%（贪心）。差距 **-0.35 点 ≤ 0.5**。
- 备注：模型权重 SHA256 与官方 `llava-hf/llava-1.5-7b-hf@main` 逐一相同，模板/评分逐字一致，权重与评测代码均无问题；根因纯在数据源 prompt。
- **※ v0.3 把解码改回贪心后，57.85% 不再代表无压缩基线**，需重跑 v0.3 贪心无压缩基线。

---

## 下一步
- GQA（数据 config 待定）→ T1.2 Random baseline → T1.3-1.5 复现三种方法（VisPruner / SparseVLM / PruMerge）。
### [OBS] 2026-09-21 · 配对评测：TLB 对齐后方法 vs 随机对照（重要）
- 同 base/greedy/逐题实记 TLB 下，TextVQA+GQA 两套：
  - SparseVLM 三档真实 TLB(0.359/0.252/0.143) 全高于名义(0.33/0.22/0.11)。
  - TLB≈0.222 档：TextVQA 上方法只领先随机 ≤0.42（VisPruner 56.66, SparseVLM 56.36, random/spatial 56.24）；**GQA 上随机(58.08) 反超 VisPruner(57.88) 与 SparseVLM(57.78)**。
  - 逐题答案三个随机 seed 100% 相同、且与 spatial 相同 → 此档"保留哪个 token"对逐题答案鲁棒，数量(TLB)起主导。**待 keep_index 审计确认**：是否为 kernel 层把不同子集输入抹平，还是方法真与子集身份无关。
  - 高压缩处方法分化明显：SparseVLM"64"(0.143)GQA=52.97 vs VisPruner(0.111)=55.75。
