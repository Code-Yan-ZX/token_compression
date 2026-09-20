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

### [RUN] T1.0 · 环境骨架
- pytest 全绿；dummy run 生成合法 JSON（含 git_commit / config_hash / A40 指纹）。

### [RUN] ./T1.1 · 复现 LLaVA-1.5-7B 官方 TextVQA 基线（通过）
- **关键坑**：TextVQA 的 prompt 必须用官方 `llava_textvqa_val_v051_ocr.jsonl` 每条的 `text` 字段（`问题\nReference OCR token: <逗号分隔>\nAnswer the question...`）。曾用 lmms-lab 的 `ocr_tokens` 拼 prompt，得全量 val 40.87%，比官方 58.2 差 17 点（OCR 太稀疏模型读不出文字）。
- **修复**：改用官方 v051 prompt，gold/图像取自 TextVQA 数据集本身，彻底不用 lmms 的 ocr 逻辑。评测入口 `src/tcbench/eval/run_official_textvqa.py`。
- **结果**：全量 val `20260920_151736_87c203` = **57.85%**，官方 58.2%，差距 **-0.35 点 ≤ 0.5**，T1.1 baseline 通过。（dev1000 官方 prompt = 56.01%）
- 备注：模型权重 SHA256 与官方 `llava-hf/llava-1.5-7b-hf@main` 逐一相同，模板/评分逐字一致，权重与评测代码均无问题；根因纯在数据源 prompt。

---

## 下一步
- GQA（数据 config 待定）→ T1.2 Random baseline → T1.3-1.5 复现三种方法（VisPruner / SparseVLM / PruMerge）。