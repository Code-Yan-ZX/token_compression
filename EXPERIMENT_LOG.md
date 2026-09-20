# EXPERIMENT_LOG.md

人工与 Claude Code 共同维护。**append-only，不要修改历史条目**（修正写新条目并引用旧条目 ID）。

三类条目：`[RUN]` 实验记录 / `[OBS]` 观察 / `[DEC]` 决策。

---

## 写作规则

- `[OBS]` 必须包含：现象、支撑它的 run_id、**你的怀疑**（可能是 bug 还是真实现象）、下一步验证手段。
- 不确定的写"不确定"。禁止用"可能是因为……"收尾后不给验证方案。
- 复现失败照实写。排查过程比结论有价值。
- Claude Code 每完成一个任务追加一条 `[RUN]`；发现异常追加 `[OBS]` 并**停下来告诉我**。

---

## 模板

```markdown
### [RUN] <日期> · <任务号> · <一句话>
- run_ids: `...`
- 配置: base=... method=... TLB=... bench=... seed=...
- 结果: <表或几个关键数字>
- 判定: OK / PARTIAL / FAILED（对照 PROTOCOL §5）
- 备注: <遇到的坑>

### [OBS] <日期> · <一句话现象>
- 证据: run_id ...
- 怀疑: <bug 还是真实现象？为什么？>
- 验证方案: <具体怎么排除 bug 的可能>
- 状态: 待验证 / 已确认 / 已排除

### [DEC] <日期> · <决策>
- 背景:
- 选项: A / B / C
- 选择与理由:
- 影响范围: <是否需要 bump PROTOCOL 版本>
```

---

## 日志

### [DEC] (初始) · 采用 TLB 作为统一 budget 口径
- 背景：各论文的 "N tokens" 含义不一致，pre-LLM 与 in-LLM 方法不可比。
- 选项：A) 按保留 token 数对齐；B) 按 FLOPs 对齐；C) 按 TLB 对齐。
- 选择 C。A 系统性地偏袒 in-LLM 方法（见 PROTOCOL §2.1 例子）；B 依赖具体实现且难以复核；C 可从 `token_schedule` 直接验算，不可作弊。
- 已知局限记录在 PROTOCOL §2.3，论文中必须写明。
- 影响：PROTOCOL v0.1。

### [DEC] (初始) · 精度实验统一用 eager attention，时延实验分开
- 背景：FastV / SparseVLM 需读 attention map，FlashAttention 不返回权重。
- 选择：精度全用 eager 保证可比；时延单独一组，各方法用各自可用的最快实现并记录。
- 风险：两组结果若被混用会得出错误结论，已在 PROTOCOL §1.1 加警告。

<!-- 新条目追加到此处下方 -->
