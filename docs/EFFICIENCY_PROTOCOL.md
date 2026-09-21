# EFFICIENCY_PROTOCOL.md — 端到端效率评测协议

**效率协议版本：`e1.0`**

**配套质量协议：`PROTOCOL v0.3`**

**状态：冻结。**

本协议是 `PROTOCOL.md` 的加法扩展。它不改变 base model、数据、prompt、解码、
官方评分或 TLB 定义，因此已有和正在运行的 v0.3 质量实验继续有效，无需重跑。
效率实验使用新的 profile run，并通过 `linked_run_ids` 关联对应的质量 run。

---

## 1. 研究问题与主结论口径

视觉 token 压缩方法不以单独提高 benchmark 分数为目标。主比较必须回答：

1. **同质量时省多少成本**：在相对 NoOp 掉分不超过给定阈值时，时延、显存、
   FLOPs 和 TLB 分别降低多少？
2. **同成本时保留多少质量**：在相同实测时延或相同 actual TLB 下，哪个方法分数更高？
3. **代理成本是否转化为真实收益**：TLB / FLOPs 的下降是否带来 TTFT、总时延、
   吞吐量和显存的改善？

任何“更高效”“更快”“更省显存”的结论必须来自本协议的实测指标。不得用名义 token
数、TLB 或 FLOPs 替代真实时延和显存。

### 1.1 必须报告的质量派生量

设无压缩分数为 `S0`，压缩方法分数为 `S`：

```text
quality_drop_pp  = 100 * (S0 - S)
quality_retention = S / S0
```

主表至少给出 `quality_drop_pp`。`quality_retention` 用于跨 benchmark 汇总，不能掩盖
原始分数和绝对掉点。

### 1.2 Pareto 判定

在同一 benchmark、硬件、attention track 和 batch size 下，若 A 满足：

```text
score_A >= score_B
cost_A  <= cost_B
```

且至少一项严格更好，则 A Pareto 支配 B。成本分别用 actual TLB、TTFT、总时延、
峰值显存作图，不合并成一个人为加权总分。

---

## 2. 两条 attention 实现轨道

### 2.1 `matched_eager`：算法归因轨道

- 所有方法统一 `eager` attention。
- 用于把效率差异归因于 token schedule 和方法本身。
- 必须关联同配置的 v0.3 质量 run。

### 2.2 `best_compatible`：实际部署轨道

- 每个方法使用其正确实现允许的最快 attention kernel。
- 必须记录准确的 `attn_impl`、版本和 fallback；静默回退视为无效 run。
- 例如需要读取 LLM attention weights 的方法可能无法使用 FlashAttention，这一限制本身
  属于方法成本，不能为“公平”而隐藏。

两条轨道分别成表、分别画 Pareto 曲线，禁止把一条轨道的准确率和另一条轨道的时延
拼成一个点。

---

## 3. 时延边界与必测指标

所有 GPU 计时在计时区间前后调用 `torch.cuda.synchronize()`。模型加载、首次编译、数据集
读取和磁盘 I/O 不计入主时延。图像预处理需单独记录，不能时而包含、时而排除。

每个样本记录以下分量，单位均为毫秒：

| 字段 | 起止边界 |
|---|---|
| `image_preprocess_ms` | 原始 PIL/bytes 到可输入模型的 CPU/GPU tensor |
| `vision_encoder_ms` | vision encoder forward |
| `compression_ms` | 打分、排序、采样、聚类、merge/recycle 等方法额外步骤 |
| `llm_prefill_ms` | 压缩后序列进入 LLM 到首个生成 token 所需 prefill |
| `ttft_model_ms` | `vision_encoder + compression + llm_prefill` |
| `decode_ms` | 首 token 之后到完整回答结束 |
| `total_model_ms` | 模型收到已预处理 tensor 到完整回答结束 |
| `total_pipeline_ms` | 图像预处理开始到完整回答结束 |

同时记录：

- `output_tokens`；
- `decode_ms_per_token`，避免不同回答长度污染 decode 比较；
- `throughput_samples_per_s`；
- `peak_gpu_allocated_mb`，使用 `torch.cuda.max_memory_allocated()`；
- `peak_gpu_reserved_mb`，使用 `torch.cuda.max_memory_reserved()`；
- `kv_cache_mb`，优先按实际 cache tensor 字节数求和；
- `selector_peak_extra_mb`：能可靠分离时记录，否则写 `NOT_MEASURED`；
- `prefill_flops`、`selector_flops`、`vision_encoder_flops`：有经验证的计数器才报告，
  否则写 `NOT_MEASURED`，不得猜测。

`compression_ms` 对 NoOp 为 `NOT_APPLICABLE`。对 Random/Spatial 也必须实测索引生成、gather
和重排成本，不能按 0 处理。

---

## 4. 测量设计

### 4.1 固定条件

- 单张 NVIDIA A40；GPU 型号、驱动、CUDA、PyTorch、Transformers、FlashAttention 版本入档。
- dtype、量化、CPU 线程数、GPU power mode 与频率策略固定并记录。
- greedy 解码、prompt、图像分辨率、最大生成长度与 v0.3 完全一致。
- 主时延使用 `batch_size=1`。吞吐量另跑 `batch_size=8`；若某方法 OOM 或代码不支持，
  如实标 `OOM` / `UNSUPPORTED`，不得为该方法单独降低 batch size 后放进同表。
- 固定 profile 样本 ID 列表并提交到 `data/subsets/`。所有方法顺序一致。

### 4.2 预热与重复

- 每个方法/档位/轨道至少预热 5 次，不记录预热结果。
- 固定 profile 集至少 100 个样本；每个样本记录原始计时，不只保留汇总值。
- 至少完成 3 个独立 pass；pass 顺序轮换，避免温度、频率和后台负载长期偏向某方法。
- 汇总报告 median、p25、p75、p95；不得只报告最小值或最好一次。
- 若三个 pass 的 TTFT 中位数极差超过总体中位数的 5%，标记 `UNSTABLE` 并排查，
  不进入主表。

### 4.3 显存测量

- 每个样本前调用 `torch.cuda.reset_peak_memory_stats()`。
- 同时报告包含模型权重的总峰值，以及相对 NoOp 的差值。
- 不以 `nvidia-smi` 瞬时读数代替 PyTorch peak 指标；`nvidia-smi` 仅作环境审计。

---

## 5. Profile run 的落盘要求

效率实验是独立 run：

```json
{
  "run_kind": "efficiency",
  "measurement_protocol_version": "e1.0",
  "linked_run_ids": ["对应的质量 run_id"],
  "metrics": {
    "efficiency": {
      "attention_track": "matched_eager | best_compatible",
      "attn_impl": "eager | flash_attention_2 | ...",
      "batch_size": 1,
      "warmup_iterations": 5,
      "profile_samples": 100,
      "passes": 3,
      "raw_timings_path": "results/artifacts/<run_id>/timings.jsonl",
      "summary": {
        "ttft_model_ms": {"median": 0, "p25": 0, "p75": 0, "p95": 0},
        "total_model_ms": {"median": 0, "p25": 0, "p75": 0, "p95": 0},
        "throughput_samples_per_s": {"median": 0, "p25": 0, "p75": 0, "p95": 0},
        "peak_gpu_allocated_mb": {"median": 0, "p25": 0, "p75": 0, "p95": 0}
      }
    }
  }
}
```

每行 raw timing 必须带 `question_id`、各阶段时间、输出 token 数、token schedule、TLB、
峰值显存和异常状态。汇总表必须由脚本从 raw timing 生成。

---

## 6. 主表和主图

每个 benchmark 至少生成：

1. `score vs actual TLB`；
2. `score vs TTFT`；
3. `score vs total_model_ms`；
4. `score vs peak_gpu_allocated_mb`；
5. 在 `quality_drop_pp <= 1.0` 和 `<= 2.0` 两个阈值下，各方法最省成本的 operating point；
6. 相对 NoOp 的 `TTFT speedup`、总时延 speedup、吞吐提升和显存节省。

```text
speedup       = latency_noop / latency_method
memory_saving = 1 - memory_method / memory_noop
```

若不存在满足掉分阈值的档位，记为 `NO_FEASIBLE_POINT`，不得用最接近的点代替。

---

## 7. 与当前实验的衔接

1. 先完成正在运行的 v0.3 准确率、TLB、keep-index 审计和 PruMerge 复现。
2. 不重跑已完成的全量准确率；profile run 关联其 `run_id`。
3. 先在 `{NoOp, Random, VisPruner, SparseVLM}` 的代表档位验证计时工具，再扩到完整矩阵。
4. 若 profile instrumentation 改变模型输出，判为实现错误；修复前不得报告效率数字。

---

## 变更记录

| 版本 | 日期 | 变更 | 原因 |
|---|---|---|---|
| e1.0 | 2026-09-21 | 建立独立效率协议；增加分阶段时延、吞吐、显存、KV cache、选择器开销、双 attention 轨道和 Pareto 报告。 | 原 v0.3 主要完成质量/TLB 复现，无法支撑端到端效率结论。 |
