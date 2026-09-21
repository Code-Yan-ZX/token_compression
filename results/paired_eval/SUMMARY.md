# Paired eval summary — TextVQA + GQA (greedy v0.3, same llava-1.5-7b base, real per-question TLB)

Decode: do_sample=False, temperature=0, num_beams=1, max_new_tokens=128. Base = verified legacy container (HF SHA B234B804B... / modelscope llava-hf).
TextVQA: 5000 val, official soft score (mean over samples).  GQA: testdev_balanced 12578, official `pred==answer` balanced accuracy.

| method / tier | actual TLB | TextVQA | GQA |
|---|---|---|---|
| Noop (no compression) | 1.000 | 57.88% | 61.27% |
| VisPruner 192 | 0.333 | 56.75% | 58.98% |
| SparseVLM "192" | 0.359 | 56.45% | 58.70% |
| VisPruner 128 | 0.222 | 56.66% | 57.88% |
| SparseVLM "128" | 0.252 | 56.36% | 57.78% |
| **random 128 (3 seeds)** | 0.222 | **56.24%** | **58.08%** |
| **spatial-uniform 128** | 0.222 | 56.24% | 58.08% |
| VisPruner 64 | 0.111 | 55.53% | 55.75% |
| SparseVLM "64" | 0.143 | 53.31% | 52.97% |

Notes:
- SparseVLM actual TLB per-real runtime: "192/128/64" -> 0.359/0.252/0.143 (higher than nominal).
- random/spatial: controlled at TLB 0.222 only (seeds 0/1/2 all identical per-question answers; = count matters, which tokens don't, at this budget).
- Pruner>random by <=0.42 (TextVQA) and random>pruner by 0.20-0.30 (GQA).
- GQA scored with official formula; validity/plausibility N/A (no choices/scenes). 
