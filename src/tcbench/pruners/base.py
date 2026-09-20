"""统一 Pruner 接口（CLAUDE.md §4，Phase 1 最重要的工程决策）。

所有方法实现同一接口，runner 不允许为某方法开特例分支。

**merge_map 是硬性要求**：Phase 2 的 evidence recall 需要知道"某个原始 patch
的信息是否还以某种形式存在"。纯删除方法此处全 -1，merge 方法给出每个原 token
被并入哪个 keep 位置。实现时若难以给出 merge_map，停下来问，不要留空。
"""

from __future__ import annotations

from typing import Literal, Optional, Protocol

import torch


class PruneResult:
    """一次剪枝动作的结果。"""

    __slots__ = ("keep_index", "merge_map", "new_features")

    def __init__(
        self,
        keep_index: torch.LongTensor,
        merge_map: Optional[torch.LongTensor] = None,
        new_features: Optional[torch.Tensor] = None,
        token_schedule_note: Optional[str] = None,
    ) -> None:
        self.keep_index = keep_index          # [N_keep] 在原始 token 序列中的下标
        self.merge_map = merge_map            # [N_orig] 每个原 token 并入哪个 keep 位置；纯删除为 -1
        self.new_features = new_features      # 若方法修改了 token 表示（merge/aggregate）
        self.token_schedule_note = token_schedule_note


class Pruner(Protocol):
    """所有压缩方法必须实现的最小三钩子。stage 必须静态声明。"""

    stage: Literal["vit", "pre_llm", "in_llm"]

    def on_vit_layer(self, layer_idx: int, feats: torch.Tensor, meta: dict) -> Optional[PruneResult]: ...

    def on_vit_output(self, feats: torch.Tensor, meta: dict) -> Optional[PruneResult]: ...

    def on_llm_layer(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        attn: Optional[torch.Tensor],
        meta: dict,
    ) -> Optional[PruneResult]: ...


class NoOpPruner:
    """不压缩。用于 T1.1 验证挂 hook 后与原生 pipeline 逐 token 完全一致。"""

    stage: Literal["pre_llm"] = "pre_llm"

    def on_vit_layer(self, *args, **kwargs) -> Optional[PruneResult]:
        return None

    def on_vit_output(self, *args, **kwargs) -> Optional[PruneResult]:
        return None

    def on_llm_layer(self, *args, **kwargs) -> Optional[PruneResult]:
        return None