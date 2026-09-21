"""LLaVA-1.5 model wrapper：统一 hook 点。

CLAUDE.md §3 / PROTOCOL §1：封装 LLaVA-1.5-7B，暴露三个 hook——
ViT 逐层 / ViT 输出 / LLM 逐层。`meta` 携带 patch_grid / image_token_span /
question_token_span / text_token_ids。

T1.1 目标：挂上 NoOpPruner 后输出与原生 pipeline 逐 token 完全一致。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from tcbench.pruners.base import NoOpPruner, PruneResult, Pruner

# 模型缓存根：本机真权重在 /data/models（modelscope / HF 两个 hub）
MODEL_CACHE_ROOT = os.environ.get("TCBENCH_MODEL_CACHE", "/data/models")


@dataclass
class HookMeta:
    """传给 pruner 的 meta，含 evidence recall 需要的一切。"""

    patch_grid: tuple[int, int]
    image_token_span: tuple[int, int]  # (start, end) 在完整 token 序列中的下标
    question_token_span: tuple[int, int]
    text_token_ids: list[int] = field(default_factory=list)
    image_index: int = 0


@dataclass
class LayerState:
    """每个 LLM 层输入端观察到的状态，供逐层 hook 使用。"""

    n_visual: int  # 该层输入端的 visual token 数量
    prune_result: Optional[PruneResult] = None


class LLaVAWrapper:
    """统一封装 LLaVA-1.5-7B，插入三个 hook 并逐层记录 n_ℓ。

    - 精度用 eager attention（PROTOCOL §1.1）
    - meta 每次图像前 prevision 计算并传递
    - token_schedule[len=32] 记录 LLM 各层输入端 visual token 数
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        revision: str = "",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        pruner: Optional[Pruner] = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.dtype = dtype
        self.device = device
        self.pruner = pruner or NoOpPruner()
        self._model = None
        self._tokenizer = None
        self.image_processor = None
        self.token_schedule: list[int] = []
        self._layer_states: list[LayerState] = []

    # ------------------------------------------------------------------ #
    # 加载
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """按 TCBENCH_MODEL_CACHE 从本地缓存加载模型与 tokenizer。"""
        from transformers import LlavaForConditionalGeneration, LlavaProcessor

        cache_dir = str(MODEL_CACHE_ROOT)
        kwargs = {
            "cache_dir": cache_dir,
            "torch_dtype": self.dtype,
            "device_map": self.device,
            "attn_implementation": "eager",  # PROTOCOL §1.1 精度统一 eager
            "revision": self.revision or None,
            "local_files_only": True,  # 只从本地缓存加载，不联网
        }
        self._model = LlavaForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
        self._tokenizer = LlavaProcessor.from_pretrained(
            self.model_id, cache_dir=cache_dir, local_files_only=True
        )
        self._model.eval()
        self._install_hooks()

    def _install_hooks(self) -> None:
        """在 LLM 每层前插入 hook，作为视觉 token 计数与 pruner 调用点。"""
        lm = getattr(self._model, "language_model", None)
        if lm is None:
            raise RuntimeError("model 无 language_model，非 LLaVA 结构")
        # transformers LLaVA：lm = LlamaForCausalLM，层在 lm.model.layers
        base_lm = getattr(lm, "model", None) or lm
        self._llm_layers = getattr(base_lm, "layers", None)
        if self._llm_layers is None:
            raise RuntimeError("无法定位 LLM 层列表")
        self._layers = self._llm_layers
        self._install_layer_forward_patches()

    def _install_layer_forward_patches(self) -> None:
        """包一层每个 transformer 层的 forward：在其返回前记录输入端 n_visual 并调 pruner。"""

        for idx, layer in enumerate(self._layers):
            orig = layer.forward

            def make_closure(cur_idx: int, orig_fwd: Callable):
                def patched_forward(*args, **kwargs):
                    hidden = kwargs.get("hidden_states")
                    if hidden is None and args:
                        hidden = args[0]
                    meta = getattr(self, "_current_meta", None)
                    if meta is not None and self.pruner is not None:
                        result = self.pruner.on_llm_layer(cur_idx, hidden, None, meta)
                        if result is not None:
                            self._last_prune_result = result
                        # --- 逐层 token_schedule 记录：prefill 状态机 ---
                        # 记录规则：一个 prefill pass 内，layer 0→31 各来一次，
                        # 且这些调用共享同一个 seq_len（= prompt 长度）。
                        # decode 阶段每生成一个 token 也跑 0→31，但 seq_len 递增，据此区分。
                        if hidden is not None and hidden.dim() == 3:
                            seq_len = hidden.shape[1]
                            # 用"当前 pass 起始 seq_len + 层号"判定：仅当 layer 0 换了一个新 seq_len 时开新 pass
                            if cur_idx == 0:
                                self._pass_seqlen = seq_len
                            # 属于本次 prefill pass 的层？层号==记录到的进度且 pass_seqlen==seq_len 才记
                            if getattr(self, "_pass_seqlen", None) == seq_len and cur_idx == len(self.token_schedule):
                                # n_ℓ = 该层输入端 visual token 数。
                                # transformers 4.57 在 processor 阶段把 image 展开成 N0=576 个
                                # image_token(id=32000)；剪枝方法运行时该值这里更新为保留数。
                                n_visual = getattr(self, "_n_visual", 0)
                                self.token_schedule.append(n_visual)
                    return orig_fwd(*args, **kwargs)

                return patched_forward

            layer.forward = make_closure(idx, orig)

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, images, question: str, bench_suffix: str = "", max_new_tokens: int = 128,
                 do_sample: bool | None = None, temperature: float | None = None, num_beams: int | None = None):
        """端到端生成，同时逐层记录 token_schedule 并运行 pruner hook。

        PROTOCOL v0.3: 默认贪心 do_sample=False, temperature=0, num_beams=1。
        保留采样参数入口仅用于对比、不作主实验默认。
        """
        if self._model is None:
            self.load()
        self.token_schedule = []
        self._current_meta = None
        self._generation_in_progress = False

        processor = self._tokenizer
        # 官方 LLaVA vicuna_v1 会话模板（对齐 SparseVLMs/llava/conversation.py conv_llava_v1）。
        # 完整 prompt：
        #   <system> USER: <image>\n<q_body> ASSISTANT:
        # 与 transformers 默认模板不同，必须手工复刻才能对齐官方数字。
        system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        user_text = question
        if bench_suffix:
            user_text = f"{question}\n{bench_suffix}"
        text = f"{system} USER: <image>\n{user_text} ASSISTANT:"
        inputs = processor(images=images, text=text, return_tensors="pt").to(self.device, self.dtype)
        ids = inputs["input_ids"][0].tolist()
        img_tok_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.image_token)
        # n_visual：input_ids 里 image_token 的个数(=N0=576)。剪枝运行时会更新为保留数并同步缩 input_ids。
        self._n_visual = sum(1 for v in ids if v == img_tok_id)
        self._image_ids = ids
        self._prompt_len = len(ids)
        # 记录 meta（span 从 input_ids 里连续 image_token 段精确计算）
        image_span = self._locate_image_tokens(inputs, processor)
        meta = HookMeta(
            patch_grid=(24, 24),
            image_token_span=image_span,
            question_token_span=(0, 0),  # question 位于 image 之后部分，精确 span 后续完善
            text_token_ids=ids,
        )
        self._current_meta = meta

        gen_kwargs = {
            "do_sample": False if do_sample is None else do_sample,
            "temperature": 0.0 if temperature is None else temperature,
            "num_beams": 1 if num_beams is None else num_beams,
            "max_new_tokens": max_new_tokens,
        }
        # image_inputs 需单独传给 generate；input_ids/pixel_values 都由处理器给出
        self._generation_in_progress = True
        out = self._model.generate(**inputs, **gen_kwargs)
        self._generation_in_progress = False
        self._current_meta = None

        generated_ids = out[0]
        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[input_len:]
        decoded = processor.decode(new_ids, skip_special_tokens=True)
        return decoded

    def _locate_image_tokens(self, inputs, processor) -> tuple[int, int]:
        """定位 input_ids 里连续 image_token(<image>=id) 段的首末下标。

        transformers 4.57 在 `processor(images, text)` 阶段已把 image 展开为
        N0=576 个 image_token(id=32000) 直接放进 input_ids，因此这里统计它们的连续区间。
        """
        img_tok_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.image_token)
        ids = inputs["input_ids"][0].tolist()
        positions = [i for i, v in enumerate(ids) if v == img_tok_id]
        if not positions:
            return (0, 0)
        # 连续 block：取 [min, max+1]；若多个 image(多图)则用首段（T1.1 单图）
        return (positions[0], positions[-1] + 1)

    def token_schedule_len(self) -> int:
        return len(self.token_schedule)