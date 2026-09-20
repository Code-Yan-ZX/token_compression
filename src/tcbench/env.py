"""运行环境的路径与环境变量约定。

机器约束：
- 根分区 `/` 97% 满（31G 空闲），个人 HF 缓存默认落在其上，风险高。
- 模型真权重在 `/data/models`（root 只读，但文件可读）。
- 工作盘 `/media/disk2` 有 1.3T 空闲且可写。

约定：运行时把 datasets / hub / transformers 缓存重定向到工作盘共享缓存，
避免占满根分区。本模块提供 `configure()`，在 import 任何 transformers/datasets 前调用。
"""

from __future__ import annotations

import os
from pathlib import Path


_HF_CACHE_ROOT = Path("/media/disk2/YZX/hf_caches")
_MODEL_CACHE_ROOT = Path("/data/models")
_MODELSCOPE_HUB = Path("/data/models/modelscope/hub/models")


def msc_dir() -> str:
    """返回 modelscope 缓存根（用于 snapshot_download / 加载）"""
    return str(_MODELSCOPE_HUB)


def model_dir(model_id: str) -> str | None:
    """把 HF 风格 model_id 映射到本机模型实际路径。

    LLaVA-1.5-7B 经由 modelscope 下载，缓存名做了下划线转义：
        llava-hf/llava-1.5-7b-hf -> /data/models/modelscope/hub/models/llava-hf/llava-1___5-7b-hf
    """
    if model_id == "llava-hf/llava-1.5-7b-hf":
        return str(_MODELSCOPE_HUB / "llava-hf" / "llava-1___5-7b-hf")
    return str(_MODELSCOPE_HUB)  # 兜底回 modelscope 根


def configure() -> None:
    """在导入 transformers/datasets 之前调用，把缓存重定向到工作盘。

    注意:TextVQA 数据已存在于默认 ~/.cache/huggingface/datasets(根分区)。
    为避免重复下载/处理，datasets 保持默认不上移；仅把 hub 模型缓存移到工作盘。
    （若后续需要新数据集，会在 EXPERIMENT_LOG 记录并用 HF_DATASETS_CACHE 显式指定。）
    """
    os.environ.setdefault("HF_HUB_CACHE", str(_HF_CACHE_ROOT / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_CACHE_ROOT / "hub"))
    # 注意:不要覆盖 HF_HOME —— datasets 需要默认根来找已缓存的 TextVQA 元数据。
    # 避免 modelscope 把新下载写进 97% 满的根分区家目录
    os.environ.setdefault("MODELSCOPE_CACHE", msc_dir())
    # 不主动联网（模型/数据都在本地缓存）
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def summary() -> dict[str, str]:
    return {
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", ""),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", ""),
        "MODELSCOPE_CACHE": os.environ.get("MODELSCOPE_CACHE", ""),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
    }