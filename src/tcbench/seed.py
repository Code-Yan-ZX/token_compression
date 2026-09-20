"""随机性与环境指纹采集。

CLAUDE.md §6:随机性用 `seed_everything(seed)` 统一入口;解码固定 `do_sample=False`。
每个 run 必须记录完整环境指纹(hardware / 依赖版本)以便复现。
"""

from __future__ import annotations

import platform
import random
import subprocess  # noqa: S404
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """固定的全局随机种子入口。kernel 层仍可能有 ~0.1 分非确定，需多 seed 均报。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 解码确定性由 runner 固定 do_sample=False 保证，此处不额外设置 deterministic 以保留性能


def _try_version_import(import_name: str) -> str | None:
    try:
        mod = __import__(import_name)
        return getattr(mod, "__version__", "<no __version__>")
    except Exception:
        return None


def torch_cuda_version() -> str | None:
    return torch.version.cuda


def gpu_info() -> list[dict[str, Any]]:
    """GPU 列表:名称、显存 MB。多卡时给出全部，主实验记为 driver 下的单卡。"""
    out: list[dict[str, Any]] = []
    if not torch.cuda.is_available():
        return out
    for i in range(torch.cuda.device_count()):
        out.append(
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "mem_mb": torch.cuda.get_device_properties(i).total_memory // (1024 * 1024),
            }
        )
    return out


def nvidia_driver_version() -> str | None:
    try:
        r = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        lines = [line.strip() for line in r.stdout.strip().splitlines() if line.strip()]
        return lines[0] if lines else None
    except Exception:
        return None


@dataclass
class EnvFingerprint:
    """一次运行的环境指纹，写入 run JSON 保证可复现。"""

    python_version: str
    platform: str
    torch_version: str | None
    cuda_version: str | None
    nvidia_driver: str | None
    transformers_version: str | None
    flash_attn_version: str | None
    accelerate_version: str | None
    datasets_version: str | None
    gpu: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def capture(cls) -> "EnvFingerprint":
        return cls(
            python_version=platform.python_version(),
            platform=platform.platform(),
            torch_version=_try_version_import("torch"),
            cuda_version=torch_cuda_version(),
            nvidia_driver=nvidia_driver_version(),
            transformers_version=_try_version_import("transformers"),
            flash_attn_version=_try_version_import("flash_attn"),
            accelerate_version=_try_version_import("accelerate"),
            datasets_version=_try_version_import("datasets"),
            gpu=gpu_info(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)