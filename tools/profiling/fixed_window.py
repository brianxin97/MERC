"""Reusable measurement windows for the paper's H100 profiling protocol.

The historical runs changed the relevant YAML batch-size fields to eight before
constructing each data loader.  ``set_profile_batch_size`` provides the same
operation explicitly for reproductions from an unmodified upstream checkout.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any

import torch


PROFILE_BATCH_SIZE = 8


def _set_field(container: Any, key: str, value: int) -> None:
    """Set a field on a mapping or an attribute-style configuration object."""

    if isinstance(container, MutableMapping):
        container[key] = value
    else:
        setattr(container, key, value)


def _has_field(container: Any, key: str) -> bool:
    if isinstance(container, MutableMapping):
        return key in container
    return hasattr(container, key)


def set_profile_batch_size(cfg: Any, batch_size: int = PROFILE_BATCH_SIZE) -> Any:
    """Override loader batch-size fields before any data loader is constructed.

    ``cfg`` may be the complete configuration (with a ``train`` section) or the
    train section itself.  Flock may define ``test_batch_size`` separately, so
    that field is overwritten when present.  The function mutates and returns
    ``cfg`` to match the configuration objects used by all three codebases.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(cfg, MutableMapping) and "train" in cfg:
        train_cfg = cfg["train"]
    else:
        train_cfg = getattr(cfg, "train", cfg)
    _set_field(train_cfg, "batch_size", batch_size)
    if _has_field(train_cfg, "test_batch_size"):
        _set_field(train_cfg, "test_batch_size", batch_size)
    return cfg


def _cuda_device(device: torch.device | str | int) -> torch.device:
    resolved = torch.device(f"cuda:{device}") if isinstance(device, int) else torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The reported efficiency protocol requires a CUDA device")
    return resolved


@dataclass
class InferenceWindow:
    """Measure 50 batches after five warm-up batches with CUDA events.

    A link-prediction batch evaluates both tail and head candidates. Dividing by
    `forward_passes_per_batch=2` reports time per forward pass, as in the paper.
    """

    warmup_batches: int = 5
    measured_batches: int = 50
    forward_passes_per_batch: int = 2
    device: torch.device | str | int = "cuda"
    timings: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.warmup_batches < 0 or self.measured_batches <= 0:
            raise ValueError("invalid warm-up or measurement-window length")
        if self.forward_passes_per_batch <= 0:
            raise ValueError("forward_passes_per_batch must be positive")
        self.device = _cuda_device(self.device)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

    def begin_batch(self) -> None:
        torch.cuda.synchronize(self.device)
        self.start_event.record(torch.cuda.current_stream(self.device))

    def end_batch(self, batch_index: int) -> dict[str, float] | None:
        """Close a batch, where ``batch_index`` is zero-based."""

        if len(self.timings) >= self.measured_batches:
            raise RuntimeError("the inference measurement window is already complete")
        self.end_event.record(torch.cuda.current_stream(self.device))
        torch.cuda.synchronize(self.device)
        if batch_index >= self.warmup_batches:
            seconds = self.start_event.elapsed_time(self.end_event) / 1000.0
            self.timings.append(seconds / self.forward_passes_per_batch)
        if len(self.timings) < self.measured_batches:
            return None
        return {
            "mean_seconds_per_forward": statistics.fmean(self.timings),
            "std_seconds_per_forward": statistics.pstdev(self.timings),
            "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1024**2,
            "warmup_batches": float(self.warmup_batches),
            "measured_batches": float(self.measured_batches),
        }


@dataclass
class TrainingWindow:
    """Measure wall-clock time for 50 training batches after five warm-ups."""

    warmup_batches: int = 5
    measured_batches: int = 50
    device: torch.device | str | int = "cuda"
    start_time: float | None = None

    def __post_init__(self) -> None:
        if self.warmup_batches < 0 or self.measured_batches <= 0:
            raise ValueError("invalid warm-up or measurement-window length")
        self.device = _cuda_device(self.device)

    def end_batch(self, completed_batches: int) -> dict[str, float] | None:
        """Observe a completed optimization batch using a one-based count."""

        if completed_batches == self.warmup_batches:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_time = time.perf_counter()
            return None
        target = self.warmup_batches + self.measured_batches
        if completed_batches != target or self.start_time is None:
            return None
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self.start_time
        return {
            "mean_seconds_per_batch": elapsed / self.measured_batches,
            "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1024**2,
            "warmup_batches": float(self.warmup_batches),
            "measured_batches": float(self.measured_batches),
        }
