from __future__ import annotations

from dataclasses import dataclass

import psutil  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class SystemMetrics:
    gpu_util_pct: float | None
    vram_used_mb: float | None
    cpu_util_pct: float | None
    ram_used_mb: float | None


def read_system_metrics() -> SystemMetrics:
    gpu_util: float | None = None
    vram_used: float | None = None
    nvml_initialized = False
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        nvml_initialized = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        vram_used = float(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / (1024 * 1024)
    except Exception:
        gpu_util = None
        vram_used = None
    finally:
        if nvml_initialized:
            pynvml.nvmlShutdown()
    return SystemMetrics(
        gpu_util_pct=gpu_util,
        vram_used_mb=vram_used,
        cpu_util_pct=float(psutil.cpu_percent(interval=None)),
        ram_used_mb=float(psutil.virtual_memory().used) / (1024 * 1024),
    )
