"""GPU and NVML metadata collection for local inference."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class GpuMetadata:
    """Static GPU device information."""

    gpu_available: bool
    device: str
    device_index: int
    gpu_name: str | None = None
    gpu_memory_total_gb: float | None = None
    gpu_compute_capability: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_available": self.gpu_available,
            "device": self.device,
            "device_index": self.device_index,
            "gpu_name": self.gpu_name,
            "gpu_memory_total_gb": self.gpu_memory_total_gb,
            "gpu_compute_capability": self.gpu_compute_capability,
            "cuda_version": self.cuda_version,
            "driver_version": self.driver_version,
        }


@dataclass
class NvmlReading:
    """Power/energy sample from NVML."""

    available: bool = False
    device_index: int = 0
    power_draw_watts: float | None = None
    energy_joules: float | None = None
    duration_ms: float | None = None
    samples: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        avg_power = (
            sum(self.samples) / len(self.samples)
            if self.samples
            else self.power_draw_watts
        )
        return {
            "nvml_available": self.available,
            "nvml_device_index": self.device_index,
            "power_draw_watts": self.power_draw_watts,
            "avg_power_watts": avg_power,
            "energy_joules": self.energy_joules,
            "nvml_duration_ms": self.duration_ms,
            "nvml_sample_count": len(self.samples),
        }


def _parse_device_index(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    if device == "cuda":
        return 0
    return 0


def _read_nvml_driver_version(device_index: int) -> str | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            return pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def collect_gpu_metadata(*, device: str = "cuda:0") -> GpuMetadata:
    """Collect GPU metadata when PyTorch CUDA is available."""
    device_index = _parse_device_index(device)
    try:
        import torch
    except ImportError:
        return GpuMetadata(gpu_available=False, device=device, device_index=device_index)

    if not torch.cuda.is_available():
        return GpuMetadata(gpu_available=False, device=device, device_index=device_index)

    try:
        props = torch.cuda.get_device_properties(device_index)
        cuda_version = getattr(torch.version, "cuda", None)
        driver_version = _read_nvml_driver_version(device_index)

        return GpuMetadata(
            gpu_available=True,
            device=device,
            device_index=device_index,
            gpu_name=props.name,
            gpu_memory_total_gb=round(props.total_memory / (1024**3), 3),
            gpu_compute_capability=f"{props.major}.{props.minor}",
            cuda_version=str(cuda_version) if cuda_version else None,
            driver_version=driver_version,
        )
    except Exception as exc:
        logger.warning("local.gpu_metadata_failed", error=str(exc))
        return GpuMetadata(gpu_available=False, device=device, device_index=device_index)


def snapshot_gpu_memory(*, device_index: int = 0) -> dict[str, float | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"gpu_memory_allocated_gb": None, "gpu_memory_reserved_gb": None}
        return {
            "gpu_memory_allocated_gb": round(
                torch.cuda.memory_allocated(device_index) / (1024**3),
                4,
            ),
            "gpu_memory_reserved_gb": round(
                torch.cuda.memory_reserved(device_index) / (1024**3),
                4,
            ),
        }
    except ImportError:
        return {"gpu_memory_allocated_gb": None, "gpu_memory_reserved_gb": None}


class NvmlSampler:
    """Optional NVML power/energy sampling during inference."""

    def __init__(self, *, device_index: int = 0, enabled: bool = True) -> None:
        self.device_index = device_index
        self.enabled = enabled
        self._handle: Any = None
        self._pynvml: Any = None
        self._available = False
        if enabled:
            self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._available = True
        except Exception as exc:
            logger.info("local.nvml_unavailable", error=str(exc))
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def read_power_watts(self) -> float | None:
        if not self._available or self._pynvml is None or self._handle is None:
            return None
        try:
            milliwatts = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
            return milliwatts / 1000.0
        except Exception:
            return None

    def measure(self, operation: Callable[[], T]) -> tuple[T, NvmlReading]:
        if not self._available:
            started = time.perf_counter()
            result = operation()
            duration_ms = (time.perf_counter() - started) * 1000
            return result, NvmlReading(available=False, duration_ms=duration_ms)

        samples: list[float] = []
        start_power = self.read_power_watts()
        if start_power is not None:
            samples.append(start_power)

        started = time.perf_counter()
        result = operation()
        duration_s = time.perf_counter() - started

        end_power = self.read_power_watts()
        if end_power is not None:
            samples.append(end_power)

        duration_ms = duration_s * 1000
        avg_power = sum(samples) / len(samples) if samples else None
        energy = avg_power * duration_s if avg_power is not None else None

        return result, NvmlReading(
            available=True,
            device_index=self.device_index,
            power_draw_watts=end_power,
            energy_joules=energy,
            duration_ms=duration_ms,
            samples=samples,
        )

    def shutdown(self) -> None:
        if self._available and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
