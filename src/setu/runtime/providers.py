"""Execution-provider discovery.

SETU never assumes a Snapdragon device is present. It asks onnxruntime what it has,
maps the answer onto our four logical devices, and lets the router decide from there.
That is what makes the same wheel run on a reviewer's x64 ThinkPad and on an
HP OmniBook with a Hexagon NPU.
"""

from __future__ import annotations

import enum
import logging
import platform
from dataclasses import dataclass, field
from functools import lru_cache

log = logging.getLogger(__name__)


class Device(str, enum.Enum):
    """Logical compute engines SETU can target."""

    NPU = "npu"      # Hexagon NPU via QNN EP (HTP backend)
    GPU = "gpu"      # Adreno via DirectML, or CUDA/CoreML elsewhere
    CPU = "cpu"      # ARM64 oneDNN / x64
    STUB = "stub"    # deterministic no-hardware backend, used by tests & demos

    @property
    def is_accelerator(self) -> bool:
        return self in (Device.NPU, Device.GPU)


#: onnxruntime EP name -> logical device, in descending order of preference.
_EP_MAP: list[tuple[str, Device]] = [
    ("QNNExecutionProvider", Device.NPU),
    ("DmlExecutionProvider", Device.GPU),
    ("CUDAExecutionProvider", Device.GPU),
    ("CoreMLExecutionProvider", Device.GPU),
    ("CPUExecutionProvider", Device.CPU),
]


@dataclass(frozen=True)
class ProviderInfo:
    """One usable execution provider on this machine."""

    device: Device
    ep_name: str
    options: dict[str, str] = field(default_factory=dict)

    def as_ort_provider(self):
        """Shape onnxruntime expects: bare name, or (name, options) tuple."""
        return (self.ep_name, dict(self.options)) if self.options else self.ep_name


def _qnn_options(perf_mode: str = "burst") -> dict[str, str]:
    """QNN EP options for the Hexagon HTP backend.

    ``htp_performance_mode`` is the single most impactful knob: ``burst`` for
    interactive turns, ``sustained_high_performance`` for long background jobs,
    ``power_saver`` when the router decides the battery matters more than the
    millisecond. Hexa-Router rewrites this per placement.
    """
    return {
        "backend_path": "QnnHtp.dll",
        "htp_performance_mode": perf_mode,
        "htp_graph_finalization_optimization_mode": "3",
        "enable_htp_fp16_precision": "1",
    }


@lru_cache(maxsize=1)
def _available_ep_names() -> tuple[str, ...]:
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - depends on install extras
        log.warning("onnxruntime unavailable (%s); running in stub mode", exc)
        return ()
    return tuple(ort.get_available_providers())


@lru_cache(maxsize=1)
def discover_providers() -> tuple[ProviderInfo, ...]:
    """Return every usable provider, best-first.

    Always ends with a STUB entry so no call site ever has to handle "nothing available".
    """
    found: list[ProviderInfo] = []
    seen: set[Device] = set()
    available = _available_ep_names()

    for ep_name, device in _EP_MAP:
        if ep_name not in available or device in seen:
            continue
        opts = _qnn_options() if device is Device.NPU else {}
        found.append(ProviderInfo(device=device, ep_name=ep_name, options=opts))
        seen.add(device)

    found.append(ProviderInfo(device=Device.STUB, ep_name="stub"))
    log.info("providers: %s", ", ".join(p.ep_name for p in found))
    return tuple(found)


def provider_for(device: Device) -> ProviderInfo | None:
    for p in discover_providers():
        if p.device is device:
            return p
    return None


def is_snapdragon_windows() -> bool:
    """True on Windows-on-ARM, which is the shipping target for this project."""
    return platform.system() == "Windows" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }


def describe_host() -> dict[str, object]:
    return {
        "os": platform.system(),
        "release": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "windows_on_arm": is_snapdragon_windows(),
        "onnxruntime_providers": list(_available_ep_names()),
        "devices": [p.device.value for p in discover_providers()],
    }
