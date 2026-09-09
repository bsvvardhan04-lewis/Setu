"""Hexa-Router - power- and thermal-aware placement across NPU / GPU / CPU.

Most on-device demos hard-code "put everything on the NPU". That is wrong for a real
workload: the Hexagon NPU is a shared, finite resource, some ops fall back to CPU anyway,
and the right answer changes the moment the user unplugs the charger.

Hexa-Router decides placement per (model, request) from four inputs:

  1. a static ``ModelSpec`` - what the model can even run on, and its rough cost shape
  2. live ``PowerState`` - AC vs battery, charge %, thermal pressure
  3. request ``Priority`` - is a human waiting for this token, or is it a background index
  4. a learned cost model - EWMA of *measured* latency and energy per (model, device)
     on THIS machine, so the router gets better the longer the app runs

Placement is sticky: re-creating an ORT session with a new EP costs hundreds of
milliseconds (QNN graph finalisation is not free), so we apply hysteresis and only
migrate when the projected win clears a margin.
"""

from __future__ import annotations

import enum
import logging
import threading
from dataclasses import dataclass, field

from .providers import Device, ProviderInfo, discover_providers, provider_for
from .telemetry import PowerState, read_power_state

log = logging.getLogger(__name__)


class Priority(str, enum.Enum):
    """Why this inference is running, which changes what we optimise for."""

    INTERACTIVE = "interactive"  # a human is watching a cursor blink
    STREAMING = "streaming"      # continuous, must keep up with real time
    BACKGROUND = "background"    # indexing, prefetch; latency is nearly free


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one model's deployment envelope."""

    name: str
    role: str
    allowed: tuple[Device, ...]
    #: rough relative latency per device before we have measurements (1.0 = CPU baseline)
    prior_latency: dict[Device, float] = field(default_factory=dict)
    #: rough relative energy per device before we have measurements
    prior_energy: dict[Device, float] = field(default_factory=dict)
    #: models that must stay resident (LLM weights) resist migration harder
    resident: bool = False

    def prior(self, device: Device, energy: bool = False) -> float:
        table = self.prior_energy if energy else self.prior_latency
        return table.get(device, 1.0)


@dataclass
class _Cost:
    """EWMA of observed cost for one (model, device) pair."""

    latency_ms: float | None = None
    energy_mwh: float | None = None
    samples: int = 0

    def update(self, latency_ms: float, energy_mwh: float | None, alpha: float = 0.3) -> None:
        self.latency_ms = (
            latency_ms
            if self.latency_ms is None
            else (1 - alpha) * self.latency_ms + alpha * latency_ms
        )
        if energy_mwh is not None:
            self.energy_mwh = (
                energy_mwh
                if self.energy_mwh is None
                else (1 - alpha) * self.energy_mwh + alpha * energy_mwh
            )
        self.samples += 1


@dataclass(frozen=True)
class Placement:
    """The router's answer: where to run, and how to configure the engine."""

    model: str
    device: Device
    provider: ProviderInfo
    perf_mode: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "device": self.device.value,
            "ep": self.provider.ep_name,
            "perf_mode": self.perf_mode,
            "reason": self.reason,
        }


#: How much better a candidate must look before we pay the session-rebuild cost.
_MIGRATION_MARGIN = 0.20
_RESIDENT_MIGRATION_MARGIN = 0.45


class HexaRouter:
    """Thread-safe placement engine. One instance per process."""

    def __init__(self, *, energy_weight: float | None = None) -> None:
        self._lock = threading.RLock()
        self._specs: dict[str, ModelSpec] = {}
        self._costs: dict[tuple[str, Device], _Cost] = {}
        self._current: dict[str, Placement] = {}
        self._forced_energy_weight = energy_weight
        self._devices = {p.device for p in discover_providers()}

    # ---------------------------------------------------------------- registration

    def register(self, spec: ModelSpec) -> None:
        with self._lock:
            self._specs[spec.name] = spec

    def register_all(self, specs: list[ModelSpec]) -> None:
        for spec in specs:
            self.register(spec)

    # ---------------------------------------------------------------- cost model

    def observe(
        self, model: str, device: Device, latency_ms: float, energy_mwh: float | None = None
    ) -> None:
        """Feed a real measurement back in. Called after every inference."""
        with self._lock:
            self._costs.setdefault((model, device), _Cost()).update(latency_ms, energy_mwh)

    def _score(self, spec: ModelSpec, device: Device, w_energy: float) -> float:
        """Lower is better. Blends measured cost with priors, normalised to CPU."""
        measured = self._costs.get((spec.name, device))
        cpu_measured = self._costs.get((spec.name, Device.CPU))

        if measured and measured.latency_ms is not None:
            base_ms = (
                cpu_measured.latency_ms
                if cpu_measured and cpu_measured.latency_ms
                else measured.latency_ms
            )
            latency = measured.latency_ms / max(base_ms, 1e-6)
        else:
            latency = spec.prior(device)

        if (
            measured
            and measured.energy_mwh is not None
            and cpu_measured
            and cpu_measured.energy_mwh
        ):
            energy = measured.energy_mwh / max(cpu_measured.energy_mwh, 1e-9)
        else:
            energy = spec.prior(device, energy=True)

        return (1.0 - w_energy) * latency + w_energy * energy

    def _energy_weight(self, power: PowerState, priority: Priority) -> float:
        """How much we care about milliwatts versus milliseconds, in [0, 1]."""
        if self._forced_energy_weight is not None:
            return self._forced_energy_weight
        if priority is Priority.BACKGROUND:
            w = 0.85
        elif priority is Priority.STREAMING:
            w = 0.55  # must keep up, but this is the all-day workload
        else:
            w = 0.20  # a human is waiting
        if power.energy_constrained:
            w = min(1.0, w + 0.30)
        if power.on_ac:
            w = max(0.0, w - 0.15)
        return w

    def _perf_mode(self, device: Device, power: PowerState, priority: Priority) -> str:
        """QNN HTP performance mode for this placement."""
        if device is not Device.NPU:
            return "default"
        if power.thermally_constrained or (
            power.energy_constrained and priority is not Priority.INTERACTIVE
        ):
            return "power_saver"
        if priority is Priority.INTERACTIVE:
            return "burst"
        if priority is Priority.STREAMING:
            return "sustained_high_performance"
        return "balanced"

    # ---------------------------------------------------------------- placement

    def place(
        self,
        model: str,
        *,
        priority: Priority = Priority.INTERACTIVE,
        power: PowerState | None = None,
    ) -> Placement:
        with self._lock:
            spec = self._specs.get(model)
            if spec is None:
                spec = ModelSpec(name=model, role="unknown", allowed=(Device.CPU, Device.STUB))
                self._specs[model] = spec

            power = power or read_power_state()
            w_energy = self._energy_weight(power, priority)

            candidates = [d for d in spec.allowed if d in self._devices]
            if not candidates:
                candidates = [Device.STUB]

            # Thermal backstop: stop feeding a throttling NPU unless nothing else can run it.
            if power.thermally_constrained and len(candidates) > 1:
                cooled = [d for d in candidates if d is not Device.NPU]
                if cooled:
                    candidates = cooled

            scored = sorted(candidates, key=lambda d: self._score(spec, d, w_energy))
            best = scored[0]

            current = self._current.get(model)
            reason = f"w_energy={w_energy:.2f} priority={priority.value}"

            if current is not None and current.device != best:
                margin = _RESIDENT_MIGRATION_MARGIN if spec.resident else _MIGRATION_MARGIN
                cur_score = self._score(spec, current.device, w_energy)
                new_score = self._score(spec, best, w_energy)
                if new_score > cur_score * (1.0 - margin):
                    best = current.device  # not worth the session rebuild
                    reason += " hysteresis=held"
                else:
                    reason += f" migrated={current.device.value}->{best.value}"

            provider = provider_for(best) or ProviderInfo(Device.STUB, "stub")
            placement = Placement(
                model=model,
                device=best,
                provider=provider,
                perf_mode=self._perf_mode(best, power, priority),
                reason=reason,
            )
            self._current[model] = placement
            return placement

    # ---------------------------------------------------------------- introspection

    def snapshot(self) -> dict[str, object]:
        """Everything the "why is it fast" panel in the UI needs."""
        with self._lock:
            return {
                "devices": sorted(d.value for d in self._devices),
                "placements": {m: p.as_dict() for m, p in self._current.items()},
                "cost_model": {
                    f"{m}@{d.value}": {
                        "latency_ms": None if c.latency_ms is None else round(c.latency_ms, 2),
                        "energy_mwh": None if c.energy_mwh is None else round(c.energy_mwh, 5),
                        "samples": c.samples,
                    }
                    for (m, d), c in self._costs.items()
                },
            }


#: Deployment envelopes for SETU's nine models. Priors are conservative starting
#: guesses only; the EWMA replaces them within a handful of real inferences.
DEFAULT_SPECS: list[ModelSpec] = [
    ModelSpec(
        name="vad",
        role="voice-activity",
        allowed=(Device.NPU, Device.CPU, Device.STUB),
        prior_latency={Device.NPU: 0.6, Device.CPU: 1.0},
        prior_energy={Device.NPU: 0.15, Device.CPU: 1.0},
    ),
    ModelSpec(
        name="asr",
        role="speech-recognition",
        allowed=(Device.NPU, Device.GPU, Device.CPU, Device.STUB),
        prior_latency={Device.NPU: 0.25, Device.GPU: 0.45, Device.CPU: 1.0},
        prior_energy={Device.NPU: 0.12, Device.GPU: 0.70, Device.CPU: 1.0},
    ),
    ModelSpec(
        name="lid",
        role="language-id",
        allowed=(Device.CPU, Device.STUB),
        prior_latency={Device.CPU: 1.0},
    ),
    ModelSpec(
        name="ocr_detect",
        role="text-detection",
        allowed=(Device.NPU, Device.GPU, Device.CPU, Device.STUB),
        prior_latency={Device.NPU: 0.20, Device.GPU: 0.40, Device.CPU: 1.0},
        prior_energy={Device.NPU: 0.10, Device.GPU: 0.65, Device.CPU: 1.0},
    ),
    ModelSpec(
        name="ocr_recognize",
        role="text-recognition",
        allowed=(Device.NPU, Device.GPU, Device.CPU, Device.STUB),
        prior_latency={Device.NPU: 0.22, Device.GPU: 0.45, Device.CPU: 1.0},
        prior_energy={Device.NPU: 0.11, Device.GPU: 0.70, Device.CPU: 1.0},
    ),
    ModelSpec(
        name="embed",
        role="embeddings",
        allowed=(Device.NPU, Device.CPU, Device.STUB),
        prior_latency={Device.NPU: 0.30, Device.CPU: 1.0},
        prior_energy={Device.NPU: 0.15, Device.CPU: 1.0},
    ),
    ModelSpec(
        name="translate",
        role="translation",
        allowed=(Device.NPU, Device.CPU, Device.STUB),
        prior_latency={Device.NPU: 0.30, Device.CPU: 1.0},
        prior_energy={Device.NPU: 0.18, Device.CPU: 1.0},
    ),
    ModelSpec(
        name="llm",
        role="reasoning",
        allowed=(Device.NPU, Device.CPU, Device.STUB),
        prior_latency={Device.NPU: 0.28, Device.CPU: 1.0},
        prior_energy={Device.NPU: 0.14, Device.CPU: 1.0},
        resident=True,
    ),
    ModelSpec(
        name="tts",
        role="speech-synthesis",
        allowed=(Device.NPU, Device.CPU, Device.STUB),
        prior_latency={Device.NPU: 0.40, Device.CPU: 1.0},
        prior_energy={Device.NPU: 0.25, Device.CPU: 1.0},
    ),
]


def build_default_router() -> HexaRouter:
    router = HexaRouter()
    router.register_all(DEFAULT_SPECS)
    return router
