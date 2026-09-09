"""Session cache - the one place that actually talks to onnxruntime.

Two jobs:

* Build an ``InferenceSession`` for a (model, device) pair and keep it. QNN graph
  finalisation is expensive enough that rebuilding per request would dominate the
  latency we are trying to win, so sessions are cached and reference-counted by key.
* Wrap every ``run`` in timing + energy measurement, and feed the result back into
  Hexa-Router's cost model. Routing quality is a function of measurement quality.

If onnxruntime or the model file is missing, ``get`` returns ``None`` and the calling
adapter falls back to its stub path. That is what keeps the whole app runnable on a
machine with no models downloaded.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .providers import Device
from .router import HexaRouter, Placement, Priority
from .telemetry import EnergyProbe

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    outputs: Any
    latency_ms: float
    device: Device
    energy_mwh: float | None = None


class _Session:
    """Thin wrapper so callers do not import onnxruntime directly."""

    def __init__(self, ort_session: Any, placement: Placement) -> None:
        self.ort = ort_session
        self.placement = placement
        self.input_names = [i.name for i in ort_session.get_inputs()]
        self.output_names = [o.name for o in ort_session.get_outputs()]


class SessionCache:
    """Creates, caches, and instruments ONNX sessions."""

    def __init__(self, router: HexaRouter, model_root: Path | str = "models") -> None:
        self.router = router
        self.model_root = Path(model_root)
        self._lock = threading.RLock()
        self._sessions: dict[tuple[str, str, Device], _Session] = {}
        self._missing: set[tuple[str, str]] = set()

    # ---------------------------------------------------------------- building

    def _resolve_path(self, model: str, filename: str | None) -> Path | None:
        candidate = self.model_root / model / (filename or f"{model}.onnx")
        return candidate if candidate.is_file() else None

    def _build(self, model: str, path: Path, placement: Placement) -> _Session | None:
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - install-dependent
            log.debug("onnxruntime import failed for %s: %s", model, exc)
            return None

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3

        provider = placement.provider
        provider_entry = provider.as_ort_provider()
        if placement.device is Device.NPU and isinstance(provider_entry, tuple):
            # Let the router's chosen HTP performance mode win over the static default.
            name, options = provider_entry
            options = dict(options)
            options["htp_performance_mode"] = placement.perf_mode
            provider_entry = (name, options)

            # Cache the finalised QNN context so subsequent launches skip compilation.
            ctx_dir = self.model_root / model / "qnn_ctx"
            ctx_dir.mkdir(parents=True, exist_ok=True)
            opts.add_session_config_entry("ep.context_enable", "1")
            opts.add_session_config_entry(
                "ep.context_file_path", str(ctx_dir / f"{model}_ctx.onnx")
            )

        try:
            sess = ort.InferenceSession(str(path), opts, providers=[provider_entry])
        except Exception as exc:
            log.warning("session build failed %s@%s: %s", model, placement.device.value, exc)
            return None
        return _Session(sess, placement)

    def get(
        self,
        model: str,
        *,
        filename: str | None = None,
        priority: Priority = Priority.INTERACTIVE,
    ) -> _Session | None:
        """Return a live session for ``model``, or ``None`` to signal stub fallback."""
        placement = self.router.place(model, priority=priority)
        if placement.device is Device.STUB:
            return None

        # The filename is part of the identity, not a detail. Several models ship as two
        # graphs under one key - Whisper is an encoder plus a decoder, so is the
        # translator - and keying on the model alone silently hands back whichever half
        # was built first. That surfaces far downstream as a missing-input error naming a
        # tensor the caller never mentioned.
        asset = filename or f"{model}.onnx"
        key = (model, asset, placement.device)
        with self._lock:
            cached = self._sessions.get(key)
            if cached is not None:
                cached.placement = placement
                return cached

            if (model, asset) in self._missing:
                return None

            path = self._resolve_path(model, filename)
            if path is None:
                self._missing.add((model, asset))
                log.info("no ONNX asset %s/%s under %s - using stub", model, asset, self.model_root)
                return None

            built = self._build(model, path, placement)
            if built is None:
                # The chosen device refused the graph; fall back one rung and retry once.
                if placement.device is not Device.CPU:
                    from .providers import provider_for

                    cpu = provider_for(Device.CPU)
                    if cpu is not None:
                        fallback = Placement(
                            model=model,
                            device=Device.CPU,
                            provider=cpu,
                            perf_mode="default",
                            reason="fallback: accelerator rejected graph",
                        )
                        built = self._build(model, path, fallback)
                        if built is not None:
                            self._sessions[(model, asset, Device.CPU)] = built
                            return built
                self._missing.add((model, asset))
                return None

            self._sessions[key] = built
            return built

    # ---------------------------------------------------------------- running

    def run(
        self,
        model: str,
        feeds: dict[str, Any],
        *,
        filename: str | None = None,
        priority: Priority = Priority.INTERACTIVE,
        measure_energy: bool = False,
    ) -> RunResult | None:
        """Run one inference, time it, and report the cost back to the router."""
        session = self.get(model, filename=filename, priority=priority)
        if session is None:
            return None

        probe = EnergyProbe() if measure_energy else None
        start = time.perf_counter()
        if probe is not None:
            probe.__enter__()
        try:
            outputs = session.ort.run(session.output_names, feeds)
        finally:
            if probe is not None:
                probe.__exit__(None, None, None)
        latency_ms = (time.perf_counter() - start) * 1000.0

        energy = probe.reading.energy_mwh if probe is not None else None
        device = session.placement.device
        self.router.observe(model, device, latency_ms, energy)
        return RunResult(
            outputs=outputs, latency_ms=latency_ms, device=device, energy_mwh=energy
        )

    # ---------------------------------------------------------------- introspection

    def loaded(self) -> dict[str, str]:
        with self._lock:
            return {f"{m}/{asset}": d.value for (m, asset, d) in self._sessions}

    def missing(self) -> list[str]:
        with self._lock:
            return sorted(f"{m}/{asset}" for m, asset in self._missing)
