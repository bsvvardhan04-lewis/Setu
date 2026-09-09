"""Measurement harness.

The claim "the NPU makes this workload viable" is only worth making if you can produce
the table that proves it. This harness runs each model on each available device, N times,
and reports p50/p90 latency plus energy per inference measured off the battery counter.

It forces placement by constructing a router pinned to one device, so the A/B is real and
not a side effect of the adaptive policy.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import Settings
from ..runtime import Device, EnergyProbe, HexaRouter, SessionCache, describe_host
from ..runtime.router import DEFAULT_SPECS, ModelSpec


@dataclass
class BenchRow:
    model: str
    device: str
    iterations: int
    latencies_ms: list[float] = field(default_factory=list)
    energy_mwh: list[float] = field(default_factory=list)
    degraded: bool = False
    error: str | None = None
    baseline_mw: float | None = None

    def _pct(self, p: float) -> float | None:
        if not self.latencies_ms:
            return None
        ordered = sorted(self.latencies_ms)
        index = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
        return ordered[index]

    def as_dict(self) -> dict[str, object]:
        mean_energy = statistics.fmean(self.energy_mwh) if self.energy_mwh else None

        # Marginal energy: what this model cost ON TOP of an idle machine.
        marginal = None
        if mean_energy is not None and self.baseline_mw and self.latencies_ms:
            seconds = statistics.fmean(self.latencies_ms) / 1000.0
            baseline_mwh = self.baseline_mw * (seconds / 3600.0)
            marginal = max(0.0, mean_energy - baseline_mwh)

        return {
            "model": self.model,
            "device": self.device,
            "iterations": self.iterations,
            "p50_ms": None if self._pct(0.5) is None else round(self._pct(0.5), 2),
            "p90_ms": None if self._pct(0.9) is None else round(self._pct(0.9), 2),
            "mean_ms": round(statistics.fmean(self.latencies_ms), 2)
            if self.latencies_ms
            else None,
            "energy_mwh_per_inference": None if mean_energy is None else round(mean_energy, 5),
            "marginal_mwh_per_inference": None if marginal is None else round(marginal, 6),
            "degraded": self.degraded,
            "error": self.error,
        }


def _pinned_router(device: Device) -> HexaRouter:
    """A router that can only answer with one device, so the A/B is honest."""
    router = HexaRouter()
    for spec in DEFAULT_SPECS:
        router.register(
            ModelSpec(
                name=spec.name,
                role=spec.role,
                allowed=(device,),
                prior_latency=spec.prior_latency,
                prior_energy=spec.prior_energy,
                resident=spec.resident,
            )
        )
    return router


def _workloads(settings: Settings) -> dict[str, Callable]:
    """One representative unit of work per model, sized like the real pipeline."""
    from PIL import Image

    from ..models import Asr, Embedder, Llm, Ocr, Tts, Vad
    from ..models.llm import GenParams

    page = Image.fromarray(
        (np.random.default_rng(7).random((1400, 1000)) * 40 + 200).astype(np.uint8)
    ).convert("RGB")
    audio = np.sin(
        np.linspace(0, 2 * np.pi * 220, 16000 * 5, dtype=np.float32)
    ).astype(np.float32)
    frame = audio[:512]
    paragraph = (
        "The applicant must submit the completed form along with proof of residence "
        "to the district office before the last date mentioned above."
    )

    return {
        "vad": lambda cache: Vad(cache).is_speech(frame),
        "asr": lambda cache: Asr(cache).transcribe(audio),
        "ocr_detect": lambda cache: Ocr(cache).read_page(page),
        "embed": lambda cache: Embedder(cache).encode([paragraph] * 8),
        "llm": lambda cache: Llm(cache).generate(
            f"<CONTEXT>{paragraph}</CONTEXT>\n<QUESTION>What is the deadline?</QUESTION>",
            GenParams(max_tokens=128),
        ),
        "tts": lambda cache: Tts(cache).synthesize(paragraph, "hi"),
    }


def _idle_baseline_mw(seconds: float = 3.0) -> float | None:
    """Whole-system draw with SETU doing nothing.

    Without this the energy column is dishonest. The battery counter reports what the
    entire laptop is drawing - screen, radios, background services - so attributing all
    of it to one inference would overstate the model's cost by an order of magnitude.
    Subtracting a quiet baseline gives the *marginal* cost of running the model, which is
    the only figure that makes an NPU-versus-CPU comparison mean anything.
    """
    from ..runtime.telemetry import _sampler

    samples: list[float] = []
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        value = _sampler.force()
        if value is not None:
            samples.append(value)
    return statistics.fmean(samples) if samples else None


def run_benchmarks(
    settings: Settings,
    *,
    iterations: int = 20,
    warmup: int = 3,
    devices: list[Device] | None = None,
    models: list[str] | None = None,
) -> dict[str, object]:
    from ..runtime import discover_providers

    available = [p.device for p in discover_providers() if p.device is not Device.STUB]
    targets = devices or available or [Device.CPU]
    workloads = _workloads(settings)
    selected = {k: v for k, v in workloads.items() if not models or k in models}

    baseline_mw = _idle_baseline_mw()

    rows: list[BenchRow] = []
    for device in targets:
        cache = SessionCache(_pinned_router(device), settings.model_root)
        for name, workload in selected.items():
            row = BenchRow(
                model=name,
                device=device.value,
                iterations=iterations,
                baseline_mw=baseline_mw,
            )
            try:
                for _ in range(warmup):
                    workload(cache)
                for _ in range(iterations):
                    with EnergyProbe() as probe:
                        start = time.perf_counter()
                        result = workload(cache)
                        row.latencies_ms.append((time.perf_counter() - start) * 1000.0)
                    if probe.reading.energy_mwh is not None:
                        row.energy_mwh.append(probe.reading.energy_mwh)
                    if getattr(result, "degraded", False):
                        row.degraded = True
            except Exception as exc:
                row.error = f"{type(exc).__name__}: {exc}"
            rows.append(row)

    return {
        "host": describe_host(),
        "python": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "iterations": iterations,
        "warmup": warmup,
        "idle_baseline_mw": None if baseline_mw is None else round(baseline_mw, 1),
        "rows": [r.as_dict() for r in rows],
    }


def to_markdown(results: dict) -> str:
    """Render the benchmark JSON as the table that goes in docs/BENCHMARKS.md."""
    host = results["host"]
    lines = [
        "# Benchmarks",
        "",
        "Generated by `python scripts/run_bench.py`. Every number below was measured on",
        "the machine named here - nothing is copied from a datasheet.",
        "",
        f"- Host: {host['os']} {host['release']} / {host['arch']}",
        f"- Windows on ARM: {host['windows_on_arm']}",
        f"- ONNX Runtime providers: {', '.join(host['onnxruntime_providers']) or 'none'}",
        f"- Iterations: {results['iterations']} (after {results['warmup']} warmup)",
        f"- Timestamp: {results['timestamp']}",
        "",
        "Energy is sampled from the Windows `BatteryStatus.DischargeRate` counter and is",
        "only populated when the machine is running on battery. `degraded` marks a row",
        "that ran on a fallback path because the model asset was absent.",
        "",
        "**How to read the energy columns.** That counter reports what the *whole laptop*",
        "is drawing - screen, radios, background services - not what one model costs. So",
        "`system mWh` is the total draw during the inference window, and `marginal mWh` is",
        f"that figure minus a measured idle baseline of "
        f"**{results.get('idle_baseline_mw') or 'n/a'} mW**. Only the marginal column is a",
        "fair basis for comparing NPU against CPU; quoting the system column as the cost of",
        "a model would overstate it by an order of magnitude.",
        "",
        "| Model | Device | p50 ms | p90 ms | system mWh | marginal mWh | Degraded | Error |",
        "| --- | --- | ---: | ---: | ---: | ---: | :-: | --- |",
    ]
    for row in results["rows"]:
        lines.append(
            "| {model} | {device} | {p50} | {p90} | {energy} | {marginal} | {deg} | {err} |".format(
                model=row["model"],
                device=row["device"],
                p50=row["p50_ms"] if row["p50_ms"] is not None else "-",
                p90=row["p90_ms"] if row["p90_ms"] is not None else "-",
                energy=row["energy_mwh_per_inference"]
                if row["energy_mwh_per_inference"] is not None
                else "-",
                marginal=row.get("marginal_mwh_per_inference")
                if row.get("marginal_mwh_per_inference") is not None
                else "-",
                deg="yes" if row["degraded"] else "",
                err=(row["error"] or "")[:60],
            )
        )

    speedups = _speedup_table(results["rows"])
    if speedups:
        lines += [
            "",
            "## NPU versus CPU",
            "",
            "| Model | CPU p50 ms | NPU p50 ms | Speed-up | Energy ratio |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        lines += speedups
    return "\n".join(lines) + "\n"


def _speedup_table(rows: list[dict]) -> list[str]:
    by_model: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["device"]] = row

    out: list[str] = []
    for model, devices in sorted(by_model.items()):
        cpu, npu = devices.get("cpu"), devices.get("npu")
        if not cpu or not npu or not cpu["p50_ms"] or not npu["p50_ms"]:
            continue
        speedup = cpu["p50_ms"] / npu["p50_ms"]
        # Compare marginal energy, not system draw - see the note above the main table.
        cpu_e = cpu.get("marginal_mwh_per_inference")
        npu_e = npu.get("marginal_mwh_per_inference")
        ratio = f"{npu_e / cpu_e:.2f}x" if cpu_e and npu_e else "-"
        out.append(
            f"| {model} | {cpu['p50_ms']} | {npu['p50_ms']} | {speedup:.2f}x | {ratio} |"
        )
    return out


def save(results: dict, json_path: Path | None = None, md_path: Path | None = None) -> None:
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(to_markdown(results), encoding="utf-8")
