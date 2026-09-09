"""Compile and profile SETU's models on REAL Snapdragon hardware, via Qualcomm AI Hub.

This is the script that turns "designed for Snapdragon" into evidence.

AI Hub provisions physical Snapdragon devices in the cloud. You submit a model, it is
compiled for the target and executed on real silicon, and you get back measured inference
time, peak memory, and - the number that actually matters for this project - the
proportion of layers that landed on the NPU rather than falling back to CPU.

Every job gets a shareable URL on Qualcomm's own site. Those URLs go in the submission,
so a judge can click through and verify each number independently instead of taking our
word for it.

Usage
-----
    pip install qai-hub
    qai-hub configure --api_token <token from aihub.qualcomm.com>
    python scripts/aihub_profile.py --list-devices
    python scripts/aihub_profile.py --all
    python scripts/aihub_profile.py --model embed --device "Snapdragon X Elite CRD"

Nothing here runs at import time and nothing is required for the app itself; this is a
build-time tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_JSON = REPO / "bench_results" / "aihub_profile.json"
OUT_MD = REPO / "docs" / "AIHUB_PROFILE.md"

#: Preference order. AI Hub's catalogue changes, so we resolve against the live list
#: rather than hard-coding one name and failing when it is renamed.
PREFERRED_DEVICES = [
    "Snapdragon X2 Elite CRD",
    "Snapdragon X Elite CRD",
    "Snapdragon X Plus 8-Core CRD",
]

#: The ONNX graphs SETU ships, with the input shapes their QNN graphs are frozen at.
#: Static shapes are not incidental - a dynamic axis is the most common reason a layer
#: silently falls back off the NPU, which is exactly what this script exists to detect.
TARGETS: dict[str, dict] = {
    "embed": {
        "file": "embed/embed.onnx",
        "inputs": {
            "input_ids": ((1, 256), "int64"),
            "attention_mask": ((1, 256), "int64"),
            "token_type_ids": ((1, 256), "int64"),
        },
        "note": "MiniLM sentence embeddings for the local retrieval store.",
    },
    "vad": {
        "file": "vad/vad.onnx",
        "inputs": {"input": ((1, 512), "float32"), "state": ((2, 1, 128), "float32")},
        "note": "Silero VAD. Gates the ASR model; runs on every 32 ms frame all day.",
    },
    "asr_encoder": {
        "file": "asr/asr_encoder.onnx",
        "inputs": {"mel": ((1, 80, 3000), "float32")},
        "note": "Whisper encoder, 30 s window. The dominant cost in the live pipeline.",
    },
    "asr_decoder": {
        "file": "asr/asr_decoder.onnx",
        "inputs": {"encoder_hidden_states": ((1, 1500, 768), "float32")},
        "note": "Whisper decoder.",
    },
    "ocr_detect": {
        "file": "ocr_detect/ocr_detect.onnx",
        "inputs": {"x": ((1, 1, 960, 960), "float32")},
        "note": "DB text detector for the document-capture mode.",
    },
    "ocr_recognize": {
        "file": "ocr_recognize/ocr_recognize.onnx",
        "inputs": {"x": ((32, 1, 32, 320), "float32")},
        "note": "CRNN recogniser, batch of 32 line crops per NPU submission.",
    },
    "translate_encoder": {
        "file": "translate/translate_encoder.onnx",
        "inputs": {
            "input_ids": ((1, 512), "int64"),
            "attention_mask": ((1, 512), "int64"),
        },
        "note": "IndicTrans2 distilled encoder.",
    },
}


@dataclass
class ProfileRow:
    model: str
    device: str
    inference_ms: float | None = None
    peak_memory_mb: float | None = None
    layers_total: int = 0
    layers_npu: int = 0
    layers_gpu: int = 0
    layers_cpu: int = 0
    compile_url: str | None = None
    profile_url: str | None = None
    error: str | None = None
    note: str = ""

    @property
    def npu_fraction(self) -> float | None:
        return self.layers_npu / self.layers_total if self.layers_total else None

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "device": self.device,
            "inference_ms": self.inference_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "layers": {
                "total": self.layers_total,
                "npu": self.layers_npu,
                "gpu": self.layers_gpu,
                "cpu": self.layers_cpu,
            },
            "npu_fraction": self.npu_fraction,
            "compile_url": self.compile_url,
            "profile_url": self.profile_url,
            "error": self.error,
            "note": self.note,
        }


def _require_hub():
    try:
        import qai_hub as hub
    except ImportError:
        sys.exit(
            "qai-hub is not installed.\n"
            "  pip install qai-hub\n"
            "  qai-hub configure --api_token <token from https://aihub.qualcomm.com>"
        )
    return hub


def list_devices() -> None:
    hub = _require_hub()
    try:
        devices = hub.get_devices()
    except Exception as exc:
        sys.exit(f"Could not reach AI Hub: {exc}\nHave you run 'qai-hub configure'?")
    for device in devices:
        attrs = " ".join(sorted(getattr(device, "attributes", []) or []))
        print(f"{device.name:<42} {getattr(device, 'os', ''):<10} {attrs}")


def pick_device(hub, requested: str | None):
    """Resolve a usable device, preferring a Snapdragon X-series Windows part."""
    if requested:
        return hub.Device(name=requested)
    try:
        available = {d.name for d in hub.get_devices()}
    except Exception:
        available = set()
    for name in PREFERRED_DEVICES:
        if name in available:
            return hub.Device(name=name)
    if available:
        # Fall back to anything Windows-on-ARM before giving up entirely.
        for name in sorted(available):
            if "X Elite" in name or "X Plus" in name or "X2" in name:
                return hub.Device(name=name)
    return hub.Device(name=PREFERRED_DEVICES[0])


def _count_compute_units(profile: dict) -> tuple[int, int, int, int]:
    """Count layers per compute unit from an AI Hub profile payload.

    The exact shape of the profile dict has changed across AI Hub releases, so this
    walks defensively and tolerates missing keys rather than assuming a schema.
    """
    detail = profile.get("execution_detail") or profile.get("execution_details") or []
    if isinstance(detail, dict):
        detail = list(detail.values())

    npu = gpu = cpu = 0
    for layer in detail:
        if not isinstance(layer, dict):
            continue
        unit = str(
            layer.get("compute_unit") or layer.get("computeUnit") or ""
        ).upper()
        if unit.startswith("NPU"):
            npu += 1
        elif unit.startswith("GPU"):
            gpu += 1
        elif unit.startswith("CPU"):
            cpu += 1

    total = npu + gpu + cpu
    if total == 0:
        # Some releases only report a rollup rather than per-layer detail.
        summary = profile.get("execution_summary") or {}
        for key, target in (
            ("compute_unit_npu_layers", "npu"),
            ("compute_unit_gpu_layers", "gpu"),
            ("compute_unit_cpu_layers", "cpu"),
        ):
            value = summary.get(key)
            if isinstance(value, int):
                if target == "npu":
                    npu = value
                elif target == "gpu":
                    gpu = value
                else:
                    cpu = value
        total = npu + gpu + cpu
    return total, npu, gpu, cpu


def _summary_numbers(profile: dict) -> tuple[float | None, float | None]:
    summary = profile.get("execution_summary") or {}
    micros = (
        summary.get("estimated_inference_time")
        or summary.get("inference_time")
        or summary.get("estimated_inference_time_microseconds")
    )
    peak = (
        summary.get("estimated_inference_peak_memory")
        or summary.get("inference_peak_memory")
    )
    inference_ms = float(micros) / 1000.0 if isinstance(micros, (int, float)) else None
    peak_mb = float(peak) / (1024 * 1024) if isinstance(peak, (int, float)) else None
    return inference_ms, peak_mb


def profile_one(hub, key: str, spec: dict, device, model_root: Path) -> ProfileRow:
    row = ProfileRow(model=key, device=device.name, note=spec.get("note", ""))
    path = model_root / spec["file"]
    if not path.is_file():
        row.error = f"asset missing: {path.relative_to(REPO) if REPO in path.parents else path}"
        return row

    input_specs = {name: (shape, dtype) for name, (shape, dtype) in spec["inputs"].items()}

    try:
        compile_job = hub.submit_compile_job(
            model=str(path),
            device=device,
            input_specs=input_specs,
            # Target the QNN context binary so the graph is compiled for the Hexagon NPU
            # rather than left to a generic runtime.
            options="--target_runtime qnn_context_binary",
            name=f"setu-{key}-compile",
        )
        row.compile_url = getattr(compile_job, "url", None)
        target_model = compile_job.get_target_model()
        if target_model is None:
            row.error = "compile job produced no target model (see compile_url)"
            return row

        profile_job = hub.submit_profile_job(
            model=target_model, device=device, name=f"setu-{key}-profile"
        )
        row.profile_url = getattr(profile_job, "url", None)
        profile = profile_job.download_profile()
    except Exception as exc:
        row.error = f"{type(exc).__name__}: {exc}"
        return row

    if not isinstance(profile, dict):
        row.error = "unexpected profile payload"
        return row

    row.inference_ms, row.peak_memory_mb = _summary_numbers(profile)
    (
        row.layers_total,
        row.layers_npu,
        row.layers_gpu,
        row.layers_cpu,
    ) = _count_compute_units(profile)
    return row


def to_markdown(payload: dict) -> str:
    rows = payload["rows"]
    lines = [
        "# On-device profiling (Qualcomm AI Hub)",
        "",
        "Every number here was measured on **real Snapdragon silicon**, provisioned by",
        "Qualcomm AI Hub. Nothing is estimated and nothing is copied from a datasheet.",
        "Each row links to its job on Qualcomm's own site so it can be verified directly.",
        "",
        f"- Generated: {payload['generated']}",
        f"- Device: `{payload['device']}`",
        f"- Target runtime: QNN context binary (Hexagon NPU)",
        "",
        "`NPU layers` is the important column. A model that compiles but scatters half its",
        "layers back onto the CPU is not really running on the NPU, and the whole premise",
        "of this project depends on knowing the difference.",
        "",
        "| Model | Inference | Peak memory | NPU layers | Verify |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row["error"]:
            lines.append(
                f"| `{row['model']}` | — | — | — | {row['error'][:70]} |"
            )
            continue
        layers = row["layers"]
        frac = row["npu_fraction"]
        npu_cell = (
            f"{layers['npu']}/{layers['total']} ({frac * 100:.0f}%)"
            if frac is not None
            else "—"
        )
        link = f"[job]({row['profile_url']})" if row["profile_url"] else "—"
        lines.append(
            "| `{model}` | {ms} | {mem} | {npu} | {link} |".format(
                model=row["model"],
                ms=f"{row['inference_ms']:.2f} ms" if row["inference_ms"] else "—",
                mem=f"{row['peak_memory_mb']:.1f} MB" if row["peak_memory_mb"] else "—",
                npu=npu_cell,
                link=link,
            )
        )

    lines += ["", "## What each model does", ""]
    for row in rows:
        if row["note"]:
            lines.append(f"- **`{row['model']}`** — {row['note']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", help="AI Hub device name; default picks a Snapdragon X part")
    parser.add_argument("--model", action="append", help="profile only these (repeatable)")
    parser.add_argument("--all", action="store_true", help="profile every target")
    parser.add_argument("--model-root", default=str(REPO / "models"))
    args = parser.parse_args(argv)

    if args.list_devices:
        list_devices()
        return 0

    if not args.all and not args.model:
        parser.error("pass --all, or --model NAME (repeatable), or --list-devices")

    hub = _require_hub()
    device = pick_device(hub, args.device)
    model_root = Path(args.model_root)
    selected = {k: v for k, v in TARGETS.items() if args.all or k in (args.model or [])}
    if not selected:
        parser.error(f"no such model. known: {', '.join(TARGETS)}")

    print(f"Profiling {len(selected)} model(s) on {device.name}\n")
    rows: list[ProfileRow] = []
    for key, spec in selected.items():
        print(f"  {key} ... ", end="", flush=True)
        row = profile_one(hub, key, spec, device, model_root)
        rows.append(row)
        if row.error:
            print(f"SKIPPED ({row.error[:60]})")
        else:
            frac = row.npu_fraction
            print(
                f"{row.inference_ms:.2f} ms"
                + (f", {frac * 100:.0f}% on NPU" if frac is not None else "")
            )

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "device": device.name,
        "rows": [r.as_dict() for r in rows],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(to_markdown(payload), encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(REPO)} and {OUT_JSON.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
