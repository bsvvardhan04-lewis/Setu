"""Benchmark every model on every engine this machine has, and write the report.

Local counterpart to `aihub_profile.py`: that one measures on Qualcomm's cloud devices,
this one measures on the machine in front of you, including real battery draw.

    python scripts/run_bench.py --iters 30
    python scripts/run_bench.py --device cpu --device npu --model asr
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--device", action="append", help="pin to these engines (npu/gpu/cpu); repeatable"
    )
    parser.add_argument("--model", action="append", help="benchmark only these; repeatable")
    parser.add_argument("--json", default=str(REPO / "bench_results" / "latest.json"))
    parser.add_argument("--report", default=str(REPO / "docs" / "BENCHMARKS.md"))
    args = parser.parse_args(argv)

    from setu.bench import run_benchmarks, save
    from setu.config import settings
    from setu.runtime import Device, read_power_state

    power = read_power_state()
    if power.on_ac:
        print(
            "NOTE: running on AC power, so energy-per-inference cannot be measured.\n"
            "      Unplug the charger to get the milliwatt column filled in.\n"
        )

    devices = None
    if args.device:
        try:
            devices = [Device(d.lower()) for d in args.device]
        except ValueError as exc:
            parser.error(f"{exc}; valid: npu, gpu, cpu")

    results = run_benchmarks(
        settings,
        iterations=args.iters,
        warmup=args.warmup,
        devices=devices,
        models=args.model,
    )
    save(results, json_path=Path(args.json), md_path=Path(args.report))

    for row in results["rows"]:
        flag = " [degraded]" if row["degraded"] else ""
        p50 = f"{row['p50_ms']:.2f} ms" if row["p50_ms"] else "—"
        print(f"  {row['model']:<16} {row['device']:<5} {p50:>10}{flag}")
    print(f"\nWrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
