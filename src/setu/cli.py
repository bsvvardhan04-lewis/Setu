"""SETU command line: serve, doctor, bench, ask."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_serve(args: argparse.Namespace) -> int:
    from .config import settings
    from .server.app import main

    if args.port:
        settings.port = args.port
    main()
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Print exactly what this machine can do. First thing to run on a new device."""
    from .pipeline import get_engine

    report = get_engine().system_report()
    host = report["host"]
    print("SETU doctor")
    print("-" * 60)
    print(f"  OS            : {host['os']} {host['release']} ({host['arch']})")
    print(f"  Windows-on-ARM: {host['windows_on_arm']}")
    print(f"  ORT providers : {', '.join(host['onnxruntime_providers']) or 'onnxruntime missing'}")
    print(f"  Devices       : {', '.join(host['devices'])}")
    power = report["power"]
    print(f"  Power         : {'AC' if power['on_ac'] else 'battery'} "
          f"{power['battery_percent']}% draw={power['discharge_mw']} mW")
    print()
    print("  Model assets")
    for name, status in report["adapters"].items():
        mark = "OK  " if status["loaded"] else "MISS"
        place = status["placement"]["device"]
        print(f"    [{mark}] {name:<10} -> {place}")
    if report["sessions"]["missing"]:
        print()
        print("  Missing assets. Run: python scripts/fetch_models.py --all")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_benchmarks, save
    from .config import settings

    results = run_benchmarks(settings, iterations=args.iters, warmup=args.warmup)
    save(
        results,
        json_path=Path(args.json) if args.json else None,
        md_path=Path(args.report) if args.report else None,
    )
    print(json.dumps(results, indent=2))
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from .pipeline import DocAgent, get_engine

    agent = DocAgent(get_engine())
    if args.ingest:
        report = agent.ingest([Path(p) for p in args.ingest], title=args.title)
        print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    if args.question:
        answer = agent.ask(args.question, language=args.language)
        print(json.dumps(answer.as_dict(), indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="setu", description="Offline document assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the local server + UI")
    serve.add_argument("--port", type=int)
    serve.set_defaults(func=_cmd_serve)

    doctor = sub.add_parser("doctor", help="report hardware, providers and model assets")
    doctor.set_defaults(func=_cmd_doctor)

    bench = sub.add_parser("bench", help="benchmark every model on every device")
    bench.add_argument("--iters", type=int, default=20)
    bench.add_argument("--warmup", type=int, default=3)
    bench.add_argument("--json", default="bench_results/latest.json")
    bench.add_argument("--report", default="docs/BENCHMARKS.md")
    bench.set_defaults(func=_cmd_bench)

    ask = sub.add_parser("ask", help="ingest pages and/or ask a question")
    ask.add_argument("--ingest", nargs="*", help="image paths to ingest")
    ask.add_argument("--title", default="CLI document")
    ask.add_argument("--question")
    ask.add_argument("--language")
    ask.set_defaults(func=_cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
