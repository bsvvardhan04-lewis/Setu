"""Measure whether the comprehension check actually works.

    python scripts/run_eval.py
    python scripts/run_eval.py --compare   # semantic vs lexical scoring, side by side
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", default=str(REPO / "bench_results" / "eval.json"))
    parser.add_argument("--report", default=str(REPO / "docs" / "EVALUATION.md"))
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also score with embeddings disabled, to show what they buy",
    )
    args = parser.parse_args(argv)

    from setu.bench.corpus import CORPUS
    from setu.bench.evaluate import run_evaluation, save

    report = run_evaluation(CORPUS)
    _print(report)

    if args.compare:
        # Force the lexical path by pointing the embedder at an empty model root, so the
        # adapter falls back to hashed n-grams and the comparison is like-for-like.
        import tempfile

        from setu.config import Settings

        lexical_settings = Settings()
        lexical_settings.model_root = Path(tempfile.mkdtemp()) / "no-models"
        lexical_settings.db_path = Path(tempfile.mkdtemp()) / "eval.db"
        baseline = run_evaluation(CORPUS, lexical_settings)
        print("\n--- without sentence embeddings (lexical fallback) ---")
        _print(baseline)
        report["baseline_lexical"] = {
            k: baseline[k]
            for k in ("method", "recall_on_missed", "precision_on_missed", "f1",
                      "dangerous_misses", "false_alarms")
        }

    save(report, json_path=Path(args.json), md_path=Path(args.report))
    print(f"\nWrote {args.report}")
    return 0


def _print(report: dict) -> None:
    def pct(v):
        return "n/a" if v is None else f"{v * 100:.1f}%"

    print(f"  method            : {report['method']}")
    print(f"  cases / items     : {report['cases']} / {report['items']}")
    print(f"  recall on missed  : {pct(report['recall_on_missed'])}   <- the number that matters")
    print(f"  precision         : {pct(report['precision_on_missed'])}")
    print(f"  f1                : {pct(report['f1'])}")
    print(f"  DANGEROUS misses  : {report['dangerous_misses']}  (said covered, actually missed)")
    print(f"  false alarms      : {report['false_alarms']}")


if __name__ == "__main__":
    raise SystemExit(main())
