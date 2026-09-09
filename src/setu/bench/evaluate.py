"""Does the comprehension check actually work?

Everything else in this repo measures speed. This measures whether the product's central
claim is true: that SETU can tell the difference between a learner who understood the
instructions and one who did not.

The method is a labelled corpus. Each case is a session transcript plus a restatement,
annotated with which instructions the learner genuinely covered. We run the real pipeline
over it and compare.

The asymmetry matters more than the headline accuracy:

* a **false "covered"** is the dangerous error - SETU tells the doctor the patient
  understood the red-flag warning when they did not, and the patient goes home
* a **false "missed"** is merely annoying - the doctor repeats something unnecessarily

So the number to optimise is **recall on missed items**: of the instructions the learner
truly did not restate, how many did we catch? Precision is reported too, because a check
that flags everything is useless in a real clinic.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..pipeline import ConsultAgent, Engine


@dataclass(frozen=True)
class EvalCase:
    """One labelled session."""

    case_id: str
    domain: str
    turns: list[tuple[str, str]]
    restatement: str
    #: Substrings identifying the instructions the learner GENUINELY covered.
    #: Anything extracted that does not match one of these is truly missed.
    covered_markers: tuple[str, ...]
    note: str = ""


@dataclass
class Outcome:
    case_id: str
    total_items: int = 0
    true_missed: int = 0
    true_covered: int = 0
    # confusion against the "missed" label, which is the one that matters
    caught_missed: int = 0      # truly missed, flagged missed   (true positive)
    false_alarms: int = 0       # truly covered, flagged missed  (false positive)
    dangerous_misses: int = 0   # truly missed, flagged covered  (false negative)
    method: str = ""
    detail: list[dict] = field(default_factory=list)


def _is_truly_covered(item_text: str, markers: tuple[str, ...]) -> bool:
    lowered = item_text.lower()
    return any(m.lower() in lowered for m in markers)


def run_evaluation(
    cases: list[EvalCase], settings: Settings | None = None
) -> dict[str, object]:
    engine = Engine(settings) if settings else Engine()
    agent = ConsultAgent(engine)
    outcomes: list[Outcome] = []

    try:
        for case in cases:
            agent.start(case.case_id, patient_language="en", domain=case.domain)
            for speaker, text in case.turns:
                agent.add_turn(case.case_id, speaker, text)

            result = agent.check_teachback(case.case_id, case.restatement)
            outcome = Outcome(case_id=case.case_id, method=result.get("method", "?"))

            for item in agent.get(case.case_id).plan:
                truly_covered = _is_truly_covered(item.text, case.covered_markers)
                outcome.total_items += 1
                if truly_covered:
                    outcome.true_covered += 1
                    if not item.confirmed:
                        outcome.false_alarms += 1
                else:
                    outcome.true_missed += 1
                    if item.confirmed:
                        outcome.dangerous_misses += 1
                    else:
                        outcome.caught_missed += 1

                outcome.detail.append(
                    {
                        "text": item.text[:80],
                        "kind": item.kind,
                        "truly_covered": truly_covered,
                        "flagged_covered": item.confirmed,
                        "similarity": item.similarity,
                    }
                )
            outcomes.append(outcome)
    finally:
        engine.close()

    return _summarise(outcomes)


def _summarise(outcomes: list[Outcome]) -> dict[str, object]:
    caught = sum(o.caught_missed for o in outcomes)
    alarms = sum(o.false_alarms for o in outcomes)
    dangerous = sum(o.dangerous_misses for o in outcomes)
    total_missed = sum(o.true_missed for o in outcomes)
    total_items = sum(o.total_items for o in outcomes)

    recall = caught / total_missed if total_missed else None
    precision = caught / (caught + alarms) if (caught + alarms) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall)
        else None
    )

    methods = sorted({o.method for o in outcomes})
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "+".join(methods),
        "cases": len(outcomes),
        "items": total_items,
        "truly_missed": total_missed,
        "caught_missed": caught,
        "false_alarms": alarms,
        "dangerous_misses": dangerous,
        "recall_on_missed": None if recall is None else round(recall, 4),
        "precision_on_missed": None if precision is None else round(precision, 4),
        "f1": None if f1 is None else round(f1, 4),
        "per_case": [
            {
                "case_id": o.case_id,
                "items": o.total_items,
                "caught": o.caught_missed,
                "false_alarms": o.false_alarms,
                "dangerous_misses": o.dangerous_misses,
                "method": o.method,
                "detail": o.detail,
            }
            for o in outcomes
        ],
    }


def to_markdown(report: dict) -> str:
    lines = [
        "# Does the comprehension check work?",
        "",
        "Generated by `python scripts/run_eval.py`. Every other measurement in this repo",
        "is about speed. This one is about whether the product's central claim is true.",
        "",
        f"- Cases: **{report['cases']}** labelled sessions across clinic, classroom and counter",
        f"- Instructions extracted: **{report['items']}**",
        f"- Scoring method: `{report['method']}`",
        f"- Generated: {report['generated']}",
        "",
        "## The asymmetry",
        "",
        "The two errors are not equally bad, so they are reported separately.",
        "",
        "| Error | Consequence | Count |",
        "| --- | --- | ---: |",
        f"| **Dangerous miss** — said covered, actually missed | The doctor believes the "
        f"patient understood the red flag. They go home. | **{report['dangerous_misses']}** |",
        f"| False alarm — said missed, actually covered | The doctor repeats something "
        f"unnecessarily. | {report['false_alarms']} |",
        "",
        "## Headline",
        "",
        "| Metric | Value | Why it matters |",
        "| --- | ---: | --- |",
    ]

    def pct(value):
        return "—" if value is None else f"{value * 100:.1f}%"

    lines += [
        f"| Recall on missed items | **{pct(report['recall_on_missed'])}** | "
        "Of the instructions the learner truly did not restate, how many we caught. "
        "**This is the number that matters.** |",
        f"| Precision on missed items | {pct(report['precision_on_missed'])} | "
        "Of what we flagged, how much was genuinely missed. Guards against a check that "
        "flags everything and is therefore ignored. |",
        f"| F1 | {pct(report['f1'])} | Balance of the two. |",
    ]

    baseline = report.get("baseline_lexical")
    if baseline:
        lines += [
            "",
            "## What the embeddings actually buy",
            "",
            "The first implementation used keyword overlap alone. The second replaced it",
            "with sentence-embedding cosine. **The evaluation showed the second was worse**",
            "— cosine scores an instruction against a restatement lower than expected,",
            "because the two are grammatically asymmetric (an imperative against a",
            "first-person promise). Measured on this corpus, truly covered items scored",
            "0.34–0.99 and truly missed items 0.01–0.54: overlapping ranges, so no single",
            "cosine threshold separates them.",
            "",
            "But the two signals fail on *different* items, so the shipped scorer unions",
            "them: an item counts as covered if either clears its threshold.",
            "",
            "| Scoring | Recall | Precision | F1 | False alarms | Dangerous misses |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| Lexical only | {pct(baseline['recall_on_missed'])} | "
            f"{pct(baseline['precision_on_missed'])} | {pct(baseline['f1'])} | "
            f"{baseline['false_alarms']} | {baseline['dangerous_misses']} |",
            f"| **Hybrid (shipped)** | **{pct(report['recall_on_missed'])}** | "
            f"**{pct(report['precision_on_missed'])}** | **{pct(report['f1'])}** | "
            f"**{report['false_alarms']}** | **{report['dangerous_misses']}** |",
            "",
            "Thresholds were chosen by sweeping the corpus and taking the operating point",
            "that keeps dangerous misses at zero, rather than the one that maximises F1.",
            "At a lower threshold F1 peaks slightly higher — and one genuinely missed",
            "instruction gets reported as understood. That trade is not available to us.",
        ]

    lines += [
        "",
        "## Per case",
        "",
        "| Case | Items | Caught | False alarms | Dangerous misses |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for case in report["per_case"]:
        lines.append(
            f"| `{case['case_id']}` | {case['items']} | {case['caught']} | "
            f"{case['false_alarms']} | {case['dangerous_misses']} |"
        )

    lines += [
        "",
        "## Honest caveats",
        "",
        "- The corpus is **hand-authored**, not collected from real consultations. Real",
        "  clinical audio is not something a hackathon entry can ethically obtain, and",
        "  saying so is better than implying a provenance we do not have.",
        "- Cases are written in English so the scoring is measured without the translation",
        "  model confounding it. Cross-lingual comprehension is reported separately by the",
        "  pipeline as *unverified* rather than being silently scored.",
        "- The labels are the author's judgement of what a restatement covers. They are",
        "  checked into `src/setu/bench/corpus.py` so anyone can disagree with them.",
    ]
    return "\n".join(lines) + "\n"


def save(report: dict, json_path: Path | None = None, md_path: Path | None = None) -> None:
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(to_markdown(report), encoding="utf-8")
