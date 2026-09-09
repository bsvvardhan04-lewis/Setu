"""Labelled corpus for the comprehension evaluation.

Hand-authored, and said so plainly: real consultation audio is not something a hackathon
entry can ethically collect. What these cases DO capture is the set of failure modes the
product exists to catch, written by hand so each one is inspectable and arguable.

`covered_markers` are substrings identifying the instructions the learner genuinely did
restate. Anything extracted that matches none of them is truly missed. Keeping the labels
in source rather than a binary blob means a reader can disagree with a specific judgement
instead of having to trust the aggregate.

Cases are English so that scoring is measured without the translation model confounding
it; cross-lingual sessions are reported by the pipeline as *unverified* rather than scored.
"""

from __future__ import annotations

from .evaluate import EvalCase

CORPUS: list[EvalCase] = [
    # ---------------------------------------------------------------- clinic
    EvalCase(
        case_id="clinic-partial-recall",
        domain="clinic",
        turns=[
            ("doctor", "You have hypertension, we start amlodipine 5 mg tablet one at night."),
            ("doctor", "Get a lipid profile done, fasting, before the next visit."),
            ("doctor", "Come back for follow up after 2 weeks."),
            ("doctor", "If you get chest pain or breathlessness, come immediately to the emergency."),
        ],
        restatement="I will take the tablet at night and come back after two weeks.",
        covered_markers=("amlodipine", "follow up after 2 weeks"),
        note="The classic case: remembers the drug and the date, forgets the red flag.",
    ),
    EvalCase(
        case_id="clinic-paraphrase",
        domain="clinic",
        turns=[
            ("doctor", "Take the amlodipine tablet once every night before sleeping."),
            ("doctor", "Avoid salt in your food and walk for thirty minutes daily."),
        ],
        restatement=(
            "I will swallow one pill each evening at bedtime. "
            "I will cut down on salty food and go for a half hour walk."
        ),
        covered_markers=("amlodipine", "salt"),
        note="Everything covered, but almost entirely in the patient's own words. "
        "Keyword matching fails this case; meaning matching should not.",
    ),
    EvalCase(
        case_id="clinic-understood-all",
        domain="clinic",
        turns=[
            ("doctor", "Take the antibiotic capsule twice a day after food for five days."),
            ("doctor", "Come back for review next week."),
        ],
        restatement=(
            "I will take the antibiotic capsule twice a day after food. "
            "I will come back for review next week."
        ),
        covered_markers=("antibiotic", "review next week"),
        note="Nothing missed. A check that raises alarms here would be ignored in a clinic.",
    ),
    EvalCase(
        case_id="clinic-understood-nothing",
        domain="clinic",
        turns=[
            ("doctor", "Take the metformin tablet twice a day with meals."),
            ("doctor", "Get an HbA1c test done, fasting, next month."),
            ("doctor", "If your sugar drops and you feel faint, eat something sweet immediately."),
        ],
        restatement="Yes doctor, okay.",
        covered_markers=(),
        note="Acknowledgement without content. Every item must be flagged.",
    ),
    EvalCase(
        case_id="clinic-redflag-only",
        domain="clinic",
        turns=[
            ("doctor", "Take the tablet once a day in the morning."),
            ("doctor", "Get a blood test done before the next visit."),
            ("doctor", "If you start bleeding, come immediately to the emergency."),
        ],
        restatement="If there is bleeding I must come to the emergency straight away.",
        covered_markers=("bleeding",),
        note="Inverse of the usual case: the warning landed, the routine instructions did not.",
    ),
    # ------------------------------------------------------------- classroom
    EvalCase(
        case_id="classroom-missed-deadline",
        domain="classroom",
        turns=[
            ("teacher", "Submit the lab record assignment by Friday, that is the deadline."),
            ("teacher", "The internal test is next month and has high weightage."),
            ("teacher", "Read chapter four in the textbook and bring your notes."),
        ],
        restatement="I have to read chapter four and bring my notes.",
        covered_markers=("chapter four",),
        note="Student registers the reading, misses both the deadline and the test.",
    ),
    EvalCase(
        case_id="classroom-understood",
        domain="classroom",
        turns=[
            ("teacher", "The internal test is on the fifteenth and covers the whole syllabus."),
            ("teacher", "Submit your project report before the deadline on Monday."),
        ],
        restatement=(
            "The internal test is on the fifteenth and covers the syllabus. "
            "I will submit the project report by Monday."
        ),
        covered_markers=("internal test", "project report"),
    ),
    # --------------------------------------------------------------- counter
    EvalCase(
        case_id="counter-missed-deadline",
        domain="counter",
        turns=[
            ("officer", "The last date is 30 September, after that you cannot apply this year."),
            ("officer", "Bring your Aadhaar, ration card and a photocopy of the passbook."),
            ("officer", "There is a fee of 50 rupees, pay at counter number three."),
        ],
        restatement="I will pay fifty rupees at counter three.",
        covered_markers=("fee of 50 rupees",),
        note="Remembers the money, misses the deadline and the documents - the exact "
        "failure that makes a citizen lose a year of pension.",
    ),
    EvalCase(
        case_id="counter-paraphrase",
        domain="counter",
        turns=[
            ("officer", "Bring your Aadhaar and a photocopy of the bank passbook."),
            ("officer", "The last date is 30 September."),
        ],
        restatement=(
            "So I need my Aadhaar card and a xerox of the bank book, "
            "and I must come before the end of September."
        ),
        covered_markers=("Aadhaar", "last date"),
        note="Covered, but with 'xerox' for 'photocopy' and 'end of September' for the date.",
    ),
    EvalCase(
        case_id="counter-understood-nothing",
        domain="counter",
        turns=[
            ("officer", "The last date for submission is 30 September."),
            ("officer", "Bring a self attested photocopy of your income certificate."),
            ("officer", "Collect the acknowledgement receipt before you leave."),
        ],
        restatement="Theek hai sahib.",
        covered_markers=(),
        note="Polite agreement, zero content.",
    ),
]
