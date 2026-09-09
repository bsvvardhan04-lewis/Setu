"""A scripted consultation, for demos and for the regression corpus.

A live demo that depends on a microphone working in a strange room, over a projector,
in front of judges, is a demo that fails. This module lets the UI replay a realistic
consultation turn by turn through the *real* pipeline - the same LID, jargon scan,
plan extraction and teach-back run on it. Only the audio capture is bypassed, and the
UI says so on screen.
"""

from __future__ import annotations

#: A hypertension follow-up at a district OPD. Written to contain every failure mode
#: the product exists to catch: an abbreviation ("od"), a term with no clean Hindi
#: equivalent ("lipid profile"), a red flag buried at the end of a long sentence, and a
#: patient who says "haan ji" without having understood.
HYPERTENSION_VISIT: list[dict[str, str]] = [
    {"speaker": "patient", "text": "Doctor sahib, sar mein dard rehta hai aur chakkar aate hain."},
    {"speaker": "doctor", "text": "Your BP is 160 over 100. You have hypertension."},
    {"speaker": "doctor", "text": "We start amlodipine 5 mg tablet, one at night, od."},
    {"speaker": "patient", "text": "Haan ji."},
    {"speaker": "doctor", "text": "This is chronic, so it is long term, not for ten days only."},
    {"speaker": "doctor", "text": "Get a lipid profile and HbA1c done, fasting, before the next visit."},
    {"speaker": "doctor", "text": "Come back for follow up after 2 weeks with the reports."},
    {"speaker": "doctor", "text": "If you get chest pain or breathlessness, come immediately to the emergency, do not wait."},
    {"speaker": "doctor", "text": "Reduce salt in your food and walk for thirty minutes daily."},
    {"speaker": "patient", "text": "Theek hai doctor sahib."},
]

#: What the patient says when asked to repeat the plan back. Deliberately incomplete:
#: they remember the tablet, and nothing about the warning signs. That gap is the
#: entire product demonstrated in one line.
HYPERTENSION_TEACHBACK = "I will take the tablet at night and come back after two weeks."

#: A first-year data structures lecture. Same failure modes, different uniform: an
#: abbreviation the student will not decode ("weightage"), a deadline buried mid-sentence,
#: and a student who says "yes sir" without having registered the submission date.
CLASSROOM_LECTURE: list[dict[str, str]] = [
    {"speaker": "teacher", "text": "Today we cover asymptotic analysis, so please note that this is important."},
    {"speaker": "teacher", "text": "Submit the lab record assignment by Friday, that is the deadline."},
    {"speaker": "student", "text": "Yes sir."},
    {"speaker": "teacher", "text": "The internal test is next month and this topic has high weightage in the syllabus."},
    {"speaker": "teacher", "text": "Remember that the key idea is how the algorithm behaves when n becomes very large."},
    {"speaker": "teacher", "text": "Read chapter four in the textbook and bring your notes next class."},
    {"speaker": "teacher", "text": "If you copy from each other that is plagiarism and you will lose all marks."},
]
CLASSROOM_TEACHBACK = "I have to read chapter four and bring my notes."

#: A pension counter. The citizen cannot read the form and is about to miss a deadline.
COUNTER_VISIT: list[dict[str, str]] = [
    {"speaker": "citizen", "text": "Sahib, pension ka form bharna hai."},
    {"speaker": "officer", "text": "The last date is 30 September, after that you cannot apply this year."},
    {"speaker": "officer", "text": "Bring your Aadhaar, ration card and a self attested photocopy of the passbook."},
    {"speaker": "citizen", "text": "Theek hai."},
    {"speaker": "officer", "text": "There is a fee of 50 rupees, pay at counter number three."},
    {"speaker": "officer", "text": "You are eligible only if the household income is below the BPL limit."},
    {"speaker": "officer", "text": "After submitting, collect the acknowledgement receipt, do not leave without it."},
]
COUNTER_TEACHBACK = "I will pay fifty rupees at counter three."

_NOTE = (
    "Replayed from a script so the demo does not depend on room audio. "
    "Every turn runs through the real pipeline."
)

SCRIPTS: dict[str, dict] = {
    "hypertension": {
        "title": "Hypertension follow-up, district OPD",
        "domain": "clinic",
        "patient_language": "hi",
        "turns": HYPERTENSION_VISIT,
        "teachback": HYPERTENSION_TEACHBACK,
        "note": _NOTE,
    },
    "lecture": {
        "title": "Data structures lecture, first year engineering",
        "domain": "classroom",
        "patient_language": "te",
        "turns": CLASSROOM_LECTURE,
        "teachback": CLASSROOM_TEACHBACK,
        "note": _NOTE,
    },
    "pension": {
        "title": "Pension counter, block office",
        "domain": "counter",
        "patient_language": "hi",
        "turns": COUNTER_VISIT,
        "teachback": COUNTER_TEACHBACK,
        "note": _NOTE,
    },
}
