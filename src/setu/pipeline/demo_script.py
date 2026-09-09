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

SCRIPTS: dict[str, dict] = {
    "hypertension": {
        "title": "Hypertension follow-up, district OPD",
        "patient_language": "hi",
        "turns": HYPERTENSION_VISIT,
        "teachback": HYPERTENSION_TEACHBACK,
        "note": "Replayed from a script so the demo does not depend on room audio. "
        "Every turn runs through the real pipeline.",
    }
}
