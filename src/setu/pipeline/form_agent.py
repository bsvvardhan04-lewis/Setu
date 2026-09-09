"""FormAgent - fill a form by voice, with a rule layer the LLM cannot overrule.

This is the part of SETU that touches a citizen's real data, so the design point is
distrust of the model. The LLM proposes a value for a slot; deterministic validators
decide whether it is allowed anywhere near the form. An Aadhaar number that fails its
Verhoeff checksum is rejected no matter how confident the model sounded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from ..models import GenParams
from .engine import Engine

_EXTRACT = """Extract the value for one form field from what the user said.
Return ONLY the value, with no explanation and no quotes. If the user did not say
anything that fits this field, return exactly: NONE

Field: {label} ({kind})
User said: {utterance}
Value:"""


@dataclass
class Field:
    key: str
    label: str
    kind: str = "text"  # text | digits | aadhaar | mobile | pincode | date | amount
    required: bool = True
    value: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "required": self.required,
            "value": self.value,
            "error": self.error,
            "filled": self.value is not None and self.error is None,
        }


@dataclass
class FormState:
    title: str
    fields: list[Field] = field(default_factory=list)

    def by_key(self, key: str) -> Field | None:
        return next((f for f in self.fields if f.key == key), None)

    def next_empty(self) -> Field | None:
        return next((f for f in self.fields if f.value is None and f.required), None)

    def as_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "fields": [f.as_dict() for f in self.fields],
            "complete": all(f.value is not None or not f.required for f in self.fields),
        }


def _today() -> date:
    """Indirection so tests can reason about the two-digit-year pivot."""
    return date.today()


_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_TELUGU_DIGITS = str.maketrans("౦౧౨౩౪౫౬౭౮౯", "0123456789")
_TAMIL_DIGITS = str.maketrans("௦௧௨௩௪௫௬௭௮௯", "0123456789")


def normalise_digits(text: str) -> str:
    """Indic numerals are common in dictated and printed forms; ASCII them first."""
    return (
        text.translate(_DEVANAGARI_DIGITS)
        .translate(_TELUGU_DIGITS)
        .translate(_TAMIL_DIGITS)
    )


#: Verhoeff tables - the checksum UIDAI uses for Aadhaar numbers.
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) != 12 or digits[0] in (0, 1):
        return False
    check = 0
    for i, digit in enumerate(reversed(digits)):
        check = _D[check][_P[i % 8][digit]]
    return check == 0


def validate(kind: str, raw: str) -> tuple[str | None, str | None]:
    """Return (cleaned_value, error). Deterministic; the model gets no vote here."""
    value = normalise_digits(raw).strip()
    if not value or value.upper() == "NONE":
        return None, None

    if kind == "aadhaar":
        digits = re.sub(r"\D", "", value)
        if len(digits) != 12:
            return None, "Aadhaar must be exactly 12 digits"
        if not verhoeff_valid(digits):
            return None, "Aadhaar checksum failed - please repeat the number"
        return f"{digits[0:4]} {digits[4:8]} {digits[8:12]}", None

    if kind == "mobile":
        digits = re.sub(r"\D", "", value)[-10:]
        if len(digits) != 10 or digits[0] not in "6789":
            return None, "Indian mobile numbers are 10 digits starting with 6-9"
        return digits, None

    if kind == "pincode":
        digits = re.sub(r"\D", "", value)
        if len(digits) != 6 or digits[0] == "0":
            return None, "PIN code must be 6 digits and cannot start with 0"
        return digits, None

    if kind == "date":
        match = re.search(r"(\d{1,2})\D+(\d{1,2})\D+(\d{2,4})", value)
        if not match:
            return None, "Could not read a date"
        day, month, year = (int(g) for g in match.groups())
        if year < 100:
            # Two-digit years are almost always a date of birth, so pivot on today
            # rather than blindly assuming 20xx: "5-3-89" is 1989, not 2089.
            current = _today().year
            year = year + 2000 if year + 2000 <= current else year + 1900
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return None, "That date does not exist"
        return f"{day:02d}/{month:02d}/{year:04d}", None

    if kind == "amount":
        digits = re.sub(r"[^\d.]", "", value)
        if not digits:
            return None, "Could not read an amount"
        return digits, None

    if kind == "digits":
        digits = re.sub(r"\D", "", value)
        return (digits, None) if digits else (None, "Expected a number")

    return value, None


#: A small library of real forms, so the demo is concrete rather than abstract.
TEMPLATES: dict[str, FormState] = {
    "pension": FormState(
        title="National Social Assistance Pension - application",
        fields=[
            Field("name", "Full name as on Aadhaar"),
            Field("aadhaar", "Aadhaar number", "aadhaar"),
            Field("dob", "Date of birth", "date"),
            Field("mobile", "Mobile number", "mobile"),
            Field("village", "Village / ward"),
            Field("pincode", "PIN code", "pincode"),
            Field("bank_account", "Bank account number", "digits"),
            Field("income", "Annual household income", "amount", required=False),
        ],
    ),
    "scholarship": FormState(
        title="Post-matric scholarship - application",
        fields=[
            Field("name", "Student name"),
            Field("aadhaar", "Aadhaar number", "aadhaar"),
            Field("dob", "Date of birth", "date"),
            Field("institution", "College or school name"),
            Field("course", "Course of study"),
            Field("mobile", "Mobile number", "mobile"),
            Field("pincode", "PIN code", "pincode"),
        ],
    ),
    "grievance": FormState(
        title="Public grievance - complaint",
        fields=[
            Field("name", "Your name"),
            Field("mobile", "Mobile number", "mobile"),
            Field("department", "Department concerned"),
            Field("subject", "Subject of complaint"),
            Field("details", "What happened"),
        ],
    ),
}


class FormAgent:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.sessions: dict[str, FormState] = {}

    def start(self, session_id: str, template: str) -> FormState:
        source = TEMPLATES.get(template)
        if source is None:
            raise KeyError(f"unknown form template: {template}")
        state = FormState(
            title=source.title,
            fields=[Field(f.key, f.label, f.kind, f.required) for f in source.fields],
        )
        self.sessions[session_id] = state
        return state

    def state(self, session_id: str) -> FormState | None:
        return self.sessions.get(session_id)

    def fill(self, session_id: str, utterance: str, field_key: str | None = None) -> dict:
        state = self.sessions.get(session_id)
        if state is None:
            raise KeyError("no such form session")

        target = state.by_key(field_key) if field_key else state.next_empty()
        if target is None:
            return {"form": state.as_dict(), "message": "Every required field is filled."}

        # Try the deterministic parser first. For a digits-shaped field a plain regex is
        # more reliable than a language model, costs nothing, and never hallucinates.
        cleaned, error = validate(target.kind, utterance)
        used_llm = False
        devices: dict[str, str] = {}
        timings: dict[str, float] = {}

        if cleaned is None and error is None or (error and target.kind != "text"):
            generated = self.engine.llm.generate(
                _EXTRACT.format(label=target.label, kind=target.kind, utterance=utterance),
                GenParams(max_tokens=48, temperature=0.0),
            )
            used_llm = True
            devices["llm"] = generated.device
            timings["llm"] = generated.latency_ms
            proposal = generated.value.strip().splitlines()[0] if generated.value else ""
            cleaned, error = validate(target.kind, proposal)

        target.value = cleaned
        target.error = error

        following = state.next_empty()
        return {
            "form": state.as_dict(),
            "field": target.as_dict(),
            "next_prompt": following.label if following else None,
            "used_llm": used_llm,
            "devices": devices,
            "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        }
