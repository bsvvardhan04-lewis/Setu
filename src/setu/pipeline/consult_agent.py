"""ConsultAgent - the comprehension layer for a doctor-patient consultation.

SETU does not translate a consultation. Translation is table stakes and a phone can do
it. SETU verifies that the patient actually understood, which is the thing that changes
outcomes: the WHO puts non-adherence to prescribed treatment in chronic disease around
50% in developing countries, and "the patient did not understand the instruction" is a
large, unglamorous slice of that.

Four jobs, in order:

  1. TRANSCRIPT   every turn, attributed, with its language
  2. JARGON       terms the patient almost certainly did not parse, with a plain gloss
  3. CARE PLAN    the instructions actually given: drugs, doses, tests, follow-up, red flags
  4. TEACH-BACK   ask the patient to say the plan back, and check it against (3)

Only step 4 produces the artefact that matters: a take-home card, in the patient's
language, that reflects what was really said rather than what the clinician meant to say.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ..config import SUPPORTED_LANGUAGES
from ..models import GenParams
from .engine import Engine


@dataclass
class Turn:
    speaker: str  # "doctor" | "patient"
    text: str
    language: str = "en"
    at: float = field(default_factory=time.time)
    jargon: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "language": self.language,
            "at": round(self.at, 3),
            "jargon": self.jargon,
        }


@dataclass
class CarePlanItem:
    kind: str  # medication | test | followup | redflag | lifestyle
    text: str
    source_turn: int
    confirmed: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "text": self.text,
            "source_turn": self.source_turn,
            "confirmed": self.confirmed,
        }


#: Terms that routinely appear in an Indian OPD consultation and routinely are not
#: understood. Gloss text is deliberately plain: this is read aloud to the patient.
JARGON_LEXICON: dict[str, str] = {
    "hypertension": "high blood pressure",
    "hypotension": "low blood pressure",
    "diabetes mellitus": "high blood sugar",
    "hyperglycemia": "blood sugar that is too high",
    "hypoglycemia": "blood sugar that is too low",
    "anemia": "low haemoglobin, which makes you weak and tired",
    "hb": "haemoglobin, the iron level in your blood",
    "lipid profile": "a blood test for fat and cholesterol",
    "fasting": "with nothing to eat or drink for 8 to 10 hours before",
    "post prandial": "measured about two hours after eating",
    "hba1c": "a blood test showing your average sugar over three months",
    "antibiotic": "medicine that kills the infection",
    "analgesic": "pain relief medicine",
    "antipyretic": "medicine to bring the fever down",
    "prophylaxis": "medicine taken to stop a problem before it starts",
    "bd": "twice a day",
    "tds": "three times a day",
    "od": "once a day",
    "sos": "only if needed",
    "stat": "right now, immediately",
    "npo": "nothing to eat or drink",
    "chronic": "long lasting, needs ongoing care",
    "acute": "sudden and serious",
    "benign": "not cancer, not dangerous",
    "malignant": "cancer",
    "biopsy": "taking a small piece of tissue to test it",
    "ecg": "a heart tracing test",
    "usg": "an ultrasound scan",
    "cbc": "a complete blood count test",
    "renal": "related to the kidneys",
    "hepatic": "related to the liver",
    "cardiac": "related to the heart",
    "edema": "swelling from fluid",
    "dyspnea": "difficulty breathing",
    "syncope": "fainting",
    "titrate": "slowly adjust the dose",
    "adherence": "taking the medicine exactly as told",
    "follow up": "come back for a check",
    "referral": "being sent to another doctor or hospital",
}

#: Cues that a doctor turn contains an instruction rather than conversation.
_PLAN_PATTERNS: list[tuple[str, str]] = [
    (r"\b(tablet|tab|capsule|cap|syrup|mg|ml|dose|twice|thrice|once a day|bd|tds|od|sos)\b", "medication"),
    (r"\b(test|scan|x-?ray|ecg|usg|ultrasound|blood work|cbc|hba1c|lipid|biopsy|sample)\b", "test"),
    (r"\b(come back|follow.?up|review|next week|after \d+ (day|week|month)|revisit)\b", "followup"),
    (r"\b(if .*(worse|bleeding|chest pain|breathless|faint|vomit|fever above)|emergency|immediately)\b", "redflag"),
    (r"\b(avoid|stop|reduce|walk|exercise|diet|salt|sugar|water|rest|smoking|alcohol)\b", "lifestyle"),
]

_GLOSS_PROMPT = """Explain this medical term to a patient with no medical training.
Answer in ONE short sentence, under 15 words, in plain {language_name}.
Do not repeat the term itself. Do not add any preamble.

Term: {term}
Explanation:"""

_PLAN_PROMPT = """Read the doctor's instructions below and list ONLY the concrete actions
the patient must take. One per line. Start each line with one of these tags:
MEDICATION: / TEST: / FOLLOWUP: / REDFLAG: / LIFESTYLE:
Copy dosages, timings and dates exactly. Invent nothing. If there are no instructions,
output exactly: NONE

<CONTEXT>
{transcript}
</CONTEXT>

Actions:"""

_TEACHBACK_PROMPT = """The care plan the doctor gave is below, then what the patient said
when asked to repeat it back. For each plan item decide if the patient's restatement
covers it. Output one line per item: COVERED or MISSED, then the item text.

<CONTEXT>
PLAN:
{plan}

PATIENT SAID:
{restatement}
</CONTEXT>

Assessment:"""


def find_jargon(text: str) -> list[str]:
    """Longest-match lexicon scan. Deterministic, instant, and it never hallucinates
    a term that was not actually said."""
    lowered = f" {re.sub(r'[^a-z0-9 ]+', ' ', text.lower())} "
    hits: list[str] = []
    for term in sorted(JARGON_LEXICON, key=len, reverse=True):
        if f" {term} " in lowered and not any(term in h for h in hits):
            hits.append(term)
    return hits


def extract_plan_rules(text: str, turn_index: int) -> list[CarePlanItem]:
    """Regex-based plan extraction, used as the floor under the LLM.

    Sentence-level so each item keeps its dosage and timing intact.
    """
    items: list[CarePlanItem] = []
    for sentence in re.split(r"(?<=[.!?।])\s+|\n", text):
        sentence = sentence.strip()
        if len(sentence) < 6:
            continue
        for pattern, kind in _PLAN_PATTERNS:
            if re.search(pattern, sentence, flags=re.I):
                items.append(CarePlanItem(kind=kind, text=sentence, source_turn=turn_index))
                break
    return items


@dataclass
class ConsultSession:
    session_id: str
    patient_language: str = "hi"
    doctor_language: str = "en"
    turns: list[Turn] = field(default_factory=list)
    plan: list[CarePlanItem] = field(default_factory=list)
    glosses: dict[str, str] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def transcript_text(self, speaker: str | None = None) -> str:
        return "\n".join(
            f"{t.speaker}: {t.text}" for t in self.turns if speaker is None or t.speaker == speaker
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "patient_language": self.patient_language,
            "patient_language_name": SUPPORTED_LANGUAGES.get(
                self.patient_language, self.patient_language
            ),
            "doctor_language": self.doctor_language,
            "turns": [t.as_dict() for t in self.turns],
            "plan": [p.as_dict() for p in self.plan],
            "glosses": self.glosses,
            "duration_seconds": round(time.time() - self.started_at, 1),
        }


class ConsultAgent:
    """Owns live consultation sessions. One instance per process."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.sessions: dict[str, ConsultSession] = {}

    def start(self, session_id: str, patient_language: str = "hi") -> ConsultSession:
        session = ConsultSession(session_id=session_id, patient_language=patient_language)
        self.sessions[session_id] = session
        return session

    def get(self, session_id: str) -> ConsultSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"no consultation session {session_id!r}")
        return session

    # ------------------------------------------------------------------ live turn

    def add_turn(self, session_id: str, speaker: str, text: str) -> dict:
        """Ingest one utterance. Everything downstream is incremental, so this stays
        inside the budget of a live caption."""
        session = self.get(session_id)
        detected = self.engine.lid.identify(text)
        turn = Turn(speaker=speaker, text=text, language=detected.value)

        translation = None
        if speaker == "doctor" and turn.language != session.patient_language:
            result = self.engine.translate.translate(
                text, turn.language, session.patient_language
            )
            translation = {
                "text": result.value,
                "device": result.device,
                "degraded": result.degraded,
            }

        if speaker == "doctor":
            turn.jargon = find_jargon(text)
            for term in turn.jargon:
                if term not in session.glosses:
                    session.glosses[term] = self._gloss(term, session.patient_language)
            session.plan.extend(extract_plan_rules(text, len(session.turns)))

        session.turns.append(turn)
        return {
            "turn": turn.as_dict(),
            "translation": translation,
            "new_glosses": {t: session.glosses[t] for t in turn.jargon},
            "plan_size": len(session.plan),
        }

    def _gloss(self, term: str, language: str) -> str:
        """Lexicon first, model second. The lexicon is curated and safe; the model only
        fills gaps, and only for terms the doctor actually said."""
        known = JARGON_LEXICON.get(term)
        if known:
            return known
        generated = self.engine.llm.generate(
            _GLOSS_PROMPT.format(
                term=term, language_name=SUPPORTED_LANGUAGES.get(language, "English")
            ),
            GenParams(max_tokens=40, temperature=0.0),
        )
        return generated.value.strip().splitlines()[0] if generated.value else term

    # ------------------------------------------------------------------- teach-back

    def refine_plan(self, session_id: str) -> dict:
        """Run the LLM over the doctor's turns to catch instructions the regex missed.

        Rules-first, model-second, and the model can only ADD. It is never allowed to
        delete something the deterministic pass found, because a dropped red-flag
        instruction is the one failure mode that could actually hurt someone.
        """
        session = self.get(session_id)
        doctor_text = session.transcript_text("doctor")
        if not doctor_text.strip():
            return {"plan": [], "added": 0}

        generated = self.engine.llm.generate(
            _PLAN_PROMPT.format(transcript=doctor_text), GenParams(max_tokens=400)
        )
        existing = {p.text.lower().strip() for p in session.plan}
        added = 0
        for line in (generated.value or "").splitlines():
            match = re.match(
                r"\s*(MEDICATION|TEST|FOLLOWUP|REDFLAG|LIFESTYLE)\s*:\s*(.+)", line, re.I
            )
            if not match:
                continue
            kind, text = match.group(1).lower(), match.group(2).strip()
            if text.lower() in existing or len(text) < 4:
                continue
            session.plan.append(CarePlanItem(kind=kind, text=text, source_turn=-1))
            existing.add(text.lower())
            added += 1

        return {
            "plan": [p.as_dict() for p in session.plan],
            "added": added,
            "device": generated.device,
            "degraded": generated.degraded,
        }

    def teachback_questions(self, session_id: str) -> list[str]:
        """The questions the patient is asked to answer in their own words."""
        session = self.get(session_id)
        by_kind: dict[str, list[CarePlanItem]] = {}
        for item in session.plan:
            by_kind.setdefault(item.kind, []).append(item)

        prompts = {
            "medication": "Which medicines will you take, and how many times a day?",
            "test": "Which tests will you get done, and do you need to fast?",
            "followup": "When will you come back to see the doctor?",
            "redflag": "What warning signs mean you must come back immediately?",
            "lifestyle": "What changes will you make at home?",
        }
        return [prompts[k] for k in prompts if k in by_kind]

    def check_teachback(self, session_id: str, restatement: str) -> dict:
        """Score what the patient said back against the recorded plan.

        Term overlap first - it is transparent and cannot invent coverage. The model
        adds a second opinion but can only DOWNGRADE an item to missed, never upgrade a
        missed item to covered. Optimism here would defeat the entire purpose.
        """
        session = self.get(session_id)

        # The patient answers in their own language; the plan is recorded in the
        # doctor's. Comparing them directly scores every item as missed, which is a
        # worse failure than not checking at all - it would send a doctor away
        # believing the patient understood nothing. Bridge the languages first, and
        # if we cannot, say "unverified" rather than "missed".
        bridged = restatement
        cross_lingual = self.engine.lid.identify(restatement).value != "en"
        verifiable = True
        if cross_lingual:
            translated = self.engine.translate.translate(restatement, "auto", "en")
            if translated.degraded:
                verifiable = False
            else:
                bridged = translated.value

        said = {w.lower() for w in re.findall(r"\w{3,}", bridged, re.UNICODE)}

        if not verifiable:
            for item in session.plan:
                item.confirmed = False
            return {
                "plan": [p.as_dict() for p in session.plan],
                "covered": 0,
                "total": len(session.plan),
                "unverified": True,
                "reason": (
                    "The patient answered in another language and the translation model "
                    "is not loaded, so comprehension could not be checked. Ask the "
                    "patient to repeat in the doctor's language, or load IndicTrans2."
                ),
                "missed": [],
                "needs_repeat": [],
            }

        for item in session.plan:
            terms = {w.lower() for w in re.findall(r"\w{4,}", item.text, re.UNICODE)}
            keyed = terms - _STOPWORDS
            overlap = len(keyed & said) / len(keyed) if keyed else 0.0
            item.confirmed = overlap >= 0.34

        if session.plan:
            generated = self.engine.llm.generate(
                _TEACHBACK_PROMPT.format(
                    plan="\n".join(f"- {p.text}" for p in session.plan),
                    restatement=restatement,
                ),
                GenParams(max_tokens=400),
            )
            missed_by_model = {
                line.split(":", 1)[-1].strip().lower()
                for line in (generated.value or "").splitlines()
                if line.strip().upper().startswith("MISSED")
            }
            for item in session.plan:
                if item.confirmed and item.text.lower() in missed_by_model:
                    item.confirmed = False

        missed = [p for p in session.plan if not p.confirmed]
        return {
            "plan": [p.as_dict() for p in session.plan],
            "covered": len(session.plan) - len(missed),
            "total": len(session.plan),
            "unverified": False,
            "cross_lingual": cross_lingual,
            "missed": [p.as_dict() for p in missed],
            "needs_repeat": [p.text for p in missed if p.kind in ("medication", "redflag")],
        }


_STOPWORDS = {
    "will", "must", "should", "this", "that", "your", "with", "from", "have", "take",
    "come", "after", "before", "every", "when", "then", "also", "need", "please",
}


_CARD_HEADINGS = {
    "medication": "Your medicines",
    "test": "Tests to get done",
    "followup": "Come back on",
    "redflag": "Come back IMMEDIATELY if",
    "lifestyle": "At home",
}


class TakeHomeCard:
    """The artefact the patient walks out with.

    Deliberately not a transcript. A transcript is a record for the clinic; a card is an
    instrument for the patient. Ordered by what will hurt them if they forget it.
    """

    ORDER = ["redflag", "medication", "followup", "test", "lifestyle"]

    def __init__(self, agent: ConsultAgent) -> None:
        self.agent = agent

    def build(self, session_id: str) -> dict:
        session = self.agent.get(session_id)
        target = session.patient_language
        sections: list[dict] = []

        for kind in self.ORDER:
            items = [p for p in session.plan if p.kind == kind]
            if not items:
                continue
            lines = []
            for item in items:
                translated = self.agent.engine.translate.translate(item.text, "en", target)
                lines.append(
                    {
                        "source": item.text,
                        "translated": translated.value,
                        "translated_ok": not translated.degraded,
                        "confirmed": item.confirmed,
                    }
                )
            sections.append({"kind": kind, "heading": _CARD_HEADINGS[kind], "items": lines})

        glossary = [
            {"term": term, "plain": gloss} for term, gloss in sorted(session.glosses.items())
        ]
        unconfirmed = [p.text for p in session.plan if not p.confirmed]

        return {
            "session_id": session_id,
            "language": target,
            "language_name": SUPPORTED_LANGUAGES.get(target, target),
            "sections": sections,
            "glossary": glossary,
            "unconfirmed": unconfirmed,
            "duration_seconds": round(time.time() - session.started_at, 1),
            "generated_offline": True,
        }

    def to_text(self, session_id: str) -> str:
        """Plain text rendering, for printing on a clinic's thermal printer."""
        card = self.build(session_id)
        lines = [f"SETU - your visit summary ({card['language_name']})", "=" * 44, ""]
        for section in card["sections"]:
            lines.append(section["heading"].upper())
            for item in section["items"]:
                lines.append(f"  - {item['translated'] or item['source']}")
                if item["translated_ok"] and item["translated"] != item["source"]:
                    lines.append(f"    ({item['source']})")
            lines.append("")
        if card["glossary"]:
            lines.append("WORDS THE DOCTOR USED")
            for entry in card["glossary"]:
                lines.append(f"  - {entry['term']}: {entry['plain']}")
            lines.append("")
        if card["unconfirmed"]:
            lines.append("PLEASE ASK THE DOCTOR TO REPEAT")
            for text in card["unconfirmed"]:
                lines.append(f"  - {text}")
        return "\n".join(lines)
