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

import json
import logging
import re
import time
from dataclasses import dataclass, field

from ..config import SUPPORTED_LANGUAGES
from ..models import GenParams
from .domains import CLINIC, DEFAULT_DOMAIN, Domain, get_domain
from .engine import Engine

log = logging.getLogger(__name__)


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
    kind: str
    text: str
    source_turn: int
    confirmed: bool = False
    #: how strongly the learner's restatement matched this item, 0..1 (best of the two)
    similarity: float | None = None
    lexical: float | None = None
    semantic: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "text": self.text,
            "source_turn": self.source_turn,
            "confirmed": self.confirmed,
            "similarity": self.similarity,
            "lexical": self.lexical,
            "semantic": self.semantic,
        }


#: Kept as a module-level alias so existing imports keep working. The real lexicons
#: now live per-domain in `domains.py`, because the clinic's vocabulary is not the
#: classroom's and hard-coding one of them into the engine was the thing preventing
#: SETU from serving more than a single setting.
JARGON_LEXICON = CLINIC.jargon

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


def find_jargon(text: str, domain: Domain | None = None) -> list[str]:
    """Longest-match lexicon scan against this domain's vocabulary.

    Deterministic and instant, and it can never hallucinate a term that was not actually
    said. Whole words only, so "od" does not fire inside "amlodipine".
    """
    lexicon = (domain or get_domain(None)).jargon
    lowered = f" {re.sub(r'[^a-z0-9 ]+', ' ', text.lower())} "
    hits: list[str] = []
    for term in sorted(lexicon, key=len, reverse=True):
        if f" {term} " in lowered and not any(term in h for h in hits):
            hits.append(term)
    return hits


def extract_plan_rules(
    text: str, turn_index: int, domain: Domain | None = None
) -> list[CarePlanItem]:
    """Regex-based plan extraction, used as the floor under the LLM.

    Sentence-level so each item keeps its dosage and timing intact.
    """
    patterns = (domain or get_domain(None)).patterns
    items: list[CarePlanItem] = []
    for raw in re.split(r"(?<=[.!?।])\s+|\n", text):
        sentence = raw.strip()
        if len(sentence) < 6:
            continue
        for pattern, kind in patterns:
            if re.search(pattern, sentence, flags=re.IGNORECASE):
                items.append(CarePlanItem(kind=kind, text=sentence, source_turn=turn_index))
                break
    return items


@dataclass
class ConsultSession:
    session_id: str
    patient_language: str = "hi"
    doctor_language: str = "en"
    domain_key: str = DEFAULT_DOMAIN
    turns: list[Turn] = field(default_factory=list)
    plan: list[CarePlanItem] = field(default_factory=list)
    glosses: dict[str, str] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    @property
    def domain(self) -> Domain:
        return get_domain(self.domain_key)

    def transcript_text(self, speaker: str | None = None) -> str:
        return "\n".join(
            f"{t.speaker}: {t.text}" for t in self.turns if speaker is None or t.speaker == speaker
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "domain": self.domain.as_dict(),
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
    """Owns consultations. One instance per process.

    Sessions are held in memory for speed and written to disk after every change, so a
    restart does not lose a visit and a take-home card URL handed to a patient keeps
    working. The store is optional: without one the agent behaves exactly as it did
    before, which keeps tests and throwaway runs from leaving files behind.
    """

    def __init__(self, engine: Engine, store=None) -> None:
        self.engine = engine
        self.store = store
        self.sessions: dict[str, ConsultSession] = {}

    # ---------------------------------------------------------------- persistence

    def _persist(self, session: ConsultSession) -> None:
        """Write after every change rather than at some 'end' of the visit.

        There is no reliable end: a consultation stops when the patient walks out, the
        laptop lid closes, or the battery dies. Saving continuously is the only version
        that survives all three.
        """
        if self.store is None:
            return
        try:
            self.store.save(session)
        except Exception:  # pragma: no cover - storage must never break a live visit
            log.exception("could not persist consultation %s", session.session_id)

    def _restore(self, session_id: str) -> ConsultSession | None:
        if self.store is None:
            return None

        def build(head, turns, plan, glosses) -> ConsultSession:
            session = ConsultSession(
                session_id=head["session_id"],
                patient_language=head["patient_language"],
                doctor_language=head.get("doctor_language") or "en",
                domain_key=head["domain_key"],
                started_at=head["started_at"],
            )
            session.turns = [
                Turn(
                    speaker=t["speaker"],
                    text=t["text"],
                    language=t["language"] or "en",
                    at=t["at"] or time.time(),
                    jargon=json.loads(t["jargon"] or "[]"),
                )
                for t in turns
            ]
            session.plan = [
                CarePlanItem(
                    kind=p["kind"],
                    text=p["text"],
                    source_turn=p["source_turn"],
                    confirmed=bool(p["confirmed"]),
                    similarity=p["similarity"],
                    lexical=p["lexical"],
                    semantic=p["semantic"],
                )
                for p in plan
            ]
            session.glosses = dict(glosses)
            return session

        try:
            return self.store.load(session_id, build)
        except Exception:
            log.exception("could not restore consultation %s", session_id)
            return None

    def start(
        self,
        session_id: str,
        patient_language: str = "hi",
        domain: str = DEFAULT_DOMAIN,
    ) -> ConsultSession:
        session = ConsultSession(
            session_id=session_id,
            patient_language=patient_language,
            domain_key=get_domain(domain).key,
        )
        self.sessions[session_id] = session
        self._persist(session)
        return session

    def get(self, session_id: str) -> ConsultSession:
        session = self.sessions.get(session_id)
        if session is None:
            session = self._restore(session_id)
            if session is not None:
                self.sessions[session_id] = session
        if session is None:
            raise KeyError(f"no consultation session {session_id!r}")
        return session

    def forget(self, session_id: str) -> bool:
        """Delete a consultation from memory and from disk.

        Medical conversations should be easy to remove, so this is a first-class
        operation rather than something a user has to go find a database file for.
        """
        self.sessions.pop(session_id, None)
        return bool(self.store.delete(session_id)) if self.store else True

    def recent(self, limit: int = 25) -> list[dict]:
        return self.store.recent(limit) if self.store else []

    # ------------------------------------------------------------------ live turn

    def add_turn(self, session_id: str, speaker: str, text: str) -> dict:
        """Ingest one utterance. Everything downstream is incremental, so this stays
        inside the budget of a live caption."""
        session = self.get(session_id)
        detected = self.engine.lid.identify(text)
        turn = Turn(speaker=speaker, text=text, language=detected.value)

        translation = None
        if speaker in ("doctor", session.domain.expert) and turn.language != session.patient_language:
            result = self.engine.translate.translate(
                text, turn.language, session.patient_language
            )
            translation = {
                "text": result.value,
                "device": result.device,
                "degraded": result.degraded,
            }

        # "doctor" is the clinic's name for the expert; a classroom calls them the
        # teacher and a counter calls them the officer. The role is what matters.
        if speaker in ("doctor", session.domain.expert):
            turn.jargon = find_jargon(text, session.domain)
            for term in turn.jargon:
                if term not in session.glosses:
                    session.glosses[term] = self._gloss(
                        term, session.patient_language, session.domain
                    )
            session.plan.extend(
                extract_plan_rules(text, len(session.turns), session.domain)
            )

        session.turns.append(turn)
        self._persist(session)
        return {
            "turn": turn.as_dict(),
            "translation": translation,
            "new_glosses": {t: session.glosses[t] for t in turn.jargon},
            "plan_size": len(session.plan),
        }

    def _gloss(self, term: str, language: str, domain: Domain | None = None) -> str:
        """Lexicon first, model second. The lexicon is curated and safe; the model only
        fills gaps, and only for terms the doctor actually said."""
        known = (domain or get_domain(None)).jargon.get(term)
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
                r"\s*(MEDICATION|TEST|FOLLOWUP|REDFLAG|LIFESTYLE)\s*:\s*(.+)", line, re.IGNORECASE
            )
            if not match:
                continue
            kind, text = match.group(1).lower(), match.group(2).strip()
            if text.lower() in existing or len(text) < 4:
                continue
            session.plan.append(CarePlanItem(kind=kind, text=text, source_turn=-1))
            existing.add(text.lower())
            added += 1

        self._persist(session)
        return {
            "plan": [p.as_dict() for p in session.plan],
            "added": added,
            "device": generated.device,
            "degraded": generated.degraded,
        }

    def teachback_questions(self, session_id: str) -> list[str]:
        """The questions the patient is asked to answer in their own words."""
        session = self.get(session_id)
        present = {item.kind for item in session.plan}
        prompts = session.domain.questions
        # Ordered by the domain's own consequence ordering, not dict insertion order.
        return [prompts[k] for k in session.domain.kinds if k in present and k in prompts]

    def _score_coverage(self, session: ConsultSession, bridged: str, said: set[str]) -> str:
        """Decide which plan items the learner actually covered.

        Two independent signals, unioned. Neither is sufficient alone, and the evaluation
        corpus is what proved it rather than intuition:

        * **Lexical overlap** catches restatements that reuse the instruction's own words.
          It misses paraphrase entirely - "swallow one pill each evening" shares nothing
          with "amlodipine 5 mg one at night".
        * **Sentence-embedding cosine** catches paraphrase, but scores instruction against
          restatement lower than expected because the two are grammatically asymmetric
          (an imperative against a first-person promise). Measured on the corpus, truly
          covered items ran 0.34-0.99 and truly missed items 0.01-0.54 - overlapping
          ranges, so no cosine threshold separates them cleanly on its own.

        An item counts as covered if EITHER signal clears its threshold. Thresholds were
        chosen from a sweep over the labelled corpus, at the operating point that keeps
        DANGEROUS misses (telling a doctor the patient understood when they did not) at
        zero - see `docs/EVALUATION.md`. Precision is deliberately sacrificed for that:
        a needless repetition costs seconds, a missed red flag costs more.
        """
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?]) +|[\n,]", bridged)
            if part.strip()
        ]
        if not sentences or not session.plan:
            for item in session.plan:
                item.confirmed = False
            return "none"

        # --- signal 1: lexical overlap
        lexical: list[float] = []
        for item in session.plan:
            terms = {w.lower() for w in re.findall(r"\w{4,}", item.text, re.UNICODE)}
            keyed = terms - _STOPWORDS
            lexical.append(len(keyed & said) / len(keyed) if keyed else 0.0)

        # --- signal 2: semantic similarity, when real embeddings are loaded
        semantic: list[float] | None = None
        encoded = self.engine.embed.encode([p.text for p in session.plan] + sentences)
        if not encoded.degraded:
            import numpy as np

            vectors = np.asarray(encoded.value, dtype=np.float32)
            norms = np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)
            unit = vectors / norms
            similarity = unit[: len(session.plan)] @ unit[len(session.plan) :].T
            semantic = similarity.max(axis=1).tolist()

        for index, item in enumerate(session.plan):
            lex = lexical[index]
            sem = semantic[index] if semantic else None
            item.confirmed = lex >= LEXICAL_THRESHOLD or (
                sem is not None and sem >= SEMANTIC_THRESHOLD
            )
            item.similarity = round(max(lex, sem or 0.0), 3)
            item.lexical = round(lex, 3)
            item.semantic = None if sem is None else round(float(sem), 3)

        return "hybrid" if semantic else "lexical"

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
            self._persist(session)
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

        method = self._score_coverage(session, bridged, said)

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
        self._persist(session)
        return {
            "plan": [p.as_dict() for p in session.plan],
            "covered": len(session.plan) - len(missed),
            "total": len(session.plan),
            "unverified": False,
            "cross_lingual": cross_lingual,
            "method": method,
            "missed": [p.as_dict() for p in missed],
            "needs_repeat": [p.text for p in missed if p.kind in session.domain.critical],
        }


#: Chosen by sweeping the labelled corpus at the operating point that keeps dangerous
#: misses at zero. See docs/EVALUATION.md for the full sweep.
LEXICAL_THRESHOLD = 0.34
SEMANTIC_THRESHOLD = 0.54

_STOPWORDS = {
    "will", "must", "should", "this", "that", "your", "with", "from", "have", "take",
    "come", "after", "before", "every", "when", "then", "also", "need", "please",
}


class TakeHomeCard:
    """The artefact the patient walks out with.

    Deliberately not a transcript. A transcript is a record for the clinic; a card is an
    instrument for the patient. Ordered by what will hurt them if they forget it.
    """

    def __init__(self, agent: ConsultAgent) -> None:
        self.agent = agent

    def _translate_all(self, items, target: str):
        """Translate a section's items concurrently.

        Sequentially this was the slowest thing in the product - a five-item card took
        about twenty seconds, because every item pays the full encoder-plus-decode cost in
        turn. They are independent, and ONNX Runtime sessions are safe to call from
        several threads, so they go out together and the card costs roughly one
        translation instead of N.
        """
        if not items:
            return []
        if len(items) == 1:
            return [self.agent.engine.translate.translate(items[0].text, "en", target)]

        from concurrent.futures import ThreadPoolExecutor

        translate = self.agent.engine.translate.translate
        with ThreadPoolExecutor(max_workers=min(4, len(items))) as pool:
            return list(pool.map(lambda i: translate(i.text, "en", target), items))

    def build(self, session_id: str) -> dict:
        session = self.agent.get(session_id)
        domain = session.domain
        target = session.patient_language
        sections: list[dict] = []

        # domain.kinds is ordered most-consequential-first, so the card is too.
        for kind in domain.kinds:
            items = [p for p in session.plan if p.kind == kind]
            if not items:
                continue
            lines = [
                {
                    "source": item.text,
                    "translated": translated.value,
                    "translated_ok": not translated.degraded,
                    "confirmed": item.confirmed,
                }
                # strict=True: a length mismatch here would silently drop an
                # instruction from the patient's printed card, which is precisely the
                # kind of failure this project exists to prevent.
                for item, translated in zip(
                    items, self._translate_all(items, target), strict=True
                )
            ]
            sections.append({"kind": kind, "heading": domain.heading(kind), "items": lines})

        glossary = [
            {"term": term, "plain": gloss} for term, gloss in sorted(session.glosses.items())
        ]
        unconfirmed = [p.text for p in session.plan if not p.confirmed]

        return {
            "session_id": session_id,
            "domain": domain.key,
            "expert": domain.expert,
            "learner": domain.learner,
            "title": domain.card_title,
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
        lines = [f"SETU - {card['title']} ({card['language_name']})", "=" * 44, ""]
        for section in card["sections"]:
            lines.append(section["heading"].upper())
            for item in section["items"]:
                lines.append(f"  - {item['translated'] or item['source']}")
                if item["translated_ok"] and item["translated"] != item["source"]:
                    lines.append(f"    ({item['source']})")
            lines.append("")
        if card["glossary"]:
            lines.append(f"WORDS THE {self.agent.get(session_id).domain.expert.upper()} USED")
            lines.extend(f"  - {entry['term']}: {entry['plain']}" for entry in card["glossary"])
            lines.append("")
        if card["unconfirmed"]:
            expert = self.agent.get(session_id).domain.expert.upper()
            lines.append(f"PLEASE ASK THE {expert} TO REPEAT")
            lines.extend(f"  - {text}" for text in card["unconfirmed"])
        return "\n".join(lines)
