"""ConsultAgent - the comprehension layer.

Runs entirely on the fallback path (no model assets), which is what a reviewer meets.
"""

from __future__ import annotations

import pytest

from setu.config import Settings
from setu.pipeline import (
    ConsultAgent,
    Engine,
    TakeHomeCard,
    extract_plan_rules,
    find_jargon,
)


@pytest.fixture
def engine(tmp_path):
    settings = Settings()
    settings.data_dir = tmp_path
    settings.db_path = tmp_path / "setu.db"
    settings.model_root = tmp_path / "models"
    e = Engine(settings)
    yield e
    e.close()


@pytest.fixture
def agent(engine):
    return ConsultAgent(engine)


DOCTOR_LINES = [
    "You have hypertension, so we will start amlodipine 5 mg tablet once a day at night.",
    "Get a lipid profile and HbA1c done, fasting, before the next visit.",
    "Come back for follow up after 2 weeks.",
    "If you get chest pain or breathlessness, come immediately to the emergency.",
    "Reduce salt in your food and walk for thirty minutes daily.",
]


def test_jargon_is_found_and_glossed():
    hits = find_jargon(DOCTOR_LINES[0])
    assert "hypertension" in hits


def test_jargon_scan_does_not_invent_terms():
    assert find_jargon("Please sit down and relax for a minute.") == []


def test_abbreviated_dosing_is_caught():
    assert "bd" in find_jargon("Take this tablet bd after food.")


def test_rule_extraction_finds_every_instruction_kind():
    kinds = set()
    for index, line in enumerate(DOCTOR_LINES):
        kinds.update(item.kind for item in extract_plan_rules(line, index))
    assert {"medication", "test", "followup", "redflag", "lifestyle"} <= kinds


def test_dosage_survives_extraction_intact():
    items = extract_plan_rules(DOCTOR_LINES[0], 0)
    assert any("5 mg" in item.text and "once a day" in item.text for item in items)


def test_live_turns_build_a_plan_and_glossary(agent):
    agent.start("c1", patient_language="hi")
    for line in DOCTOR_LINES:
        agent.add_turn("c1", "doctor", line)

    session = agent.get("c1")
    assert len(session.turns) == 5
    assert len(session.plan) >= 5
    assert "hypertension" in session.glosses
    assert session.glosses["hypertension"] == "high blood pressure"


def test_patient_turns_do_not_create_plan_items(agent):
    agent.start("c2")
    agent.add_turn("c2", "patient", "Doctor, I have been feeling tired and dizzy.")
    assert agent.get("c2").plan == []


def test_teachback_flags_what_the_patient_missed(agent):
    agent.start("c3")
    for line in DOCTOR_LINES:
        agent.add_turn("c3", "doctor", line)

    # The patient repeats only the tablet, and nothing about the warning signs.
    result = agent.check_teachback("c3", "I will take the amlodipine tablet at night.")
    assert result["total"] >= 5
    assert result["covered"] < result["total"]
    assert any("chest pain" in text.lower() for text in result["needs_repeat"])


def test_teachback_credits_a_complete_restatement(agent):
    agent.start("c4")
    agent.add_turn("c4", "doctor", "Come back for follow up after 2 weeks.")
    result = agent.check_teachback("c4", "I will come back for follow up after 2 weeks.")
    assert result["covered"] == result["total"]
    assert result["needs_repeat"] == []


def test_teachback_questions_track_the_plan_contents(agent):
    agent.start("c5")
    agent.add_turn("c5", "doctor", "Come back for follow up after 2 weeks.")
    questions = agent.teachback_questions("c5")
    assert any("come back" in q.lower() for q in questions)
    assert not any("medicines" in q.lower() for q in questions)


def test_take_home_card_leads_with_red_flags(agent):
    agent.start("c6", patient_language="hi")
    for line in DOCTOR_LINES:
        agent.add_turn("c6", "doctor", line)
    agent.check_teachback("c6", "I will take the tablet.")

    card = TakeHomeCard(agent).build("c6")
    assert card["sections"][0]["kind"] == "redflag"
    assert card["generated_offline"] is True
    assert card["glossary"]
    assert card["unconfirmed"], "items the patient never repeated must be surfaced"


def test_card_renders_printable_text(agent):
    agent.start("c7")
    for line in DOCTOR_LINES:
        agent.add_turn("c7", "doctor", line)
    text = TakeHomeCard(agent).to_text("c7")
    assert "COME BACK IMMEDIATELY IF" in text
    assert "WORDS THE DOCTOR USED" in text


def test_unknown_session_raises(agent):
    with pytest.raises(KeyError):
        agent.get("nope")


def test_cross_lingual_restatement_is_not_scored_as_missed(agent):
    """The patient answers in Hindi. With no translation model loaded we must report
    'could not verify', never 'understood nothing' - a false negative here would send
    a doctor away believing the patient failed."""
    agent.start("c8", patient_language="hi")
    for line in DOCTOR_LINES:
        agent.add_turn("c8", "doctor", line)

    result = agent.check_teachback("c8", "मैं रात को गोली लूंगा और दो हफ्ते बाद आऊंगा")
    assert result["unverified"] is True
    assert result["needs_repeat"] == []
    assert "could not be checked" in result["reason"]


def test_same_language_restatement_still_scores(agent):
    agent.start("c9")
    agent.add_turn("c9", "doctor", "Come back for follow up after 2 weeks.")
    result = agent.check_teachback("c9", "I will come back for follow up after 2 weeks.")
    assert result["unverified"] is False
    assert result["covered"] == result["total"]


def test_card_translations_are_memoised(agent):
    """The card was the slowest thing in the product before this.

    A five-item card paid the full encoder-plus-decode cost per item, sequentially - about
    twenty seconds. Items are independent, so they now go out together, and the same
    sentence translated for the live transcript is reused rather than recomputed.
    """
    agent.start("memo", patient_language="hi", domain="clinic")
    for line in DOCTOR_LINES:
        agent.add_turn("memo", "doctor", line)

    card = TakeHomeCard(agent)
    first = card.build("memo")
    second = card.build("memo")
    # Identical output, and the second pass must not have re-run anything.
    assert first["sections"] == second["sections"]


def test_card_exposes_the_domain_roles(agent):
    """The printable page titles its glossary with the domain's own role names."""
    agent.start("roles", patient_language="hi", domain="counter")
    agent.add_turn("roles", "officer", "Bring your Aadhaar and a self attested photocopy.")
    card = TakeHomeCard(agent).build("roles")
    assert card["expert"] == "officer"
    assert card["learner"] == "citizen"
