"""The same engine has to work in three settings, not just the clinic.

These tests exist to stop the clinic's vocabulary leaking back into the engine. If a
classroom session starts flagging "hypertension" or ordering its card by "redflag", the
abstraction has quietly collapsed back into a single-purpose app.
"""

from __future__ import annotations

import pytest

from setu.config import Settings
from setu.pipeline import (
    CLASSROOM,
    CLINIC,
    COUNTER,
    DOMAINS,
    ConsultAgent,
    Engine,
    TakeHomeCard,
    extract_plan_rules,
    find_jargon,
    get_domain,
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


def test_three_domains_are_registered():
    assert set(DOMAINS) == {"clinic", "classroom", "counter"}


@pytest.mark.parametrize("domain", DOMAINS.values(), ids=lambda d: d.key)
def test_every_domain_is_internally_consistent(domain):
    # Every category must have a heading and a teach-back question, or the card and the
    # check will silently skip it.
    for kind in domain.kinds:
        assert kind in domain.headings, f"{domain.key}: {kind} has no heading"
        assert kind in domain.questions, f"{domain.key}: {kind} has no question"
    assert set(domain.critical) <= set(domain.kinds)
    assert domain.jargon, f"{domain.key} has an empty lexicon"
    assert domain.expert != domain.learner


def test_unknown_domain_falls_back_to_clinic():
    assert get_domain("nonsense").key == "clinic"
    assert get_domain(None).key == "clinic"


# ------------------------------------------------------------------ jargon isolation


def test_jargon_lexicons_do_not_leak_between_domains():
    medical = "The patient has hypertension and needs a lipid profile."
    assert "hypertension" in find_jargon(medical, CLINIC)
    assert find_jargon(medical, CLASSROOM) == []
    assert find_jargon(medical, COUNTER) == []


def test_classroom_catches_its_own_vocabulary():
    hits = find_jargon("This topic has high weightage in the syllabus.", CLASSROOM)
    assert "weightage" in hits and "syllabus" in hits


def test_counter_catches_bureaucratic_vocabulary():
    hits = find_jargon("Bring a self attested photocopy and the acknowledgement.", COUNTER)
    assert "self attested" in hits and "acknowledgement" in hits


# ------------------------------------------------------------------- extraction


def test_classroom_extracts_deadlines_and_exams():
    kinds = set()
    for text in [
        "Submit the lab record assignment by Friday, that is the deadline.",
        "The internal test is next month and has high weightage.",
        "Read chapter four in the textbook.",
    ]:
        kinds.update(i.kind for i in extract_plan_rules(text, 0, CLASSROOM))
    assert {"assignment", "exam", "resource"} <= kinds


def test_counter_extracts_deadline_documents_and_fee():
    kinds = set()
    for text in [
        "The last date is 30 September.",
        "Bring your Aadhaar and a photocopy of the passbook.",
        "There is a fee of 50 rupees.",
    ]:
        kinds.update(i.kind for i in extract_plan_rules(text, 0, COUNTER))
    assert {"deadline", "document", "fee"} <= kinds


# ------------------------------------------------------------------ end to end


def test_classroom_session_uses_teacher_and_student_roles(agent):
    agent.start("k1", patient_language="te", domain="classroom")
    result = agent.add_turn("k1", "teacher", "Submit the assignment by Friday, deadline.")
    assert result["plan_size"] >= 1

    # A student turn must not create plan items - only the expert issues instructions.
    agent.add_turn("k1", "student", "Yes sir, understood.")
    assert all(p.source_turn == 0 for p in agent.get("k1").plan)


def test_counter_card_leads_with_the_deadline(agent):
    agent.start("c1", patient_language="hi", domain="counter")
    agent.add_turn("c1", "officer", "Bring your Aadhaar and ration card.")
    agent.add_turn("c1", "officer", "The last date is 30 September, do not be late.")

    card = TakeHomeCard(agent).build("c1")
    assert card["domain"] == "counter"
    assert card["sections"][0]["kind"] == "deadline"
    assert card["title"] == "What you need to do"


def test_teachback_questions_come_from_the_active_domain(agent):
    agent.start("k2", domain="classroom")
    agent.add_turn("k2", "teacher", "The internal test is next month, high weightage.")
    questions = agent.teachback_questions("k2")
    assert any("test" in q.lower() for q in questions)
    assert not any("medicine" in q.lower() for q in questions)


def test_critical_categories_are_domain_specific(agent):
    agent.start("c2", domain="counter")
    agent.add_turn("c2", "officer", "The last date is 30 September, do not be late.")
    agent.add_turn("c2", "officer", "Bring your Aadhaar and ration card.")

    result = agent.check_teachback("c2", "I will bring something.")
    # A missed deadline is critical at a counter, the way a missed red flag is in a clinic.
    assert any("30 september" in t.lower() for t in result["needs_repeat"])


def test_clinic_behaviour_is_unchanged_by_the_refactor(agent):
    agent.start("h1", patient_language="hi", domain="clinic")
    agent.add_turn("h1", "doctor", "You have hypertension, take amlodipine 5 mg od.")
    agent.add_turn(
        "h1", "doctor", "If you get chest pain, come immediately to the emergency."
    )
    card = TakeHomeCard(agent).build("h1")
    assert card["sections"][0]["kind"] == "redflag"
    assert "hypertension" in agent.get("h1").glosses
