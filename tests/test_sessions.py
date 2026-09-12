"""Consultations must survive a restart.

Until this existed a session lived only in memory: restarting the app lost every visit,
and a take-home card URL handed to a patient stopped working the moment the clinic
rebooted. These tests cover that, and the privacy obligations that come with keeping
protected health information on a disk.
"""

from __future__ import annotations

import time

import pytest

from setu.config import Settings
from setu.pipeline import ConsultAgent, Engine, TakeHomeCard
from setu.store import SessionStore

LINES = [
    "You have hypertension, we start amlodipine 5 mg tablet one at night.",
    "Get a lipid profile done, fasting, before the next visit.",
    "If you get chest pain, come immediately to the emergency.",
]


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.data_dir = tmp_path
    s.db_path = tmp_path / "setu.db"
    s.sessions_db = tmp_path / "consultations.db"
    return s


@pytest.fixture
def engine(settings):
    e = Engine(settings)
    yield e
    e.close()


@pytest.fixture
def agent(engine):
    return ConsultAgent(engine, store=engine.sessions)


# ------------------------------------------------------------------ round trip


def test_a_consultation_survives_losing_the_agent(agent, engine):
    agent.start("visit-1", patient_language="hi", domain="clinic")
    for line in LINES:
        agent.add_turn("visit-1", "doctor", line)
    agent.check_teachback("visit-1", "I will take the tablet at night.")
    original = agent.get("visit-1")

    # A new agent over the same store is what a restart looks like.
    revived = ConsultAgent(engine, store=engine.sessions).get("visit-1")

    assert revived.session_id == original.session_id
    assert revived.domain_key == "clinic"
    assert revived.patient_language == "hi"
    assert [t.text for t in revived.turns] == [t.text for t in original.turns]
    assert [p.text for p in revived.plan] == [p.text for p in original.plan]
    assert revived.glosses == original.glosses


def test_teachback_results_survive_too(agent, engine):
    """The confirmed/missed state is the clinically meaningful part; losing it would
    leave a card that no longer knows what the patient failed to repeat."""
    agent.start("visit-2", domain="clinic")
    for line in LINES:
        agent.add_turn("visit-2", "doctor", line)
    agent.check_teachback("visit-2", "I will take the tablet at night.")
    before = [(p.text, p.confirmed) for p in agent.get("visit-2").plan]

    revived = ConsultAgent(engine, store=engine.sessions).get("visit-2")
    assert [(p.text, p.confirmed) for p in revived.plan] == before


def test_a_card_can_still_be_built_after_a_restart(agent, engine):
    """The patient was handed a URL. It has to keep working."""
    agent.start("visit-3", patient_language="hi", domain="clinic")
    for line in LINES:
        agent.add_turn("visit-3", "doctor", line)

    fresh = ConsultAgent(engine, store=engine.sessions)
    card = TakeHomeCard(fresh).build("visit-3")
    assert card["sections"], "a restored visit must still produce a card"
    assert card["sections"][0]["kind"] == "redflag"


def test_unknown_session_still_raises(agent):
    with pytest.raises(KeyError):
        agent.get("never-existed")


# -------------------------------------------------------------------- privacy


def test_deletion_removes_everything(agent, engine):
    agent.start("visit-4", domain="clinic")
    for line in LINES:
        agent.add_turn("visit-4", "doctor", line)

    assert agent.forget("visit-4") is True
    with pytest.raises(KeyError):
        ConsultAgent(engine, store=engine.sessions).get("visit-4")
    assert engine.sessions.stats()["turns"] == 0, "turns must not outlive their consultation"


def test_retention_prunes_old_consultations(agent, engine):
    agent.start("old", domain="clinic")
    agent.add_turn("old", "doctor", LINES[0])

    # Backdate it past any plausible retention window.
    with engine.sessions._lock:
        engine.sessions._conn.execute(
            "UPDATE consultations SET updated_at = ? WHERE session_id = ?",
            (time.time() - 400 * 86400, "old"),
        )
        engine.sessions._conn.commit()

    assert engine.sessions.prune(90) == 1
    assert engine.sessions.stats()["consultations"] == 0


def test_retention_of_zero_keeps_everything(agent, engine):
    """Unbounded retention has to be a deliberate choice, not an accident."""
    agent.start("keep", domain="clinic")
    assert engine.sessions.prune(0) == 0
    assert engine.sessions.stats()["consultations"] == 1


def test_recent_lists_newest_first(agent):
    for name in ("a", "b", "c"):
        agent.start(name, domain="clinic")
        agent.add_turn(name, "doctor", LINES[0])
        time.sleep(0.01)
    listed = [row["session_id"] for row in agent.recent()]
    assert listed[0] == "c"
    assert set(listed) == {"a", "b", "c"}


# ------------------------------------------------------------------- optional


def test_the_store_is_optional(engine):
    """Without a store the agent behaves exactly as before - so tests and throwaway runs
    do not leave patient data on disk."""
    agent = ConsultAgent(engine)  # no store
    agent.start("ephemeral", domain="clinic")
    agent.add_turn("ephemeral", "doctor", LINES[0])
    assert agent.recent() == []
    assert agent.forget("ephemeral") is True


def test_store_survives_reopening_the_file(tmp_path):
    path = tmp_path / "c.db"
    store = SessionStore(path)
    store.close()
    reopened = SessionStore(path)
    assert reopened.stats() == {"consultations": 0, "turns": 0}
    reopened.close()
