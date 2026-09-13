"""End-to-end pipeline behaviour on the fallback path.

These tests deliberately run with no model assets present. That is the configuration a
reviewer meets on first clone, and it must produce a working, honestly-labelled product.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from setu.config import Settings
from setu.models import detect_script
from setu.models.llm import GenParams, TemplateBackend
from setu.pipeline import DocAgent, Engine, FormAgent, VoiceAgent


@pytest.fixture
def engine(tmp_path):
    settings = Settings()
    settings.data_dir = tmp_path
    settings.db_path = tmp_path / "setu.db"
    settings.model_root = tmp_path / "models"
    e = Engine(settings)
    yield e
    e.close()


def test_script_detection_across_indian_languages():
    assert detect_script("नमस्ते दुनिया")[0] == "hi"
    assert detect_script("తెలుగు వచనం")[0] == "te"
    assert detect_script("தமிழ் உரை")[0] == "ta"
    assert detect_script("hello world")[0] == "en"


def test_template_backend_answers_from_context_only():
    prompt = (
        "<CONTEXT>The last date to apply is 31 March 2026. "
        "The office is in Bengaluru.</CONTEXT>\n"
        "<QUESTION>What is the last date to apply?</QUESTION>"
    )
    answer = TemplateBackend().generate(prompt, GenParams())
    assert "31 March 2026" in answer


def test_template_backend_admits_when_context_is_empty():
    answer = TemplateBackend().generate(
        "<CONTEXT></CONTEXT>\n<QUESTION>anything?</QUESTION>", GenParams()
    )
    assert "could not find" in answer.lower()


def test_ingest_and_ask_survive_with_no_models(engine):
    agent = DocAgent(engine)
    page = Image.new("RGB", (600, 800), "white")
    report = agent.ingest([page], title="Blank page")

    assert report.document_id > 0
    assert report.pages == 1
    assert report.degraded, "a stubbed run must declare itself degraded"

    answer = agent.ask("What is the deadline?", document_id=report.document_id)
    assert answer.text
    assert "llm:template" in answer.degraded


def test_answers_carry_device_attribution(engine):
    agent = DocAgent(engine)
    agent.ingest([Image.new("RGB", (400, 400), "white")], title="X")
    answer = agent.ask("hello?")
    assert "llm" in answer.devices and "embed" in answer.devices
    assert "llm" in answer.timings_ms


def test_hashed_embeddings_rank_related_text_higher(engine):
    result = engine.embed.encode(
        [
            "the last date to apply for the pension is 31 March",
            "the office building is painted blue",
            "apply for the pension before the last date in March",
        ]
    )
    vectors = result.value
    assert result.degraded
    related = float(vectors[0] @ vectors[2])
    unrelated = float(vectors[0] @ vectors[1])
    assert related > unrelated


def test_voice_agent_gate_counts_avoided_asr_calls(engine):
    agent = VoiceAgent(engine)
    silence = np.zeros(512, dtype=np.float32)
    for _ in range(30):
        agent.push_frame(silence)
    stats = agent.stats.as_dict()
    assert stats["frames_seen"] == 30
    assert stats["asr_invocations_avoided"] == 30


def test_voice_agent_emits_an_utterance_after_trailing_silence(engine):
    agent = VoiceAgent(engine)
    speech = (np.random.default_rng(1).random(512).astype(np.float32) - 0.5) * 0.8
    silence = np.zeros(512, dtype=np.float32)

    produced = None
    for _ in range(10):
        agent.push_frame(speech)
    for _ in range(25):
        produced = agent.push_frame(silence) or produced
    assert produced is not None
    assert produced.seconds > 0


def test_form_agent_rejects_an_invalid_aadhaar_from_the_model(engine):
    agent = FormAgent(engine)
    agent.start("s1", "pension")
    result = agent.fill("s1", "Ramesh Kumar", field_key="name")
    assert result["field"]["value"] == "Ramesh Kumar"

    bad = agent.fill("s1", "1111 1111 1111", field_key="aadhaar")
    assert bad["field"]["value"] is None
    assert bad["field"]["error"]


def test_form_agent_walks_to_the_next_required_field(engine):
    agent = FormAgent(engine)
    state = agent.start("s2", "grievance")
    assert state.next_empty().key == "name"
    result = agent.fill("s2", "Sita Devi")
    assert result["next_prompt"] == "Mobile number"


def test_system_report_is_complete(engine):
    report = engine.system_report()
    for key in ("host", "power", "router", "sessions", "adapters", "catalogue", "store"):
        assert key in report
    # Every model the router can place must be described to whoever reads the report,
    # and nothing may be described that the router cannot place.
    from setu.runtime.router import DEFAULT_SPECS

    assert {c["key"] for c in report["catalogue"]} == {s.name for s in DEFAULT_SPECS}
