"""Speech synthesis must refuse rather than guess.

Piper voices declare a `phoneme_type`. The Indic voices are `espeak`: they expect IPA
phonemes from espeak-ng, not characters. Feeding a character-derived id sequence to an
espeak voice does not raise - it synthesises confident, fluent-sounding noise.

For a tool whose entire purpose is that a patient understood an instruction, plausible
gibberish spoken aloud is worse than silence. These tests pin down that the adapter would
rather hand the job to the client's own synthesiser than fake it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from setu.config import Settings
from setu.models.speech import _ids_from_map
from setu.pipeline import Engine

REPO = Path(__file__).resolve().parents[1]
VOICE_CONFIG = REPO / "models" / "tts" / "tts.onnx.json"


@pytest.fixture
def engine(tmp_path):
    settings = Settings()
    settings.data_dir = tmp_path
    settings.db_path = tmp_path / "setu.db"
    return Engine(settings)


def test_unknown_symbols_are_dropped_not_guessed():
    """A wrong phoneme id is a wrong sound, and there is no benign wrong sound here."""
    id_map = {"_": [0], "^": [1], "$": [2], "a": [10], "b": [11]}
    ids = _ids_from_map("axb", id_map)
    assert ids is not None
    # 'x' is not in the map and must not contribute anything.
    assert 10 in ids.tolist() and 11 in ids.tolist()
    assert all(int(i) in {0, 1, 2, 10, 11} for i in ids.tolist())


def test_padding_is_interleaved():
    """Piper is trained with a pad id between symbols; dropping it wrecks prosody."""
    id_map = {"_": [0], "^": [1], "$": [2], "a": [10]}
    ids = _ids_from_map("aa", id_map).tolist()
    assert ids == [1, 0, 10, 0, 10, 0, 2]


def test_text_with_no_mappable_symbols_yields_nothing():
    id_map = {"_": [0], "^": [1], "$": [2], "a": [10]}
    assert _ids_from_map("zzz", id_map) is None


def test_synthesis_refuses_when_it_cannot_phonemise(engine):
    result = engine.tts.synthesize("रात को गोली लें", "hi")
    assert result.degraded is True
    assert result.extra["path"] == "client-speech-synthesis"
    assert result.extra.get("reason"), "a refusal must say why"
    assert len(result.value) == 0, "must not emit audio it cannot vouch for"


@pytest.mark.skipif(not VOICE_CONFIG.is_file(), reason="Piper voice not downloaded")
def test_espeak_voice_is_not_driven_from_characters(engine):
    """Regression guard for the specific trap.

    The voice IS on disk here. An adapter that simply checked "is the model present?"
    would happily run it and speak noise. This asserts we check whether we can drive it,
    not merely whether it exists.
    """
    config = json.loads(VOICE_CONFIG.read_text(encoding="utf-8"))
    if config.get("phoneme_type") != "espeak":
        pytest.skip("this voice is not espeak-typed")

    assert engine.tts._phonemiser() is None, (
        "an espeak voice must not be drivable without a phonemiser"
    )
    result = engine.tts.synthesize("Take the tablet at night.", "hi")
    assert result.degraded is True


def test_doctor_does_not_claim_a_capability_we_decline_to_use(engine):
    """`available()` must mean drivable, not merely present on disk.

    Reporting OK for a voice we refuse to run would make the self-check lie to the user
    about what the machine can do - which is the one thing that check exists to prevent.
    """
    if engine.tts._phonemiser() is None:
        assert engine.tts.available() is False
