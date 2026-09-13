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
import platform
from pathlib import Path

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


def test_doctor_reports_drivability_not_mere_presence(engine):
    """`available()` must mean "some real engine can speak", not "a file exists".

    Reporting OK for a voice we refuse to run would make the self-check lie about what
    the machine can do. Reporting MISSING while the OS can speak perfectly well would be
    the same lie in the other direction. So it tracks whether ANY real path exists.
    """
    piper_drivable = engine.tts._phonemiser() is not None
    system_voice = engine.tts._sapi_voice("en") is not None
    assert engine.tts.available() is (piper_drivable or system_voice)


# ------------------------------------------------------- the OS synthesis engine
#
# Same move as document capture: when our own model cannot be driven, use the engine the
# operating system already ships rather than reporting the capability as absent. SAPI is
# real, fully offline synthesis - it is just not ours, and it covers only the languages
# whose voices are installed. Both facts are reported rather than glossed over.


@pytest.mark.skipif(platform.system() != "Windows", reason="SAPI is a Windows engine")
def test_english_synthesises_through_the_system_voice(engine):
    if engine.tts._sapi_voice("en") is None:
        pytest.skip("no English system voice installed")

    result = engine.tts.synthesize("Take the tablet at night.", "en")
    assert result.degraded is False, "SAPI is real synthesis, not a degraded path"
    assert result.extra["path"] == "windows-sapi"
    assert result.extra["npu_accelerated"] is False
    assert len(result.value) > 0
    assert result.extra["seconds"] > 0.3


@pytest.mark.skipif(platform.system() != "Windows", reason="SAPI is a Windows engine")
def test_a_language_with_no_installed_voice_still_refuses(engine):
    """Coverage is per-language, and the refusal has to be per-language too.

    An English voice being present says nothing about Hindi. Falling back to it would
    speak Hindi text with an English voice - which is not a translation failure the user
    can see, it is confident mispronunciation.
    """
    missing = next(
        (code for code in ("hi", "te", "ta", "kn") if engine.tts._sapi_voice(code) is None),
        None,
    )
    if missing is None:
        pytest.skip("every test language has a system voice installed")

    result = engine.tts.synthesize("रात को गोली लें", missing)
    assert result.degraded is True
    assert result.extra["path"] == "client-speech-synthesis"
    assert len(result.value) == 0


def test_powershell_literals_are_escaped():
    """Care-plan text is interpolated into a PowerShell command line.

    A quote inside a clinical instruction - "doctor's note" - must not be able to end the
    string literal and let the rest be read as commands. PowerShell escapes a single quote
    by doubling it, so the invariant is: the result is wrapped in single quotes, and every
    quote inside is doubled.
    """
    from setu.models.speech import _ps_quote

    assert _ps_quote("take it") == "'take it'"
    assert _ps_quote("doctor's note") == "'doctor''s note'"

    hostile = r"x'; Remove-Item C:\ -Recurse; '"
    quoted = _ps_quote(hostile)
    assert quoted.startswith("'") and quoted.endswith("'")
    # Strip the wrapper; every remaining quote must be part of a doubled pair.
    inner = quoted[1:-1]
    assert inner.replace("''", "") .count("'") == 0, (
        f"an unpaired quote survived escaping: {quoted!r}"
    )
