"""Translation, and the language-token handling that decides whether it works at all.

NLLB is steered by two special tokens - a source-language token prepended to the encoder
input, and a target-language token forced as the decoder's first generated token. Getting
the target token wrong does not raise: the model translates into whatever it likes, usually
the source language, so the output looks like a passthrough bug rather than a prompt bug.
These tests pin both down.

The heavier round-trip tests skip without the assets, so a bare checkout stays green.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from setu.config import Settings
from setu.models.translate_seq2seq import TARGET_CODES, MarianCodec
from setu.pipeline import Engine

REPO = Path(__file__).resolve().parents[1]
TRANSLATE_DIR = REPO / "models" / "translate"
TOKENIZER = TRANSLATE_DIR / "tokenizer.json"
#: A seq2seq model is only usable when BOTH halves are on disk. Gating on the encoder
#: alone makes a half-finished download look like a translation bug.
REQUIRED = ("translate_encoder.onnx", "translate_decoder.onnx", "tokenizer.json", "config.json")

needs_translate = pytest.mark.skipif(
    not all((TRANSLATE_DIR / f).is_file() for f in REQUIRED),
    reason="OPUS-MT assets incomplete; run scripts/fetch_models.py --model translate",
)
needs_tokenizer = pytest.mark.skipif(
    not TOKENIZER.is_file(), reason="NLLB tokenizer absent"
)


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    settings = Settings()
    tmp = tmp_path_factory.mktemp("xl")
    settings.data_dir = tmp
    settings.db_path = tmp / "setu.db"
    e = Engine(settings)
    yield e
    e.close()


def test_every_shipped_language_has_an_nllb_code():
    from setu.config import SUPPORTED_LANGUAGES

    missing = set(SUPPORTED_LANGUAGES) - set(TARGET_CODES)
    assert not missing, f"no NLLB code for {missing}"


def test_target_codes_are_iso_639_3():
    # OPUS-MT expects three-letter tags in a >>xxx<< prefix. A two-letter code is not
    # rejected - the model just picks a target language itself, usually echoing the
    # source, which looks like a passthrough bug rather than a prompt bug.
    for code in TARGET_CODES.values():
        assert len(code) == 3 and code.islower(), code


@needs_tokenizer
def test_source_is_prefixed_with_the_target_tag_and_ends_in_eos():
    codec = MarianCodec(TRANSLATE_DIR)
    ids = codec.encode_source("Take the tablet at night.", "hi")
    assert ids is not None
    assert ids[0][-1] == codec.eos_id, "Marian expects a trailing EOS"
    # The >>hin<< prefix must survive tokenisation as leading tokens, not be dropped.
    plain = codec.encode_source("Take the tablet at night.", "en")
    assert not np.array_equal(ids, plain), "target tag must change the encoded source"


@needs_tokenizer
def test_special_ids_come_from_the_checkpoint():
    codec = MarianCodec(TRANSLATE_DIR)
    assert isinstance(codec.eos_id, int)
    assert isinstance(codec.decoder_start_id, int)


def test_same_language_is_a_no_op(engine):
    result = engine.translate.translate("Take the tablet.", "en", "en")
    assert result.value == "Take the tablet."
    assert result.extra["noop"] is True


def test_missing_assets_return_source_marked_degraded(engine, monkeypatch):
    """The most dangerous silent failure in the project would be returning untranslated
    text as though it were translated."""
    monkeypatch.setattr(engine.translate, "supported", lambda code: False)
    result = engine.translate.translate("Take the tablet at night.", "en", "hi")
    assert result.degraded is True
    assert result.value == "Take the tablet at night."


@needs_translate
def test_translates_english_to_hindi(engine):
    result = engine.translate.translate(
        "Take the tablet at night. Come back after two weeks.", "en", "hi"
    )
    assert not result.degraded, "assets present, so this must be a real inference"
    # Devanagari, and not a passthrough of the source.
    assert any("\u0900" <= ch <= "\u097f" for ch in result.value), result.value
    assert "tablet" not in result.value.lower()


@needs_translate
def test_translates_english_to_telugu(engine):
    result = engine.translate.translate("Submit the assignment by Friday.", "en", "te")
    assert not result.degraded
    assert any("\u0c00" <= ch <= "\u0c7f" for ch in result.value), result.value
