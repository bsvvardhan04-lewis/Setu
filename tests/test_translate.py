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

import pytest

from setu.config import Settings
from setu.models.translate_seq2seq import (
    DEDICATED_MODELS,
    SUPPORTED_TARGETS,
    TARGET_CODES,
    MarianCodec,
)
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


def test_translatable_set_is_a_subset_of_shipped_languages():
    from setu.config import SUPPORTED_LANGUAGES

    stray = SUPPORTED_TARGETS - set(SUPPORTED_LANGUAGES)
    assert not stray, f"translating into languages the app does not ship: {stray}"


def test_languages_without_a_working_checkpoint_are_excluded():
    """Marathi, Bengali and Punjabi are deliberately absent.

    The multilingual checkpoint accepts >>mar<<, >>ben<< and >>pan<< and then returns
    Devanagari word salad or echoes the source unchanged. That failure passes any "is
    this the right script?" check, so it has to be excluded by hand rather than detected.
    Listing them would mean a demo discovers it live.
    """
    for code in ("mr", "bn", "pa"):
        assert code not in SUPPORTED_TARGETS


def test_hindi_is_served_by_a_dedicated_checkpoint():
    # Hindi is the flagship demo language and the multilingual model cannot do it.
    assert DEDICATED_MODELS["hi"] == "translate_hi"
    assert "hi" in SUPPORTED_TARGETS
    assert "hi" not in TARGET_CODES, "must not also route through the multilingual model"


def test_every_translation_checkpoint_can_be_fetched_and_is_catalogued():
    """A checkpoint the code routes to must be downloadable from a fresh clone.

    `translate_hi` once existed only on the machine it was developed on: the router and
    the translator both used it, but `fetch_models.py` had no recipe for it, so anyone
    else got an English passthrough for Hindi - the flagship language - with no error.
    """
    import importlib.util

    from setu.models.registry import CATALOGUE

    script = REPO / "scripts" / "fetch_models.py"
    spec = importlib.util.spec_from_file_location("fetch_models", script)
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)

    for directory in ("translate", *DEDICATED_MODELS.values()):
        recipe = fetch.HF_RECIPES.get(directory)
        assert recipe, f"no fetch recipe for models/{directory}"
        assert {"translate_encoder.onnx", "translate_decoder.onnx", "tokenizer.json"} <= set(
            recipe["files"].values()
        )
        assert directory in CATALOGUE, f"models/{directory} missing from the catalogue"

    # The catalogue is what /api/system shows a judge; it must not claim bn/mr/pa.
    catalogued = set(CATALOGUE["translate"].languages)
    for lang, directory in DEDICATED_MODELS.items():
        assert lang in CATALOGUE[directory].languages
        catalogued.add(lang)
    assert catalogued - {"en"} == set(SUPPORTED_TARGETS)


def test_target_codes_are_iso_639_3():
    # OPUS-MT expects three-letter tags in a >>xxx<< prefix. A two-letter code is not
    # rejected - the model just picks a target language itself, usually echoing the
    # source, which looks like a passthrough bug rather than a prompt bug.
    for code in TARGET_CODES.values():
        assert len(code) == 3 and code.islower(), code


def test_sentences_are_split_before_translation():
    """Marian is a sentence-level model; handing it two sentences loses one."""
    from setu.models.translate_seq2seq import split_sentences

    assert split_sentences("Take the tablet at night. Come back after two weeks.") == [
        "Take the tablet at night.",
        "Come back after two weeks.",
    ]
    # A Devanagari danda ends a sentence too.
    assert len(split_sentences("पहला वाक्य। दूसरा वाक्य।")) == 2
    assert split_sentences("no terminator") == ["no terminator"]


def test_repetition_guard_blocks_a_repeated_ngram():
    """Without this, greedy decoding collapsed into a loop that was both wrong and slow."""
    from setu.models.translate_seq2seq import _repeat_banned

    assert _repeat_banned([1, 2]) == set()
    # 5,6 already led to 7, so 7 is banned when 5,6 recurs.
    assert 7 in _repeat_banned([5, 6, 7, 9, 5, 6])
    assert _repeat_banned([1, 2, 3, 4, 5]) == set()


@needs_tokenizer
def test_multilingual_source_carries_the_target_tag_id():
    codec = MarianCodec(TRANSLATE_DIR)
    tamil = codec.encode_source("Take the tablet at night.", "ta")
    assert tamil is not None
    assert tamil[0][0] == codec.target_token_id("ta"), "tag must be the first id"
    assert tamil[0][-1] == codec.eos_id

    telugu = codec.encode_source("Take the tablet at night.", "te")
    assert telugu[0][0] != tamil[0][0], "different targets must differ in the tag"


@needs_tokenizer
def test_bilingual_checkpoint_takes_no_language_tag():
    """A dedicated en->hi model has exactly one target, so a tag would be noise."""
    codec = MarianCodec(TRANSLATE_DIR, bilingual=True)
    ids = codec.encode_source("Take the tablet at night.", "hi")
    assert ids is not None
    assert ids[0][-1] == codec.eos_id


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
