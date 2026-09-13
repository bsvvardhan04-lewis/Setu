"""Language identification and translation."""

from __future__ import annotations

import threading
import time
import unicodedata

from ..config import SUPPORTED_LANGUAGES
from ..runtime import Priority
from .base import Adapter, Inference
from .translate_seq2seq import (
    DEDICATED_MODELS,
    SUPPORTED_TARGETS,
    MarianCodec,
    greedy_translate,
)

#: Unicode block -> language, used by the script-based identifier. Several languages share
#: a script (Hindi and Marathi are both Devanagari), so this narrows rather than decides;
#: the model, when present, disambiguates.
_SCRIPT_HINTS: list[tuple[str, str]] = [
    ("DEVANAGARI", "hi"),
    ("TELUGU", "te"),
    ("TAMIL", "ta"),
    ("BENGALI", "bn"),
    ("KANNADA", "kn"),
    ("MALAYALAM", "ml"),
    ("GUJARATI", "gu"),
    ("GURMUKHI", "pa"),
    ("ORIYA", "or"),
    ("ARABIC", "ur"),
    ("LATIN", "en"),
]


def detect_script(text: str) -> tuple[str, float]:
    """Identify language by dominant Unicode script. Cheap, offline, and correct for the
    overwhelming majority of Indian document text."""
    counts: dict[str, int] = {}
    total = 0
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        total += 1
        for block, lang in _SCRIPT_HINTS:
            if name.startswith(block):
                counts[lang] = counts.get(lang, 0) + 1
                break
    if not counts or total == 0:
        return "en", 0.0
    lang, hits = max(counts.items(), key=lambda kv: kv[1])
    return lang, hits / total


class LanguageId(Adapter):
    key = "lid"
    needs_asset = False  # Unicode-script scan, no weights
    priority = Priority.INTERACTIVE

    def identify(self, text: str) -> Inference:
        start = time.perf_counter()
        lang, confidence = detect_script(text)
        return Inference(
            lang,
            self.key,
            "cpu",
            (time.perf_counter() - start) * 1000.0,
            extra={
                "confidence": round(confidence, 3),
                "display": SUPPORTED_LANGUAGES.get(lang, lang),
                "method": "unicode-script",
            },
        )


class Translator(Adapter):
    """OPUS-MT multilingual, covering every shipped target language in one 111 MB model.

    A specialist beats asking the 3B chat model to translate: lower latency, far lower
    power, and more faithful on official and clinical register. A discharge summary is
    not conversational Hindi, and a general chat model tends to smooth it into something
    friendlier and less exact. For a care plan, exactness is the product.

    When the assets are absent the source text comes back marked `degraded`, and the UI
    says so. Returning untranslated text as though it were translated would be the single
    most dangerous silent failure in this project.
    """

    key = "translate"
    primary_asset = "translate_encoder.onnx"
    priority = Priority.INTERACTIVE

    def __init__(self, cache) -> None:
        super().__init__(cache)
        # A consultation repeats itself: the same instruction is translated for the live
        # transcript and again for the take-home card. Memoising makes the second one
        # free, which is most of why re-opening a card is instant.
        self._memo: dict[tuple[str, str], Inference] = {}
        self._memo_lock = threading.Lock()
        # The multilingual checkpoint, plus one bilingual checkpoint per language it
        # cannot actually handle. Built lazily so an absent dedicated model costs nothing.
        self._codecs: dict[str, MarianCodec] = {
            self.key: MarianCodec(cache.model_root / self.key)
        }
        for directory in DEDICATED_MODELS.values():
            self._codecs[directory] = MarianCodec(
                cache.model_root / directory, bilingual=True
            )

    def supported(self, code: str) -> bool:
        return code in SUPPORTED_TARGETS

    def _route(self, tgt: str) -> tuple[str, MarianCodec]:
        """Which checkpoint handles this target language."""
        key = DEDICATED_MODELS.get(tgt, self.key)
        return key, self._codecs[key]

    def translate(self, text: str, src: str, tgt: str) -> Inference:
        start = time.perf_counter()
        if not text.strip() or src == tgt:
            return Inference(
                text, self.key, "none", 0.0, extra={"src": src, "tgt": tgt, "noop": True}
            )

        # "auto" reaches us from the teach-back path, which knows the text is not English
        # but not which language it is. Script detection is enough to pick a code.
        if src == "auto":
            src = detect_script(text)[0]

        key = (text, tgt)
        with self._memo_lock:
            cached = self._memo.get(key)
        if cached is not None:
            return cached

        # Every checkpoint SETU ships translates FROM English. The source therefore has
        # to be English, and only the TARGET is looked up in the supported set - asking
        # whether the source is a supported target is a different question with the same
        # shape, and answering it disabled translation entirely.
        if src == "en" and self.supported(tgt):
            model_key, codec = self._route(tgt)
            result = greedy_translate(
                self.cache, codec, text, src, tgt, self.priority, model_key=model_key
            )
            if result is not None:
                translated, device, latency = result
                if translated:
                    inference = Inference(
                        translated,
                        self.key,
                        device,
                        latency,
                        extra={"src": src, "tgt": tgt, "checkpoint": model_key},
                    )
                    with self._memo_lock:
                        self._memo[key] = inference
                    return inference

        return Inference(
            text,
            self.key,
            "stub",
            (time.perf_counter() - start) * 1000.0,
            degraded=True,
            extra={
                "src": src,
                "tgt": tgt,
                "note": "translation model not loaded - showing the source text unchanged",
            },
        )
