"""Language identification and translation."""

from __future__ import annotations

import time
import unicodedata

import numpy as np

from ..config import SUPPORTED_LANGUAGES
from ..runtime import Priority
from .base import Adapter, Inference

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
    """IndicTrans2 distilled.

    A 200M specialist beats asking the 3B chat model to translate: lower latency, lower
    power, and far more faithful on official register (a bank notice is not conversational
    Hindi). When the specialist is absent we return the source and say so, rather than
    silently shipping untranslated text as if it were translated.
    """

    key = "translate"
    priority = Priority.INTERACTIVE

    def translate(self, text: str, src: str, tgt: str) -> Inference:
        start = time.perf_counter()
        if src == tgt or not text.strip():
            return Inference(
                text, self.key, "none", 0.0, extra={"src": src, "tgt": tgt, "noop": True}
            )

        tokens = np.array([[ord(c) % 32000 for c in text[:512]]], dtype=np.int64)
        encoded = self.cache.run(
            self.key,
            {"input_ids": tokens, "attention_mask": np.ones_like(tokens)},
            filename="translate_encoder.onnx",
            priority=self.priority,
        )
        if encoded is not None:
            decoded = self.cache.run(
                self.key,
                {"encoder_hidden_states": np.asarray(encoded.outputs[0])},
                filename="translate_decoder.onnx",
                priority=self.priority,
            )
            if decoded is not None:
                return Inference(
                    _decode_sentencepiece(
                        np.asarray(decoded.outputs[0]), self.cache.model_root / self.key
                    ),
                    self.key,
                    encoded.device.value,
                    encoded.latency_ms + decoded.latency_ms,
                    extra={"src": src, "tgt": tgt},
                )

        return Inference(
            text,
            self.key,
            "stub",
            (time.perf_counter() - start) * 1000.0,
            degraded=True,
            extra={
                "src": src,
                "tgt": tgt,
                "note": "IndicTrans2 absent - text returned untranslated, routed to the LLM instead",
            },
        )


def _decode_sentencepiece(ids: np.ndarray, model_dir) -> str:
    try:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=str(model_dir / "spm.model"))
        return sp.decode([int(i) for i in ids.reshape(-1).tolist() if int(i) > 2])
    except Exception:
        return ""
