"""Encoder-decoder translation (Marian / OPUS-MT).

Separated from `lang.py` because the decoding loop is real work, not a wrapper.

Marian steers the target language by **prefixing the source text with a token** like
`>>hin<<`, rather than forcing a token on the decoder side the way NLLB does. Omit it and
the model does not error - it simply picks a target language on its own, usually echoing
the source, which reads like a passthrough bug rather than a prompt bug. That failure mode
is the reason this module exists as its own tested unit.

Special-token ids come from the model's own `config.json` rather than being hard-coded,
because Marian checkpoints differ and a wrong `decoder_start_token_id` produces fluent,
confident, entirely wrong output.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np

log = logging.getLogger(__name__)

#: SETU language code -> the ISO-639-3 tag OPUS-MT's multilingual model expects.
#:
#: Only the languages the multilingual checkpoint ACTUALLY translates are listed. It
#: nominally accepts >>hin<<, >>mar<<, >>ben<< and >>pan<< and then produces Devanagari
#: word salad or echoes the source unchanged - a failure that passes any "is this the
#: right script?" check and has to be read to be caught. Those languages are served by
#: dedicated bilingual models instead; see DEDICATED_MODELS.
TARGET_CODES: dict[str, str] = {
    "ta": "tam",
    "te": "tel",
    "kn": "kan",
    "ml": "mal",
    "gu": "guj",
    "or": "ori",
    "ur": "urd",
}

#: Languages that need their own bilingual checkpoint, mapped to the directory under
#: `models/` holding it. Hindi is the flagship demo language, so it gets one.
DEDICATED_MODELS: dict[str, str] = {
    "hi": "translate_hi",
}

#: Everything SETU can translate into, however it gets there.
SUPPORTED_TARGETS: frozenset[str] = frozenset(TARGET_CODES) | frozenset(DEDICATED_MODELS)

MAX_NEW_TOKENS = 160
MAX_SOURCE_TOKENS = 220


class MarianCodec:
    """Tokenisation plus the special ids, read from the model's own config."""

    def __init__(self, model_dir, bilingual: bool = False) -> None:
        self.model_dir = model_dir
        self.bilingual = bilingual
        self._tokenizer = None
        self._config: dict | None = None
        self._loaded = False

    @property
    def tokenizer(self):
        if not self._loaded:
            self._loaded = True
            try:
                from tokenizers import Tokenizer

                self._tokenizer = Tokenizer.from_str(self._tokenizer_json())
            except Exception as exc:
                log.debug("Marian tokenizer unavailable: %s", exc)
                self._tokenizer = None
        return self._tokenizer

    def _tokenizer_json(self) -> str:
        """Load tokenizer.json, repairing a null Precompiled normaliser.

        The published Marian tokenisers carry
        ``{"type": "Precompiled", "precompiled_charsmap": null}``. The JavaScript
        tokeniser treats a null charsmap as the identity transform, so exporting it that
        way is harmless there - but the Rust ``tokenizers`` crate deserialises it
        strictly and *panics* the interpreter with "invalid type: null, expected a
        borrowed string". A panic from Rust is not a Python exception, so this cannot be
        caught around ``Tokenizer.from_file``; it has to be prevented.

        Dropping the no-op normaliser is the repair. Done in memory, so re-downloading
        the model does not silently reintroduce the crash.
        """
        raw = json.loads((self.model_dir / "tokenizer.json").read_text(encoding="utf-8"))
        normalizer = raw.get("normalizer")
        if (
            isinstance(normalizer, dict)
            and normalizer.get("type") == "Precompiled"
            and normalizer.get("precompiled_charsmap") is None
        ):
            raw["normalizer"] = None
        return json.dumps(raw)

    @property
    def config(self) -> dict:
        if self._config is None:
            try:
                self._config = json.loads(
                    (self.model_dir / "config.json").read_text(encoding="utf-8")
                )
            except Exception:
                self._config = {}
        return self._config

    @property
    def eos_id(self) -> int:
        return int(self.config.get("eos_token_id", 0))

    @property
    def decoder_start_id(self) -> int:
        """Marian starts the decoder on the PAD token, not on BOS.

        Getting this wrong yields fluent, confident, entirely wrong output - so read it
        from the checkpoint rather than assuming.
        """
        value = self.config.get("decoder_start_token_id")
        if value is None:
            value = self.config.get("pad_token_id", 0)
        return int(value)

    def supports(self, code: str) -> bool:
        return code in TARGET_CODES

    def target_token_id(self, tgt: str) -> int | None:
        """The vocab id of the `>>xxx<<` language tag."""
        tokenizer = self.tokenizer
        tag = TARGET_CODES.get(tgt)
        if tokenizer is None or tag is None:
            return None
        found = tokenizer.token_to_id(f">>{tag}<<")
        return int(found) if found is not None else None

    def encode_source(self, text: str, tgt: str) -> np.ndarray | None:
        """[>>tgt<<] source tokens [EOS] - the form Marian was trained on.

        The language tag is prepended as an ID, not as text. Encoding the literal
        string ">>hin<< ..." looks right and is wrong: the Unigram model does not treat
        the tag as atomic and shreds it into ['▁>', '>', 'hin', '<', '<'], so the model
        never sees the tag, picks a target language on its own, and returns fluent
        nonsense in a language nobody asked for - with the mangled tag echoed back in
        the output. Nothing raises.
        """
        tokenizer = self.tokenizer
        if tokenizer is None:
            return None
        ids = tokenizer.encode(text, add_special_tokens=False).ids[:MAX_SOURCE_TOKENS]

        if self.bilingual:
            # A bilingual checkpoint has exactly one target and takes no tag.
            return np.array([[*ids, self.eos_id]], dtype=np.int64)

        tag_id = self.target_token_id(tgt)
        if tag_id is None:
            return None
        return np.array([[tag_id, *ids, self.eos_id]], dtype=np.int64)

    def decode(self, ids: list[int]) -> str:
        tokenizer = self.tokenizer
        if tokenizer is None:
            return ""
        return tokenizer.decode(ids, skip_special_tokens=True).strip()


NO_REPEAT_NGRAM = 3


def _repeat_banned(tokens: list[int], n: int = NO_REPEAT_NGRAM) -> set[int]:
    """Tokens that would complete an n-gram already present in `tokens`.

    The standard no-repeat-ngram constraint. Cheap to compute at these lengths, and it
    is the difference between a translation and a stuck record.
    """
    if len(tokens) < n:
        return set()
    prefix = tuple(tokens[-(n - 1) :])
    return {
        tokens[i + n - 1]
        for i in range(len(tokens) - n + 1)
        if tuple(tokens[i : i + n - 1]) == prefix
    }


SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def split_sentences(text: str) -> list[str]:
    """Marian is a SENTENCE-level model, not a document-level one.

    Handing it two sentences at once does not raise - it translates one and silently
    drops the other, or rambles past the end of the first. Splitting first is what turns
    "come back after two weeks" into the full instruction the patient was actually given.
    """
    parts = [p.strip() for p in SENTENCE_SPLIT.split(text) if p.strip()]
    return parts or [text.strip()]


def greedy_translate(
    cache, codec: MarianCodec, text: str, src: str, tgt: str, priority,
    model_key: str = "translate",
):
    """Translate sentence by sentence, then rejoin."""
    pieces: list[str] = []
    device = "cpu"
    latency = 0.0
    for sentence in split_sentences(text):
        result = _translate_one(cache, codec, sentence, tgt, priority, model_key)
        if result is None:
            return None
        translated, device, took = result
        latency += took
        if translated:
            pieces.append(translated)
    return " ".join(pieces), device, latency


def _translate_one(
    cache, codec: MarianCodec, text: str, tgt: str, priority, model_key: str
):
    """Encode once, then decode token by token until end-of-sentence.

    Returns (text, device, latency_ms), or None when the assets are not loaded. No KV
    cache, for the same reason as the Whisper path: correctness on any machine beats
    throughput here, and Snapdragon replaces this graph entirely.
    """
    source = codec.encode_source(text, tgt)
    if source is None:
        return None

    encoded = cache.run(
        model_key,
        {"input_ids": source, "attention_mask": np.ones_like(source)},
        filename="translate_encoder.onnx",
        priority=priority,
    )
    if encoded is None:
        return None

    encoder_states = np.asarray(encoded.outputs[0], dtype=np.float32)
    encoder_mask = np.ones(encoder_states.shape[:2], dtype=np.int64)

    eos = codec.eos_id
    tokens = [codec.decoder_start_id]
    latency = encoded.latency_ms
    device = encoded.device.value

    for _ in range(MAX_NEW_TOKENS):
        decoded = cache.run(
            model_key,
            {
                "input_ids": np.array([tokens], dtype=np.int64),
                "encoder_attention_mask": encoder_mask,
                "encoder_hidden_states": encoder_states,
            },
            filename="translate_decoder.onnx",
            priority=priority,
        )
        if decoded is None:
            return None
        latency += decoded.latency_ms
        logits = np.asarray(decoded.outputs[0])[0, -1].astype(np.float32)

        # Greedy decoding without a repetition guard collapses into a loop: observed
        # output was "రెండు రెండు రెండు..." repeated until the token budget ran out,
        # which is both wrong AND the reason a short sentence took 17 seconds. Blocking
        # any token that would complete a 3-gram we have already emitted fixes the
        # quality and the latency in one move.
        for banned in _repeat_banned(tokens):
            logits[banned] = -np.inf

        next_token = int(logits.argmax())
        if next_token == eos:
            break
        tokens.append(next_token)

    return codec.decode(tokens[1:]), device, latency
