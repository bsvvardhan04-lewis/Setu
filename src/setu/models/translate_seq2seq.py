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

import numpy as np

log = logging.getLogger(__name__)

#: SETU language code -> the ISO-639-3 tag OPUS-MT's multilingual model expects.
TARGET_CODES: dict[str, str] = {
    "en": "eng",
    "hi": "hin",
    "te": "tel",
    "ta": "tam",
    "bn": "ben",
    "mr": "mar",
    "kn": "kan",
    "ml": "mal",
    "gu": "guj",
    "pa": "pan",
    "or": "ori",
    "ur": "urd",
}

MAX_NEW_TOKENS = 160
MAX_SOURCE_TOKENS = 220


class MarianCodec:
    """Tokenisation plus the special ids, read from the model's own config."""

    def __init__(self, model_dir) -> None:
        self.model_dir = model_dir
        self._tokenizer = None
        self._config: dict | None = None
        self._loaded = False

    @property
    def tokenizer(self):
        if not self._loaded:
            self._loaded = True
            try:
                from tokenizers import Tokenizer

                self._tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            except Exception as exc:
                log.debug("Marian tokenizer unavailable: %s", exc)
                self._tokenizer = None
        return self._tokenizer

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

    def encode_source(self, text: str, tgt: str) -> np.ndarray | None:
        """`>>tgt<< source text` then EOS - the form Marian was trained on."""
        tokenizer = self.tokenizer
        tag = TARGET_CODES.get(tgt)
        if tokenizer is None or tag is None:
            return None
        ids = tokenizer.encode(f">>{tag}<< {text}", add_special_tokens=False).ids
        ids = ids[:MAX_SOURCE_TOKENS]
        return np.array([[*ids, self.eos_id]], dtype=np.int64)

    def decode(self, ids: list[int]) -> str:
        tokenizer = self.tokenizer
        if tokenizer is None:
            return ""
        return tokenizer.decode(ids, skip_special_tokens=True).strip()


def greedy_translate(cache, codec: MarianCodec, text: str, src: str, tgt: str, priority):
    """Encode once, then decode token by token until end-of-sentence.

    Returns (text, device, latency_ms), or None when the assets are not loaded. No KV
    cache, for the same reason as the Whisper path: correctness on any machine beats
    throughput here, and Snapdragon replaces this graph entirely.
    """
    source = codec.encode_source(text, tgt)
    if source is None:
        return None

    encoded = cache.run(
        "translate",
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
            "translate",
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
        next_token = int(np.asarray(decoded.outputs[0])[0, -1].argmax())
        if next_token == eos:
            break
        tokens.append(next_token)

    return codec.decode(tokens[1:]), device, latency
