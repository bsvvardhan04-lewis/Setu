"""Sentence embeddings for the local RAG store.

Indexing runs at BACKGROUND priority: nobody is watching a spinner while we embed page 4,
so Hexa-Router is free to weight milliwatts heavily and put this wherever the machine has
slack. Query embedding runs INTERACTIVE.
"""

from __future__ import annotations

import hashlib
import math
import re
import time

import numpy as np

from ..config import settings
from ..runtime import Priority
from .base import Adapter, Inference

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _hash_embed(text: str, dim: int) -> np.ndarray:
    """Deterministic hashed bag-of-ngrams fallback.

    This is a genuine (if modest) retrieval model, not random noise: it uses hashed word
    and character trigrams with sublinear term weighting, so the demo retrieves sensibly
    even before MiniLM is downloaded. It is clearly labelled ``degraded`` everywhere.
    """
    vec = np.zeros(dim, dtype=np.float32)
    lowered = text.lower()
    words = _TOKEN_RE.findall(lowered)
    grams = words + [lowered[i : i + 3] for i in range(max(0, len(lowered) - 2))]
    counts: dict[str, int] = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    for gram, count in counts.items():
        h = int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")
        idx = h % dim
        sign = 1.0 if (h >> 63) & 1 else -1.0
        vec[idx] += sign * (1.0 + math.log(count))
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class Embedder(Adapter):
    key = "embed"
    priority = Priority.BACKGROUND

    def __init__(self, cache) -> None:
        super().__init__(cache)
        self.dim = settings.embed_dim
        self._tokenizer = None

    def _tokenize(self, texts: list[str], max_len: int = 256):
        if self._tokenizer is None:
            try:
                from tokenizers import Tokenizer

                path = self.cache.model_root / self.key / "tokenizer.json"
                self._tokenizer = Tokenizer.from_file(str(path))
                self._tokenizer.enable_truncation(max_len)
                self._tokenizer.enable_padding(length=max_len)
            except Exception:
                return None
        encoded = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        return ids, mask

    def encode(self, texts: list[str], *, priority: Priority | None = None) -> Inference:
        if not texts:
            return Inference(np.zeros((0, self.dim), np.float32), self.key, "none", 0.0)

        prio = priority or self.priority
        tokenized = self._tokenize(texts)
        if tokenized is not None:
            ids, mask = tokenized
            result = self.cache.run(
                self.key,
                {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "token_type_ids": np.zeros_like(ids),
                },
                priority=prio,
            )
            if result is not None:
                hidden = np.asarray(result.outputs[0], dtype=np.float32)
                # mean-pool over unmasked positions, then L2-normalise
                m = mask[..., None].astype(np.float32)
                pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
                norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
                return Inference(
                    pooled / norms,
                    self.key,
                    result.device.value,
                    result.latency_ms,
                    extra={"n": len(texts)},
                )

        start = time.perf_counter()
        vecs = np.stack([_hash_embed(t, self.dim) for t in texts])
        return Inference(
            vecs,
            self.key,
            "cpu",
            (time.perf_counter() - start) * 1000.0,
            degraded=True,
            extra={"n": len(texts), "fallback": "hashed-ngram"},
        )

    def encode_one(self, text: str, *, priority: Priority | None = None) -> np.ndarray:
        return self.encode([text], priority=priority).value[0]
