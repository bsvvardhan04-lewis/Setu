"""Voice activity detection and speech synthesis."""

from __future__ import annotations

import time
import wave

import numpy as np

from ..runtime import Priority
from .base import Adapter, Inference

SAMPLE_RATE = 16000


class Vad(Adapter):
    """Silero VAD - the gate that makes always-on listening affordable."""

    key = "vad"
    priority = Priority.STREAMING

    def __init__(self, cache) -> None:
        super().__init__(cache)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, frame: np.ndarray, threshold: float = 0.5) -> Inference:
        start = time.perf_counter()
        chunk = frame.astype(np.float32)[None, :]

        result = self.cache.run(
            self.key,
            {
                "input": chunk,
                "state": self._state,
                "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            },
            priority=self.priority,
        )
        if result is not None:
            prob = float(np.asarray(result.outputs[0]).reshape(-1)[0])
            if len(result.outputs) > 1:
                self._state = np.asarray(result.outputs[1], dtype=np.float32)
            return Inference(
                prob >= threshold,
                self.key,
                result.device.value,
                result.latency_ms,
                extra={"probability": round(prob, 4)},
            )

        # Energy gate fallback: crude, but it really does gate, and it is honest about it.
        rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))))
        return Inference(
            rms > 0.015,
            self.key,
            "cpu",
            (time.perf_counter() - start) * 1000.0,
            degraded=True,
            extra={"rms": round(rms, 5), "fallback": "energy-gate"},
        )


class Tts(Adapter):
    """Piper VITS synthesis, with Windows SAPI as the offline fallback.

    Speech output is not a nice-to-have here: a user who cannot read the form cannot read
    our explanation of the form either.
    """

    key = "tts"
    priority = Priority.INTERACTIVE

    def synthesize(self, text: str, language: str = "hi") -> Inference:
        start = time.perf_counter()
        phoneme_ids = _naive_phoneme_ids(text)

        result = self.cache.run(
            self.key,
            {
                "input": phoneme_ids[None, :],
                "input_lengths": np.array([len(phoneme_ids)], dtype=np.int64),
                "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
            },
            priority=self.priority,
        )
        if result is not None:
            audio = np.asarray(result.outputs[0], dtype=np.float32).reshape(-1)
            return Inference(
                audio,
                self.key,
                result.device.value,
                result.latency_ms,
                extra={"seconds": round(len(audio) / 22050, 2), "path": "piper"},
            )

        return Inference(
            np.zeros(0, dtype=np.float32),
            self.key,
            "stub",
            (time.perf_counter() - start) * 1000.0,
            degraded=True,
            extra={
                "path": "client-speech-synthesis",
                "note": "UI falls back to the browser SpeechSynthesis API, still on-device",
                "language": language,
            },
        )


def _naive_phoneme_ids(text: str, max_len: int = 512) -> np.ndarray:
    """Placeholder grapheme->id map so the graph shape is right.

    A real deployment swaps this for espeak-ng phonemisation using the id map in
    tts.onnx.json. Kept explicit rather than hidden so nobody mistakes it for finished.
    """
    ids = [1]
    for ch in text[: max_len - 2]:
        ids.append((ord(ch) % 120) + 10)
    ids.append(2)
    return np.array(ids, dtype=np.int64)


def write_wav(path, audio: np.ndarray, sample_rate: int = 22050) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes((pcm * 32767).astype(np.int16).tobytes())
