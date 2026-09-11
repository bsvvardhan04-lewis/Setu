"""VoiceAgent - the always-on listening loop.

Frames arrive at 16 kHz. VAD gates them; only speech reaches Whisper. Silence is
discarded for the cost of one tiny NPU submission, which is the whole reason an
8-hour always-listening session is viable on battery.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..models.speech import SAMPLE_RATE
from .engine import Engine

FRAME_SAMPLES = 512  # 32 ms, the window Silero expects
MAX_UTTERANCE_SECONDS = 20
TRAILING_SILENCE_FRAMES = 20  # ~640 ms of quiet ends an utterance


@dataclass
class Utterance:
    text: str
    language: str
    seconds: float
    devices: dict[str, str] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "language": self.language,
            "seconds": round(self.seconds, 2),
            "devices": self.devices,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "degraded": self.degraded,
        }


@dataclass
class GateStats:
    """Proof that the gate is doing its job, shown live in the UI."""

    frames_seen: int = 0
    frames_with_speech: int = 0
    utterances: int = 0
    vad_ms_total: float = 0.0
    asr_ms_total: float = 0.0

    @property
    def gate_ratio(self) -> float:
        return self.frames_with_speech / self.frames_seen if self.frames_seen else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "frames_seen": self.frames_seen,
            "frames_with_speech": self.frames_with_speech,
            "speech_fraction": round(self.gate_ratio, 4),
            "asr_invocations_avoided": self.frames_seen - self.frames_with_speech,
            "utterances": self.utterances,
            "vad_ms_total": round(self.vad_ms_total, 1),
            "asr_ms_total": round(self.asr_ms_total, 1),
        }


class VoiceAgent:
    def __init__(self, engine: Engine, language: str | None = "en") -> None:
        self.engine = engine
        self.language = language
        self.stats = GateStats()
        self._buffer: list[np.ndarray] = []
        self._silence_run = 0
        self._in_speech = False

    def reset(self) -> None:
        self._buffer.clear()
        self._silence_run = 0
        self._in_speech = False

    def push_frame(self, frame: np.ndarray) -> Utterance | None:
        """Feed one 32 ms frame. Returns an Utterance when one has just ended."""
        self.stats.frames_seen += 1
        gate = self.engine.vad.is_speech(frame)
        self.stats.vad_ms_total += gate.latency_ms

        if gate.value:
            self.stats.frames_with_speech += 1
            self._in_speech = True
            self._silence_run = 0
            self._buffer.append(frame)
        elif self._in_speech:
            self._silence_run += 1
            self._buffer.append(frame)

        buffered_seconds = len(self._buffer) * len(frame) / SAMPLE_RATE
        ended = self._in_speech and (
            self._silence_run >= TRAILING_SILENCE_FRAMES
            or buffered_seconds >= MAX_UTTERANCE_SECONDS
        )
        return self.flush() if ended else None

    def flush(self) -> Utterance | None:
        if not self._buffer:
            self.reset()
            return None
        audio = np.concatenate(self._buffer)
        self.reset()
        return self.transcribe(audio, language=self.language)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Utterance:
        """Transcribe one utterance.

        Passing the language matters more than it looks: Whisper is steered by the
        token seeded into its decoder, and leaving it to auto-detect measurably changes
        the output on the same audio. In a clinic the expert's language is known, so
        say so rather than making the model guess.
        """
        started = time.perf_counter()
        recognised = self.engine.asr.transcribe(audio, language=language)
        self.stats.asr_ms_total += recognised.latency_ms
        self.stats.utterances += 1

        text = recognised.value or ""
        identified = self.engine.lid.identify(text) if text else None

        degraded: list[str] = []
        if recognised.degraded:
            degraded.append("asr:stub")

        return Utterance(
            text=text,
            language=identified.value if identified else "en",
            seconds=len(audio) / SAMPLE_RATE,
            devices={
                "vad": self.engine.vad.card.key if self.engine.vad.card else "vad",
                "asr": recognised.device,
            },
            timings_ms={
                "asr": recognised.latency_ms,
                "total": (time.perf_counter() - started) * 1000.0,
            },
            degraded=degraded,
        )
