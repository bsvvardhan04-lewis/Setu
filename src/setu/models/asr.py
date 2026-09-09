"""Speech recognition (Whisper) with a VAD gate in front of it.

The gate matters more than the model here. Always-on ASR is the workload that justifies
an NPU, but only if you are not decoding 30-second windows of silence all day. Silero VAD
costs microseconds and cuts Whisper invocations by an order of magnitude in a quiet room.
"""

from __future__ import annotations

import time

import numpy as np

from ..runtime import Priority
from .base import Adapter, Inference

SAMPLE_RATE = 16000
N_MELS = 80
CHUNK_SECONDS = 30


def log_mel_spectrogram(audio: np.ndarray, n_mels: int = N_MELS) -> np.ndarray:
    """Whisper-compatible log-Mel features, numpy-only.

    Implemented here rather than pulled from torchaudio because the ARM64 wheel story for
    torch is unpleasant and this is 40 lines of FFT.
    """
    n_fft, hop = 400, 160
    audio = audio.astype(np.float32)
    target = SAMPLE_RATE * CHUNK_SECONDS
    audio = np.pad(audio, (0, max(0, target - len(audio))))[:target]

    window = np.hanning(n_fft).astype(np.float32)
    frames = 1 + (len(audio) - n_fft) // hop
    stft = np.empty((frames, n_fft // 2 + 1), dtype=np.complex64)
    for i in range(frames):
        segment = audio[i * hop : i * hop + n_fft] * window
        stft[i] = np.fft.rfft(segment)
    power = (np.abs(stft) ** 2).T

    mel_fb = _mel_filterbank(n_mels, n_fft, SAMPLE_RATE)
    mel = mel_fb @ power
    log_spec = np.log10(np.clip(mel, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    n_bins = n_fft // 2 + 1
    mel_points = np.linspace(_hz_to_mel(np.array(0.0)), _hz_to_mel(np.array(sr / 2)), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, min(centre, n_bins)):
            if centre > left:
                fb[m - 1, k] = (k - left) / (centre - left)
        for k in range(centre, min(right, n_bins)):
            if right > centre:
                fb[m - 1, k] = (right - k) / (right - centre)
    return fb


class Asr(Adapter):
    key = "asr"
    priority = Priority.STREAMING

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Inference:
        start = time.perf_counter()
        mel = log_mel_spectrogram(audio)[None, ...]

        encoded = self.cache.run(
            self.key, {"mel": mel}, filename="asr_encoder.onnx", priority=self.priority
        )
        if encoded is not None:
            decoded = self.cache.run(
                self.key,
                {"encoder_hidden_states": np.asarray(encoded.outputs[0])},
                filename="asr_decoder.onnx",
                priority=self.priority,
            )
            if decoded is not None:
                text = self._detokenize(np.asarray(decoded.outputs[0]))
                total = encoded.latency_ms + decoded.latency_ms
                return Inference(
                    text,
                    self.key,
                    encoded.device.value,
                    total,
                    extra={
                        "audio_seconds": round(len(audio) / SAMPLE_RATE, 2),
                        "realtime_factor": round(
                            (len(audio) / SAMPLE_RATE) / max(total / 1000.0, 1e-6), 1
                        ),
                        "language": language or "auto",
                    },
                )

        return Inference(
            "",
            self.key,
            "stub",
            (time.perf_counter() - start) * 1000.0,
            degraded=True,
            extra={
                "audio_seconds": round(len(audio) / SAMPLE_RATE, 2),
                "hint": "Whisper assets absent; run scripts/fetch_models.py",
            },
        )

    def _detokenize(self, token_ids: np.ndarray) -> str:
        try:
            from tokenizers import Tokenizer

            path = self.cache.model_root / self.key / "tokenizer.json"
            tokenizer = Tokenizer.from_file(str(path))
        except Exception:
            return ""
        ids = [int(i) for i in np.asarray(token_ids).reshape(-1).tolist() if int(i) > 0]
        return tokenizer.decode(ids).strip()
