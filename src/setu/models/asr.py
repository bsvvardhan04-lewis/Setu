"""Speech recognition (Whisper) with a VAD gate in front of it.

The gate matters more than the model here. Always-on ASR is the workload that justifies
an NPU, but only if you are not decoding 30-second windows of silence all day. Silero VAD
costs microseconds and cuts Whisper invocations by an order of magnitude in a quiet room.

Decoding is a real autoregressive loop, not a single forward pass. Whisper emits one token
at a time, conditioned on everything it has already emitted, so a single decoder call
produces nothing usable. The loop here is deliberately the simple one - greedy, no KV
cache - because correctness on any machine matters more than throughput on this path:
on Snapdragon the whole model is replaced by the AI Hub export running under Genie/QNN,
which brings its own cache.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..runtime import Priority
from .base import Adapter, Inference

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
N_MELS = 80
CHUNK_SECONDS = 30
MAX_NEW_TOKENS = 180


def log_mel_spectrogram(audio: np.ndarray, n_mels: int = N_MELS) -> np.ndarray:
    """Whisper-compatible log-Mel features, numpy-only.

    Implemented here rather than pulled from torchaudio because the ARM64 wheel story for
    torch is unpleasant and this is 40 lines of FFT.
    """
    n_fft, hop = 400, 160
    audio = audio.astype(np.float32)
    target = SAMPLE_RATE * CHUNK_SECONDS
    audio = np.pad(audio, (0, max(0, target - len(audio))))[:target]

    # Whisper's encoder is a fixed-shape graph expecting exactly 3000 mel frames, and it
    # gets them from a CENTRE-padded STFT. Framing the raw signal instead yields 2998 and
    # the graph fails deep inside with a broadcast error about 1499 versus 1500, which
    # says nothing about the real cause. Reflect-pad by half a window, then trim the
    # trailing frame, exactly as the reference implementation does.
    expected_frames = target // hop  # 3000
    padded = np.pad(audio, (n_fft // 2, n_fft // 2), mode="reflect")

    window = np.hanning(n_fft).astype(np.float32)
    frames = 1 + (len(padded) - n_fft) // hop
    stft = np.empty((frames, n_fft // 2 + 1), dtype=np.complex64)
    for i in range(frames):
        segment = padded[i * hop : i * hop + n_fft] * window
        stft[i] = np.fft.rfft(segment)
    power = (np.abs(stft) ** 2).T[:, :expected_frames]

    mel_fb = _mel_filterbank(n_mels, n_fft, SAMPLE_RATE)
    mel = mel_fb @ power
    log_spec = np.log10(np.clip(mel, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def _hz_to_mel(hz):
    """Slaney mel scale: linear below 1 kHz, logarithmic above.

    Whisper does NOT use the HTK formula (2595*log10(1+f/700)). Using it produces mel
    features that look plausible, feed the encoder without error, and yield a transcript
    of pure noise - the worst kind of bug, because nothing fails.
    """
    hz = np.asarray(hz, dtype=np.float64)
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    mel = hz / f_sp
    above = hz >= min_log_hz
    mel = np.where(above, min_log_mel + np.log(np.maximum(hz, 1e-9) / min_log_hz) / logstep, mel)
    return mel


def _mel_to_hz(mel):
    mel = np.asarray(mel, dtype=np.float64)
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    hz = f_sp * mel
    above = mel >= min_log_mel
    return np.where(above, min_log_hz * np.exp(logstep * (mel - min_log_mel)), hz)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """Triangular mel filters with Slaney area normalisation, matching librosa.

    The normalisation matters as much as the scale: without it, high-frequency filters
    integrate over far more FFT bins than low-frequency ones and the resulting features
    are tilted well outside the range the encoder was trained on.
    """
    n_bins = n_fft // 2 + 1
    fftfreqs = np.linspace(0.0, sr / 2.0, n_bins)

    mels = np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2.0), n_mels + 2)
    mel_f = _mel_to_hz(mels)

    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]

    weights = np.zeros((n_mels, n_bins), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


class Asr(Adapter):
    key = "asr"
    primary_asset = "asr_encoder.onnx"
    priority = Priority.STREAMING

    def __init__(self, cache) -> None:
        super().__init__(cache)
        self._tokenizer = None
        self._special: dict[str, int] | None = None

    # ---------------------------------------------------------------- tokenizer

    def _load_tokenizer(self):
        if self._tokenizer is None:
            try:
                from tokenizers import Tokenizer

                path = self.cache.model_root / self.key / "tokenizer.json"
                self._tokenizer = Tokenizer.from_file(str(path))
            except Exception as exc:
                log.debug("whisper tokenizer unavailable: %s", exc)
                return None
        return self._tokenizer

    def _special_tokens(self, language: str | None) -> dict[str, int] | None:
        """Whisper is steered entirely by the tokens you seed the decoder with.

        The prompt is <|startoftranscript|><|lang|><|transcribe|><|notimestamps|>; get it
        wrong and the model happily translates, or emits timestamps you then have to strip.
        """
        tokenizer = self._load_tokenizer()
        if tokenizer is None:
            return None

        def tok(text: str) -> int | None:
            found = tokenizer.token_to_id(text)
            return int(found) if found is not None else None

        start = tok("<|startoftranscript|>")
        eot = tok("<|endoftext|>")
        if start is None or eot is None:
            return None

        prompt = [start]
        lang_token = tok(f"<|{language}|>") if language and language != "auto" else None
        if lang_token is not None:
            prompt.append(lang_token)
        for name in ("<|transcribe|>", "<|notimestamps|>"):
            value = tok(name)
            if value is not None:
                prompt.append(value)
        return {"prompt": prompt, "eot": eot}

    # ------------------------------------------------------------------- decode

    def _greedy_decode(self, encoder_states: np.ndarray, prompt: list[int], eot: int):
        """Emit tokens one at a time until end-of-text.

        No KV cache: each step re-runs the decoder over the whole prefix. That is O(n^2)
        and slow, but it is correct against the plain `decoder_model.onnx` graph and needs
        no past-key plumbing. The Snapdragon path swaps this for Genie, which caches.
        """
        tokens = list(prompt)
        for _ in range(MAX_NEW_TOKENS):
            result = self.cache.run(
                self.key,
                {
                    "input_ids": np.array([tokens], dtype=np.int64),
                    "encoder_hidden_states": encoder_states,
                },
                filename="asr_decoder.onnx",
                priority=self.priority,
            )
            if result is None:
                return None, 0.0
            logits = np.asarray(result.outputs[0])
            next_token = int(logits[0, -1].argmax())
            if next_token == eot:
                break
            tokens.append(next_token)
        return tokens[len(prompt) :], 0.0

    # --------------------------------------------------------------- entrypoint

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Inference:
        start = time.perf_counter()
        audio_seconds = len(audio) / SAMPLE_RATE
        mel = log_mel_spectrogram(audio)[None, ...]

        encoded = self.cache.run(
            self.key,
            {"input_features": mel},
            filename="asr_encoder.onnx",
            priority=self.priority,
        )
        special = self._special_tokens(language)

        if encoded is not None and special is not None:
            encoder_states = np.asarray(encoded.outputs[0], dtype=np.float32)
            new_tokens, _ = self._greedy_decode(
                encoder_states, special["prompt"], special["eot"]
            )
            if new_tokens is not None:
                tokenizer = self._load_tokenizer()
                text = tokenizer.decode(new_tokens).strip() if tokenizer else ""
                total = (time.perf_counter() - start) * 1000.0
                return Inference(
                    text,
                    self.key,
                    encoded.device.value,
                    total,
                    extra={
                        "audio_seconds": round(audio_seconds, 2),
                        "realtime_factor": round(
                            audio_seconds / max(total / 1000.0, 1e-6), 2
                        ),
                        "tokens": len(new_tokens),
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
                "audio_seconds": round(audio_seconds, 2),
                "hint": "Whisper assets absent; run scripts/fetch_models.py --model asr",
            },
        )
