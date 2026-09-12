"""Voice activity detection and speech synthesis."""

from __future__ import annotations

import json
import logging
import pathlib
import platform
import subprocess
import time
import wave

import numpy as np

from ..runtime import Priority
from .base import Adapter, Inference

log = logging.getLogger(__name__)


def _ps_quote(value: str) -> str:
    """Single-quote a PowerShell literal, doubling any internal quotes."""
    return "'" + value.replace("'", "''") + "'"

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
    """Piper VITS synthesis, with the client's on-device synthesiser as the fallback.

    Speech output is not a nice-to-have here: a user who cannot read the form cannot read
    our explanation of the form either.

    **The model is deliberately not used unless it can be driven correctly.** Piper voices
    declare a ``phoneme_type``, and the Indic voices are ``espeak`` - they expect IPA
    phonemes from espeak-ng, not characters. Feeding a character-derived id sequence to an
    espeak voice does not fail; it synthesises confident, fluent-sounding noise. For a tool
    whose entire purpose is that a patient understood the instruction, plausible gibberish
    is worse than silence - so with no phonemiser installed this reports `degraded` and the
    client speaks the text instead. Still on the device, just not through our model.
    """

    key = "tts"
    priority = Priority.INTERACTIVE
    #: language -> SAPI voice name (or None). Enumerating voices costs a subprocess.
    _sapi_cache: dict[str, str | None] = {}

    # --------------------------------------------------------------- SAPI backend
    #
    # Same move as document capture: when our own model cannot be driven, use the engine
    # the operating system already ships rather than pretending the capability is absent.
    # SAPI is real, fully offline synthesis - it just is not ours, and it only covers the
    # languages whose voices are installed, so both facts get reported.

    def _sapi_voice(self, language: str) -> str | None:
        """Name of an installed SAPI voice for this language, if there is one."""
        if platform.system() != "Windows":
            return None
        if language in self._sapi_cache:
            return self._sapi_cache[language]

        script = (
            "Add-Type -AssemblyName System.Speech;"
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$s.GetInstalledVoices() | ForEach-Object {"
            " $_.VoiceInfo.Culture.TwoLetterISOLanguageName + '|' + $_.VoiceInfo.Name };"
            "$s.Dispose()"
        )
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, text=True, timeout=25,
            )
        except Exception:
            self._sapi_cache[language] = None
            return None

        chosen = None
        for line in out.stdout.splitlines():
            if "|" not in line:
                continue
            code, _, name = line.strip().partition("|")
            if code == language:
                chosen = name
                break
        self._sapi_cache[language] = chosen
        return chosen

    def _sapi_synthesise(self, text: str, voice: str) -> np.ndarray | None:
        """Render to a 16 kHz mono WAV through SAPI and read it back."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            wav = pathlib.Path(tmp) / "speech.wav"
            script = (
                "Add-Type -AssemblyName System.Speech;"
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                f"$s.SelectVoice({_ps_quote(voice)});"
                "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
                "16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,"
                " [System.Speech.AudioFormat.AudioChannel]::Mono);"
                f"$s.SetOutputToWaveFile({_ps_quote(str(wav))}, $f);"
                "$s.Rate = -1;"
                f"$s.Speak({_ps_quote(text)});"
                "$s.Dispose()"
            )
            try:
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                    capture_output=True, text=True, timeout=90,
                )
                if not wav.is_file() or wav.stat().st_size < 100:
                    return None
                with wave.open(str(wav), "rb") as fh:
                    pcm = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16)
            except Exception as exc:
                log.debug("SAPI synthesis failed: %s", exc)
                return None
        return (pcm.astype(np.float32) / 32768.0) if len(pcm) else None

    def available(self) -> bool:
        """Loaded AND drivable.

        The default check only asks whether the weights are on disk. For this voice that
        would report a capability we deliberately decline to use, and `doctor` would tell
        a user speech synthesis works when it does not.
        """
        if super().available() and self._phonemiser() is not None:
            return True
        # The OS engine counts: it is real, offline synthesis.
        return self._sapi_voice("en") is not None

    def _phonemiser(self):
        """Return a text -> phoneme-id function, or None if this voice cannot be driven."""
        try:
            config = json.loads(
                (self.cache.model_root / self.key / "tts.onnx.json").read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return None

        id_map = config.get("phoneme_id_map") or {}
        if not id_map:
            return None

        if config.get("phoneme_type") != "espeak":
            # A character-type voice can be driven straight from its own id map.
            return lambda text: _ids_from_map(text, id_map)

        try:
            import piper_phonemize
        except Exception:
            return None

        voice = (config.get("espeak") or {}).get("voice", "hi")

        def espeak_ids(text: str):
            sentences = piper_phonemize.phonemize_espeak(text, voice)
            return _ids_from_map("".join(p for s in sentences for p in s), id_map)

        return espeak_ids

    def synthesize(self, text: str, language: str = "hi") -> Inference:
        start = time.perf_counter()

        phonemise = self._phonemiser()
        if phonemise is None:
            # Our model cannot be driven. Try the engine the OS already ships before
            # giving up - it is real offline synthesis, just not ours.
            voice = self._sapi_voice(language)
            if voice is not None:
                audio = self._sapi_synthesise(text, voice)
                if audio is not None and len(audio):
                    return Inference(
                        audio,
                        self.key,
                        "cpu",
                        (time.perf_counter() - start) * 1000.0,
                        degraded=False,
                        extra={
                            "seconds": round(len(audio) / SAMPLE_RATE, 2),
                            "sample_rate": SAMPLE_RATE,
                            "path": "windows-sapi",
                            "engine": f"Windows SAPI ({voice}), offline, OS-provided",
                            "npu_accelerated": False,
                            "language": language,
                        },
                    )
            return self._fallback(
                start,
                language,
                "this Piper voice needs espeak-ng phonemes and no phonemiser is installed, "
                "and no installed system voice covers this language; synthesising from "
                "characters would produce fluent noise",
            )
        phoneme_ids = phonemise(text)
        if phoneme_ids is None:
            return self._fallback(start, language, "could not phonemise the text")

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

        return self._fallback(start, language, "synthesis graph unavailable")

    def _fallback(self, start: float, language: str, reason: str) -> Inference:
        return Inference(
            np.zeros(0, dtype=np.float32),
            self.key,
            "stub",
            (time.perf_counter() - start) * 1000.0,
            degraded=True,
            extra={
                "path": "client-speech-synthesis",
                "note": "the client speaks the text with its own on-device synthesiser",
                "reason": reason,
                "language": language,
            },
        )


def _ids_from_map(symbols: str, id_map: dict, max_len: int = 512) -> np.ndarray | None:
    """Map symbols through the voice's own phoneme_id_map.

    Piper interleaves a padding id between every symbol, which the model was trained with;
    omitting it degrades prosody badly. Unknown symbols are dropped rather than guessed -
    a wrong id is a wrong sound, and there is no benign wrong sound in a care instruction.
    """
    pad = (id_map.get("_") or [0])[0]
    bos = (id_map.get("^") or [1])[0]
    eos = (id_map.get("$") or [2])[0]

    ids: list[int] = [bos, pad]
    for symbol in symbols[:max_len]:
        mapped = id_map.get(symbol)
        if not mapped:
            continue
        ids.extend(mapped)
        ids.append(pad)
    ids.append(eos)
    return np.array(ids, dtype=np.int64) if len(ids) > 4 else None


def write_wav(path, audio: np.ndarray, sample_rate: int = 22050) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes((pcm * 32767).astype(np.int16).tobytes())
