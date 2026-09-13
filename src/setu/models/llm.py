"""LLM backends.

An LLM on Snapdragon does not go through ONNX Runtime the way a CNN does: autoregressive
decode needs a runtime that owns the KV cache and a pre-compiled HTP context binary. So
this module defines a small ``LlmBackend`` interface with four implementations, tried in
order:

  1. ``GenieBackend``      - QAIRT / Genie, the NPU path on Snapdragon. Fastest, lowest power.
  2. ``OrtGenAIBackend``   - onnxruntime-genai with the QNN provider. Also NPU, easier to build.
  3. ``LlamaCppBackend``   - llama.cpp on ARM64/x64 CPU. Universal fallback, real answers.
  4. ``TemplateBackend``   - no weights at all. Deterministic, extractive, grounded in the
                             retrieved chunks. Keeps the product demonstrable and the tests
                             hermetic; always reports ``degraded=True``.

Selecting a backend is a runtime decision, so the rest of SETU only ever sees ``generate``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..runtime import Priority
from .base import Adapter, Inference

log = logging.getLogger(__name__)


@dataclass
class GenParams:
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    stop: tuple[str, ...] = ()


class LlmBackend(Protocol):
    name: str
    device: str

    def available(self) -> bool: ...

    def generate(self, prompt: str, params: GenParams) -> str: ...


# --------------------------------------------------------------------------- Genie (NPU)


class GenieBackend:
    """Qualcomm Genie runtime - the production NPU path.

    Genie ships a CLI (``genie-t2t-run``) and a C API. We drive the CLI because it keeps
    the dependency surface to "QAIRT is installed", which is exactly what the setup script
    guarantees, and because the process boundary isolates a native crash from the server.
    """

    name = "genie"
    device = "npu"

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.config = model_dir / "genie_config.json"
        self.binary = os.environ.get("SETU_GENIE_BIN") or shutil.which("genie-t2t-run")

    def available(self) -> bool:
        return bool(self.binary) and self.config.is_file()

    def generate(self, prompt: str, params: GenParams) -> str:
        cmd = [
            self.binary,
            "-c",
            str(self.config),
            "-p",
            prompt,
        ]
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(self.model_dir),
        )
        if out.returncode != 0:
            raise RuntimeError(f"genie-t2t-run failed: {out.stderr.strip()[:400]}")
        return _strip_genie_framing(out.stdout)


def _strip_genie_framing(raw: str) -> str:
    """genie-t2t-run echoes the prompt and wraps output in [BEGIN]/[END] markers."""
    match = re.search(r"\[BEGIN\]:?(.*?)\[END\]", raw, flags=re.DOTALL)
    text = match.group(1) if match else raw
    return text.strip()


# ------------------------------------------------------------------ onnxruntime-genai (NPU)


class OrtGenAIBackend:
    name = "ort-genai"
    device = "npu"

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._model = None
        self._tokenizer = None

    def available(self) -> bool:
        return (self.model_dir / "genai_config.json").is_file()

    def _load(self) -> None:
        if self._model is not None:
            return
        import onnxruntime_genai as og

        self._og = og
        self._model = og.Model(str(self.model_dir))
        self._tokenizer = og.Tokenizer(self._model)

    def generate(self, prompt: str, params: GenParams) -> str:
        self._load()
        og = self._og
        gen_params = og.GeneratorParams(self._model)
        gen_params.set_search_options(
            max_length=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            do_sample=params.temperature > 0.0,
        )
        generator = og.Generator(self._model, gen_params)
        generator.append_tokens(self._tokenizer.encode(prompt))
        stream = self._tokenizer.create_stream()
        pieces: list[str] = []
        while not generator.is_done():
            generator.generate_next_token()
            pieces.append(stream.decode(generator.get_next_tokens()[0]))
        return "".join(pieces).strip()


# ------------------------------------------------------------------------ llama.cpp (CPU)


class LlamaCppBackend:
    name = "llama.cpp"
    device = "cpu"

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._llm = None

    def _gguf(self) -> Path | None:
        files = sorted(self.model_dir.glob("*.gguf"))
        return files[0] if files else None

    def available(self) -> bool:
        if self._gguf() is None:
            return False
        try:
            import llama_cpp  # noqa: F401
        except Exception:
            return False
        return True

    def generate(self, prompt: str, params: GenParams) -> str:
        if self._llm is None:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=str(self._gguf()),
                n_ctx=4096,
                n_threads=os.cpu_count() or 8,
                verbose=False,
            )
        out = self._llm(
            prompt,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            stop=list(params.stop) or None,
        )
        return out["choices"][0]["text"].strip()


# ------------------------------------------------------------------- template (no weights)


_ANSWER_TEMPLATE = (
    "{lead}\n\n{body}\n\n"
    "[SETU is running without LLM weights, so this answer is extracted verbatim from the "
    "document rather than written. Install the Genie or GGUF asset for full reasoning.]"
)


class TemplateBackend:
    """Grounded extractive answering with zero weights.

    It parses the prompt SETU builds (a CONTEXT block plus a QUESTION), scores each context
    chunk against the question by term overlap, and returns the best sentences. Not a
    language model - but it is honest, deterministic, useful for tests, and it keeps the
    demo alive on a machine with nothing downloaded.
    """

    name = "template"
    device = "cpu"

    def available(self) -> bool:
        return True

    def generate(self, prompt: str, params: GenParams) -> str:
        context = _between(prompt, "<CONTEXT>", "</CONTEXT>")
        question = _between(prompt, "<QUESTION>", "</QUESTION>") or prompt
        if not context.strip():
            return _ANSWER_TEMPLATE.format(
                lead="I could not find anything in this document to answer that.",
                body="Try scanning more pages, or ask about text that appears on the page.",
            )

        q_terms = {t.lower() for t in re.findall(r"\w{3,}", question, re.UNICODE)}
        sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", context) if s.strip()]
        scored = sorted(
            sentences,
            key=lambda s: -len(q_terms & {t.lower() for t in re.findall(r"\w{3,}", s, re.UNICODE)}),
        )
        best = [s for s in scored[:3] if s]
        return _ANSWER_TEMPLATE.format(
            lead="Here is what the document says about that:",
            body="\n".join(f"- {s}" for s in best),
        )


def _between(text: str, start: str, end: str) -> str:
    match = re.search(re.escape(start) + r"(.*?)" + re.escape(end), text, flags=re.DOTALL)
    return match.group(1) if match else ""


# ------------------------------------------------------------------------------- adapter


class Llm(Adapter):
    key = "llm"
    priority = Priority.INTERACTIVE

    def __init__(self, cache) -> None:
        super().__init__(cache)
        model_dir = cache.model_root / "llm"
        self._candidates: list[LlmBackend] = [
            GenieBackend(model_dir),
            OrtGenAIBackend(model_dir),
            LlamaCppBackend(model_dir),
            TemplateBackend(),
        ]
        self._chosen: LlmBackend | None = None

    @property
    def backend(self) -> LlmBackend:
        if self._chosen is None:
            for candidate in self._candidates:
                try:
                    if candidate.available():
                        self._chosen = candidate
                        break
                except Exception as exc:
                    log.warning("backend %s probe failed: %s", candidate.name, exc)
            else:  # pragma: no cover - TemplateBackend is always available
                self._chosen = TemplateBackend()
            log.info("llm backend: %s (%s)", self._chosen.name, self._chosen.device)
        return self._chosen

    def available(self) -> bool:
        return not isinstance(self.backend, TemplateBackend)

    def generate(self, prompt: str, params: GenParams | None = None) -> Inference:
        params = params or GenParams()
        backend = self.backend
        start = time.perf_counter()
        try:
            text = backend.generate(prompt, params)
        except Exception as exc:
            log.error("llm backend %s failed, falling back: %s", backend.name, exc)
            self._chosen = TemplateBackend()
            backend = self._chosen
            text = backend.generate(prompt, params)
        latency_ms = (time.perf_counter() - start) * 1000.0

        from ..runtime import Device

        device = {"npu": Device.NPU, "cpu": Device.CPU}.get(backend.device, Device.CPU)
        self.cache.router.observe(self.key, device, latency_ms)

        approx_tokens = max(1, len(text) // 4)
        return Inference(
            value=text,
            model=self.key,
            device=backend.device,
            latency_ms=latency_ms,
            degraded=isinstance(backend, TemplateBackend),
            extra={
                "backend": backend.name,
                "approx_tokens": approx_tokens,
                "tokens_per_sec": round(approx_tokens / max(latency_ms / 1000.0, 1e-6), 2),
            },
        )

    def status(self) -> dict[str, object]:
        base = super().status()
        base["backend"] = self.backend.name
        base["backend_device"] = self.backend.device
        base["candidates"] = [
            {"name": c.name, "device": c.device, "available": _safe_available(c)}
            for c in self._candidates
        ]
        return base


def _safe_available(backend: LlmBackend) -> bool:
    try:
        return backend.available()
    except Exception:
        return False
