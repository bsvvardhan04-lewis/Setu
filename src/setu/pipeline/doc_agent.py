"""DocAgent - ingest a document, then answer grounded questions about it.

Ingest:  pages -> OCR -> language id -> chunk -> embed (BACKGROUND) -> sqlite
Ask:     question -> embed (INTERACTIVE) -> retrieve -> prompt -> LLM -> translate out

Two things distinguish this from a generic RAG demo:

* Every chunk keeps its page and pixel box, so an answer can point at the region of the
  scan it came from. On an official document, "where does it say that" is the question.
* The pipeline is explicitly staged by priority, so Hexa-Router treats indexing and
  answering as different workloads with different power budgets.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..config import SUPPORTED_LANGUAGES
from ..models import PageResult
from ..runtime import Priority
from ..store import chunk_text
from .engine import Engine

log = logging.getLogger(__name__)

_SYSTEM = """You are SETU, an offline assistant that helps Indian citizens understand \
official documents. You are running on the user's own laptop; nothing they show you \
leaves the device.

Rules:
- Answer ONLY from the CONTEXT. If the context does not contain the answer, say so plainly.
- Be concrete: quote amounts, dates, names, and deadlines exactly as they appear.
- Never invent a reference number, an office address, or a deadline.
- Reply in {language_name}. Use simple words; assume the reader is not a lawyer.
- If the document asks the reader to do something, end with a short "What to do next" list.
"""

_PROMPT = """{system}

<CONTEXT>
{context}
</CONTEXT>

<QUESTION>
{question}
</QUESTION>

Answer:"""


@dataclass
class IngestReport:
    document_id: int
    title: str
    pages: int
    chunks: int
    language: str
    timings_ms: dict[str, float] = field(default_factory=dict)
    devices: dict[str, str] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "pages": self.pages,
            "chunks": self.chunks,
            "language": self.language,
            "language_name": SUPPORTED_LANGUAGES.get(self.language, self.language),
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "devices": self.devices,
            "degraded": self.degraded,
        }


@dataclass
class Answer:
    text: str
    citations: list[dict]
    timings_ms: dict[str, float]
    devices: dict[str, str]
    degraded: list[str]
    language: str

    def as_dict(self) -> dict[str, object]:
        return {
            "answer": self.text,
            "citations": self.citations,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "devices": self.devices,
            "degraded": self.degraded,
            "language": self.language,
            "language_name": SUPPORTED_LANGUAGES.get(self.language, self.language),
        }


class DocAgent:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # ---------------------------------------------------------------------- ingest

    def ingest(self, images: list, title: str) -> IngestReport:
        engine = self.engine
        timings: dict[str, float] = {}
        devices: dict[str, str] = {}
        degraded: list[str] = []

        pages: list[PageResult] = []
        ocr_ms = 0.0
        for index, image in enumerate(images):
            inference = engine.ocr.read_page(image, page=index)
            ocr_ms += inference.latency_ms
            devices["ocr"] = inference.device
            if inference.degraded:
                degraded.append(f"ocr:{inference.extra.get('path', 'stub')}")
            pages.append(inference.value)
        timings["ocr"] = ocr_ms

        full_text = "\n".join(p.text for p in pages)
        lid = engine.lid.identify(full_text)
        timings["lid"] = lid.latency_ms
        devices["lid"] = lid.device
        language = lid.value

        document_id = engine.store.add_document(
            title=title, language=language, meta={"pages": len(pages)}
        )

        texts: list[str] = []
        page_numbers: list[int] = []
        boxes: list[list[int]] = []
        for page in pages:
            for piece in chunk_text(
                page.text,
                size=engine.settings.chunk_chars,
                overlap=engine.settings.chunk_overlap,
            ):
                texts.append(piece)
                page_numbers.append(page.page)
                boxes.append(_span_box(page, piece))

        if texts:
            embedded = engine.embed.encode(texts, priority=Priority.BACKGROUND)
            timings["embed"] = embedded.latency_ms
            devices["embed"] = embedded.device
            if embedded.degraded:
                degraded.append("embed:hashed-ngram")
            engine.store.add_chunks(document_id, texts, embedded.value, page_numbers, boxes)

        return IngestReport(
            document_id=document_id,
            title=title,
            pages=len(pages),
            chunks=len(texts),
            language=language,
            timings_ms=timings,
            devices=devices,
            degraded=degraded,
        )

    # ------------------------------------------------------------------------- ask

    def ask(
        self,
        question: str,
        *,
        document_id: int | None = None,
        language: str | None = None,
    ) -> Answer:
        engine = self.engine
        timings: dict[str, float] = {}
        devices: dict[str, str] = {}
        degraded: list[str] = []

        target = language or engine.lid.identify(question).value
        started = time.perf_counter()

        query_vec = engine.embed.encode([question], priority=Priority.INTERACTIVE)
        timings["embed_query"] = query_vec.latency_ms
        devices["embed"] = query_vec.device
        if query_vec.degraded:
            degraded.append("embed:hashed-ngram")

        chunks = engine.store.search(
            query_vec.value[0], top_k=engine.settings.top_k, document_id=document_id
        )
        timings["retrieve"] = (time.perf_counter() - started) * 1000.0 - query_vec.latency_ms

        context = "\n\n".join(f"[p{c.page + 1}] {c.text}" for c in chunks)
        prompt = _PROMPT.format(
            system=_SYSTEM.format(language_name=SUPPORTED_LANGUAGES.get(target, "English")),
            context=context,
            question=question,
        )

        generated = engine.llm.generate(prompt)
        timings["llm"] = generated.latency_ms
        devices["llm"] = generated.device
        if generated.degraded:
            degraded.append("llm:template")

        answer = generated.value
        if target != "en" and generated.degraded:
            # The template backend cannot write in the target language; the specialist can.
            translated = engine.translate.translate(answer, "en", target)
            timings["translate"] = translated.latency_ms
            devices["translate"] = translated.device
            if translated.degraded:
                degraded.append("translate:absent")
            else:
                answer = translated.value

        return Answer(
            text=answer,
            citations=[c.as_dict() for c in chunks],
            timings_ms=timings,
            devices=devices,
            degraded=degraded,
            language=target,
        )


def _span_box(page: PageResult, text_piece: str) -> list[int]:
    """Union of the boxes of every region whose text appears in this chunk.

    Gives the UI a rectangle to highlight on the scan when the user taps a citation.
    """
    hits = [r.box for r in page.regions if r.text and r.text in text_piece]
    if not hits:
        return []
    return [
        min(b[0] for b in hits),
        min(b[1] for b in hits),
        max(b[2] for b in hits),
        max(b[3] for b in hits),
    ]
