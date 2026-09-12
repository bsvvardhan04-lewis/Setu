"""FastAPI backend. Binds to loopback only; makes no outbound connection, ever."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..config import SUPPORTED_LANGUAGES, settings
from ..models.speech import SAMPLE_RATE
from ..pipeline import ConsultAgent, DocAgent, FormAgent, TakeHomeCard, VoiceAgent, get_engine
from ..pipeline.form_agent import TEMPLATES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("setu")

UI_DIR = Path(__file__).resolve().parents[3] / "ui"

app = FastAPI(title="SETU", version="0.1.0", docs_url="/api/docs")

engine = get_engine()
doc_agent = DocAgent(engine)
form_agent = FormAgent(engine)
voice_agent = VoiceAgent(engine)
consult_agent = ConsultAgent(engine)


class AskRequest(BaseModel):
    question: str
    document_id: int | None = None
    language: str | None = None


class FormStartRequest(BaseModel):
    session_id: str
    template: str


class FormFillRequest(BaseModel):
    session_id: str
    utterance: str
    field_key: str | None = None


class SpeakRequest(BaseModel):
    text: str
    language: str = "hi"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/api/system")
def system() -> dict:
    """Full transparency payload: host, providers, placements, learned costs, assets."""
    return engine.system_report()


@app.get("/api/languages")
def languages() -> dict:
    """Every language, and whether translation is actually available for it.

    All of them are transcribed, language-identified, glossed and plan-extracted. Only
    some have a translation checkpoint that genuinely works, and the UI says which -
    offering a language that silently returns English would be worse than not listing it.
    """
    from ..models.translate_seq2seq import SUPPORTED_TARGETS

    return {
        "languages": SUPPORTED_LANGUAGES,
        "translatable": sorted(SUPPORTED_TARGETS),
        "default": settings.default_language,
    }


@app.get("/api/documents")
def documents() -> dict:
    return {"documents": engine.store.documents()}


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: int) -> dict:
    engine.store.delete_document(document_id)
    return {"deleted": document_id}


@app.post("/api/ingest")
async def ingest(files: list[UploadFile] = File(...), title: str = Form("Untitled")) -> dict:
    from PIL import Image

    images = []
    for upload in files:
        raw = await upload.read()
        name = (upload.filename or "").lower()
        if name.endswith(".pdf"):
            images.extend(_pdf_pages(raw))
        else:
            images.append(Image.open(io.BytesIO(raw)))
    if not images:
        raise HTTPException(400, "no readable pages in upload")

    report = doc_agent.ingest(images, title=title)
    return report.as_dict()


def _pdf_pages(raw: bytes) -> list:
    """Rasterise a PDF locally. pypdfium2 if present, else ask the user for images."""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(raw)
        return [page.render(scale=2).to_pil() for page in pdf]
    except Exception as exc:
        raise HTTPException(
            400,
            "PDF rendering needs pypdfium2 (pip install pypdfium2). "
            f"Upload page images instead. ({exc})",
        )


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    if not request.question.strip():
        raise HTTPException(400, "question is empty")
    return doc_agent.ask(
        request.question, document_id=request.document_id, language=request.language
    ).as_dict()


@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...), language: str = Form("en")
) -> dict:
    raw = await audio.read()
    samples = _decode_wav(raw)
    return voice_agent.transcribe(samples, language=language or None).as_dict()


def _decode_wav(raw: bytes) -> np.ndarray:
    import wave

    with wave.open(io.BytesIO(raw), "rb") as fh:
        frames = fh.readframes(fh.getnframes())
        width = fh.getsampwidth()
        channels = fh.getnchannels()
        rate = fh.getframerate()

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width, np.int16)
    samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    samples /= float(np.iinfo(dtype).max)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE and len(samples):
        # Linear resample. Adequate for 16 kHz speech and avoids a scipy dependency.
        target_len = int(len(samples) * SAMPLE_RATE / rate)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, target_len),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)
    return samples


@app.get("/api/voice/stats")
def voice_stats() -> dict:
    return voice_agent.stats.as_dict()


@app.get("/api/forms")
def form_templates() -> dict:
    return {
        "templates": [
            {"key": key, "title": state.title, "fields": len(state.fields)}
            for key, state in TEMPLATES.items()
        ]
    }


@app.post("/api/forms/start")
def form_start(request: FormStartRequest) -> dict:
    try:
        state = form_agent.start(request.session_id, request.template)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    first = state.next_empty()
    return {"form": state.as_dict(), "next_prompt": first.label if first else None}


@app.post("/api/forms/fill")
def form_fill(request: FormFillRequest) -> dict:
    try:
        return form_agent.fill(request.session_id, request.utterance, request.field_key)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/speak")
def speak(request: SpeakRequest) -> dict:
    """Return synthesised audio as base64 WAV, or tell the client to synthesise locally."""
    result = engine.tts.synthesize(request.text, request.language)
    if result.degraded or not len(result.value):
        return JSONResponse(
            {
                "audio": None,
                "fallback": "client",
                "language": request.language,
                "detail": result.as_dict(),
            }
        )

    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(22050)
        fh.writeframes((np.clip(result.value, -1, 1) * 32767).astype(np.int16).tobytes())
    return {
        "audio": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "fallback": None,
        "detail": result.as_dict(),
    }




# --------------------------------------------------------------------- consultation

class ConsultStartRequest(BaseModel):
    session_id: str
    patient_language: str = "hi"
    domain: str = "clinic"


class ConsultTurnRequest(BaseModel):
    session_id: str
    speaker: str = "doctor"
    text: str


class TeachbackRequest(BaseModel):
    session_id: str
    restatement: str


@app.get("/api/domains")
def domains() -> dict:
    """The settings SETU can run. Same engine, different vocabulary each time."""
    from ..pipeline.domains import DEFAULT_DOMAIN, DOMAINS
    from ..pipeline.demo_script import SCRIPTS

    scripts_by_domain: dict[str, list[dict]] = {}
    for name, script in SCRIPTS.items():
        scripts_by_domain.setdefault(script.get("domain", "clinic"), []).append(
            {"name": name, "title": script["title"]}
        )

    return {
        "default": DEFAULT_DOMAIN,
        "domains": [
            {**d.as_dict(), "scripts": scripts_by_domain.get(key, [])}
            for key, d in DOMAINS.items()
        ],
    }


@app.post("/api/consult/start")
def consult_start(request: ConsultStartRequest) -> dict:
    session = consult_agent.start(
        request.session_id, request.patient_language, request.domain
    )
    return session.as_dict()


@app.post("/api/consult/turn")
def consult_turn(request: ConsultTurnRequest) -> dict:
    try:
        session = consult_agent.get(request.session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    allowed = {"doctor", "patient", session.domain.expert, session.domain.learner}
    if request.speaker not in allowed:
        raise HTTPException(400, f"speaker must be one of {sorted(allowed)}")
    try:
        result = consult_agent.add_turn(request.session_id, request.speaker, request.text)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    # The live placement panel is the point of the whole project, so every turn
    # reports where the work actually ran.
    result["router"] = engine.router.snapshot()
    result["power"] = _power_dict()
    return result


@app.post("/api/consult/scan")
async def consult_scan(
    session_id: str = Form(...),
    speaker: str = Form("doctor"),
    image: UploadFile = File(...),
) -> dict:
    """Read a prescription or notice and fold it into the session as a turn.

    Same downstream path as speech: the extracted text goes through jargon glossing and
    care-plan extraction, so a scanned prescription produces exactly the same take-home
    card a spoken consultation does.
    """
    from PIL import Image

    try:
        session = consult_agent.get(session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))

    raw = await image.read()
    try:
        page = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"could not read that image: {exc}")

    read = engine.ocr.read_page(page)
    text = read.value.text.strip()
    if not text:
        return {
            "text": "",
            "ocr": read.as_dict(),
            "detail": "no text found on the page",
        }

    speaker = speaker if speaker in {session.domain.expert, session.domain.learner} else session.domain.expert
    result = consult_agent.add_turn(session_id, speaker, text)
    result["ocr"] = read.as_dict()
    result["text"] = text
    result["router"] = engine.router.snapshot()
    result["power"] = _power_dict()
    return result


@app.get("/api/consult/{session_id}")
def consult_state(session_id: str) -> dict:
    try:
        return consult_agent.get(session_id).as_dict()
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/consult/{session_id}/refine")
def consult_refine(session_id: str) -> dict:
    try:
        return consult_agent.refine_plan(session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/consult/{session_id}/teachback")
def consult_teachback_questions(session_id: str) -> dict:
    try:
        return {"questions": consult_agent.teachback_questions(session_id)}
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/consult/teachback")
def consult_teachback(request: TeachbackRequest) -> dict:
    try:
        return consult_agent.check_teachback(request.session_id, request.restatement)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/consult/{session_id}/card")
def consult_card(session_id: str) -> dict:
    try:
        card = TakeHomeCard(consult_agent).build(session_id)
        card["printable"] = TakeHomeCard(consult_agent).to_text(session_id)
        return card
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/consult/demo/{name}")
def consult_demo_script(name: str) -> dict:
    """Serve a scripted consultation for the UI to replay through the real pipeline."""
    from ..pipeline.demo_script import SCRIPTS

    script = SCRIPTS.get(name)
    if script is None:
        raise HTTPException(404, f"no demo script named {name!r}")
    return script


def _power_dict() -> dict:
    from ..runtime import read_power_state

    state = read_power_state()
    return {
        "on_ac": state.on_ac,
        "battery_percent": state.battery_percent,
        "discharge_mw": state.discharge_mw,
        "cpu_percent": state.cpu_percent,
        "thermal_pressure": round(state.thermal_pressure, 3),
        "energy_constrained": state.energy_constrained,
    }


def main() -> None:
    import uvicorn

    log.info("SETU listening on http://%s:%s (offline)", settings.host, settings.port)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
