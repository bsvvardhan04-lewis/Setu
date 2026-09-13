"""Document capture.

Three tiers: our quantised ONNX graphs (NPU-capable), the OCR engine that ships with
Windows (real, offline, CPU), then a labelled stub. The middle tier is why scanning works
on any Windows machine with nothing downloaded.

The bug these tests exist for: `_windows_ocr` drove WinRT with `asyncio.run`, which raises
inside a live event loop - exactly the case in a FastAPI handler. The adapter caught it and
fell back, so OCR worked perfectly when called in-process and silently returned nothing
over HTTP. Only testing through the real endpoint surfaced it.
"""

from __future__ import annotations

import platform
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from setu.config import Settings
from setu.pipeline import ConsultAgent, Engine

REPO = Path(__file__).resolve().parents[1]

PRESCRIPTION = [
    "Diagnosis: Hypertension",
    "Tab. Amlodipine 5 mg - once at night",
    "Get lipid profile and HbA1c, fasting",
    "Review after 2 weeks with reports",
    "If chest pain or breathlessness,",
    "come immediately to emergency",
]


def make_page() -> Image.Image:
    """Render the prescription at a size a real scan would be.

    PIL's default bitmap font is far too small for any OCR engine to read reliably, so a
    TrueType face is required here - testing against unreadably small text would measure
    the fixture, not the adapter.
    """
    font = None
    for candidate in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "calibri.ttf"):
        try:
            font = ImageFont.truetype(candidate, 26)
            break
        except Exception:
            continue
    if font is None:
        pytest.skip("no TrueType font available to render a legible test page")

    img = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(img)
    y = 40
    for line in PRESCRIPTION:
        draw.text((50, y), line, fill="black", font=font)
        y += 48
    return img


@pytest.fixture
def engine(tmp_path):
    settings = Settings()
    settings.data_dir = tmp_path
    settings.db_path = tmp_path / "setu.db"
    return Engine(settings)


def has_windows_ocr(engine) -> bool:
    return engine.ocr._windows_ocr_available()


def test_blank_page_yields_no_regions(engine):
    result = engine.ocr.read_page(Image.new("RGB", (400, 300), "white"))
    assert result.value.text.strip() == ""


def test_availability_reflects_any_real_path(engine):
    """`doctor` must not report OCR missing while the OS can read the page."""
    if has_windows_ocr(engine):
        assert engine.ocr.available() is True
        assert engine.ocr.status()["windows_ocr"] is True


@pytest.mark.skipif(platform.system() != "Windows", reason="uses the Windows OCR engine")
def test_reads_a_prescription(engine):
    if not has_windows_ocr(engine):
        pytest.skip("no OCR language pack installed")

    result = engine.ocr.read_page(make_page())
    text = result.value.text.lower()

    # A real OCR engine doing real work is not a degraded path.
    assert result.degraded is False
    assert result.extra["path"] == "windows-ocr"
    assert result.extra["npu_accelerated"] is False
    for word in ("amlodipine", "hypertension", "emergency"):
        assert word in text, f"{word!r} missing from {text!r}"


@pytest.mark.skipif(platform.system() != "Windows", reason="uses the Windows OCR engine")
def test_a_scan_produces_the_same_care_plan_a_conversation_would(engine):
    if not has_windows_ocr(engine):
        pytest.skip("no OCR language pack installed")

    result = engine.ocr.read_page(make_page())
    agent = ConsultAgent(engine)
    agent.start("scan", patient_language="hi", domain="clinic")
    agent.add_turn("scan", "doctor", result.value.text)

    kinds = {item.kind for item in agent.get("scan").plan}
    assert {"medication", "redflag"} <= kinds, f"got {kinds}"
    assert "hypertension" in agent.get("scan").glosses


def test_ocr_works_inside_a_running_event_loop(engine):
    """The regression guard.

    `asyncio.run` raises inside a live loop, which is every FastAPI request. The adapter's
    broad except turned that into a silent empty result. This reproduces the condition
    directly so it can never regress unnoticed again.
    """
    import asyncio

    if not has_windows_ocr(engine):
        pytest.skip("no OCR language pack installed")

    page = make_page()

    async def inside_a_loop():
        return engine.ocr.read_page(page)

    result = asyncio.run(inside_a_loop())
    assert result.extra["path"] == "windows-ocr", (
        "OCR fell back to the stub inside an event loop - the asyncio.run bug is back"
    )
    assert "amlodipine" in result.value.text.lower()
