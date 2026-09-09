"""Document text extraction: detect regions, then recognise each region.

Two separate models on purpose. The detector is a fixed-shape convolutional graph that
sits perfectly on the HTP; the recogniser is a batched CRNN. Splitting them lets
Hexa-Router place them independently and lets us batch line crops, which is the single
biggest throughput lever on an NPU (one submission of 32 crops beats 32 submissions).

Fallback ladder: ONNX models -> Windows.Media.Ocr (built into Windows, works offline,
covers Latin + several Indic scripts) -> embedded PDF text layer -> labelled stub.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..runtime import Priority
from .base import Adapter, Inference

log = logging.getLogger(__name__)

_DET_SIZE = 960
_REC_HEIGHT = 32
_REC_WIDTH = 320
_REC_BATCH = 32


@dataclass
class TextRegion:
    """One recognised region, with the page geometry needed to cite it later."""

    text: str
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in page pixels
    confidence: float = 1.0
    page: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "box": list(self.box),
            "confidence": round(self.confidence, 3),
            "page": self.page,
        }


@dataclass
class PageResult:
    regions: list[TextRegion] = field(default_factory=list)
    width: int = 0
    height: int = 0
    page: int = 0

    @property
    def text(self) -> str:
        """Reading-order text. Sort by row band, then by x, so multi-column pages
        do not interleave into nonsense."""
        if not self.regions:
            return ""
        band = max(12, int(self.height * 0.012)) if self.height else 12
        ordered = sorted(self.regions, key=lambda r: (r.box[1] // band, r.box[0]))
        lines: list[str] = []
        current_band = None
        buffer: list[str] = []
        for region in ordered:
            rb = region.box[1] // band
            if current_band is None or rb == current_band:
                buffer.append(region.text)
            else:
                lines.append(" ".join(buffer))
                buffer = [region.text]
            current_band = rb
        if buffer:
            lines.append(" ".join(buffer))
        return "\n".join(line.strip() for line in lines if line.strip())


def _to_gray_array(image) -> np.ndarray:
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    return np.asarray(image.convert("L"), dtype=np.uint8)


class Ocr(Adapter):
    """Detector + recogniser pair, presented as one adapter."""

    key = "ocr_detect"
    priority = Priority.INTERACTIVE
    recognizer_key = "ocr_recognize"

    def __init__(self, cache) -> None:
        super().__init__(cache)
        self._charset: list[str] | None = None

    # ------------------------------------------------------------------ detection

    def _detect_onnx(self, gray: np.ndarray) -> list[tuple[int, int, int, int]] | None:
        h, w = gray.shape
        from PIL import Image

        resized = np.asarray(
            Image.fromarray(gray).resize((_DET_SIZE, _DET_SIZE), Image.BILINEAR), dtype=np.float32
        )
        tensor = ((resized / 255.0 - 0.5) / 0.5)[None, None, :, :].astype(np.float32)

        result = self.cache.run(self.key, {"x": tensor}, priority=self.priority)
        if result is None:
            return None

        prob = np.asarray(result.outputs[0]).squeeze()
        mask = prob > 0.3
        boxes = _connected_boxes(mask)
        sx, sy = w / _DET_SIZE, h / _DET_SIZE
        return [
            (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)) for x0, y0, x1, y1 in boxes
        ]

    # ---------------------------------------------------------------- recognition

    def _charset_list(self) -> list[str]:
        if self._charset is None:
            path = self.cache.model_root / self.recognizer_key / "charset.txt"
            try:
                self._charset = ["<blank>"] + path.read_text(encoding="utf-8").splitlines()
            except Exception:
                self._charset = ["<blank>"]
        return self._charset

    def _recognize_onnx(
        self, gray: np.ndarray, boxes: list[tuple[int, int, int, int]]
    ) -> list[TextRegion] | None:
        from PIL import Image

        crops = []
        for x0, y0, x1, y1 in boxes:
            crop = gray[max(0, y0) : y1, max(0, x0) : x1]
            if crop.size == 0:
                crop = np.zeros((_REC_HEIGHT, _REC_WIDTH), np.uint8)
            crops.append(
                np.asarray(
                    Image.fromarray(crop).resize((_REC_WIDTH, _REC_HEIGHT), Image.BILINEAR),
                    dtype=np.float32,
                )
            )

        charset = self._charset_list()
        regions: list[TextRegion] = []
        for start in range(0, len(crops), _REC_BATCH):
            chunk = crops[start : start + _REC_BATCH]
            # Pad the batch to a fixed size: a static shape keeps the graph on the HTP.
            padded = chunk + [np.zeros((_REC_HEIGHT, _REC_WIDTH), np.float32)] * (
                _REC_BATCH - len(chunk)
            )
            tensor = (np.stack(padded)[:, None] / 255.0 - 0.5) / 0.5
            result = self.cache.run(
                self.recognizer_key, {"x": tensor.astype(np.float32)}, priority=self.priority
            )
            if result is None:
                return None
            logits = np.asarray(result.outputs[0])
            for i in range(len(chunk)):
                text, conf = _ctc_decode(logits[i], charset)
                regions.append(TextRegion(text=text, box=boxes[start + i], confidence=conf))
        return regions

    # ------------------------------------------------------------- Windows OCR path

    def _windows_ocr(self, image) -> list[TextRegion] | None:
        """Windows ships a fully offline OCR engine. On a Snapdragon HP PC it is already
        installed, needs no download, and is a legitimate accuracy floor when our ONNX
        assets are absent."""
        if platform.system() != "Windows":
            return None
        try:
            import asyncio
            import io

            from winsdk.windows.globalization import Language
            from winsdk.windows.graphics.imaging import BitmapDecoder
            from winsdk.windows.media.ocr import OcrEngine
            from winsdk.windows.storage.streams import DataWriter, InMemoryRandomAccessStream
        except Exception:
            return None

        async def run() -> list[TextRegion]:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            stream = InMemoryRandomAccessStream()
            writer = DataWriter(stream.get_output_stream_at(0))
            writer.write_bytes(buffer.getvalue())
            await writer.store_async()
            decoder = await BitmapDecoder.create_async(stream)
            bitmap = await decoder.get_software_bitmap_async()

            engine = OcrEngine.try_create_from_user_profile_languages()
            if engine is None:
                engine = OcrEngine.try_create_from_language(Language("en-US"))
            if engine is None:
                return []
            result = await engine.recognize_async(bitmap)
            out: list[TextRegion] = []
            for line in result.lines:
                rects = [w.bounding_rect for w in line.words]
                if not rects:
                    continue
                x0 = int(min(r.x for r in rects))
                y0 = int(min(r.y for r in rects))
                x1 = int(max(r.x + r.width for r in rects))
                y1 = int(max(r.y + r.height for r in rects))
                out.append(TextRegion(text=line.text, box=(x0, y0, x1, y1), confidence=0.9))
            return out

        try:
            return asyncio.run(run()) or None
        except Exception as exc:
            log.debug("Windows OCR unavailable: %s", exc)
            return None

    # ------------------------------------------------------------------- entrypoint

    def read_page(self, image, page: int = 0) -> Inference:
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        gray = _to_gray_array(image)
        h, w = gray.shape
        start = time.perf_counter()

        boxes = self._detect_onnx(gray)
        if boxes:
            regions = self._recognize_onnx(gray, boxes)
            if regions is not None:
                result = PageResult(regions=regions, width=w, height=h, page=page)
                return Inference(
                    result,
                    self.key,
                    self.cache.router.place(self.key).device.value,
                    (time.perf_counter() - start) * 1000.0,
                    extra={"regions": len(regions), "path": "onnx"},
                )

        windows_regions = self._windows_ocr(image)
        if windows_regions is not None:
            for r in windows_regions:
                r.page = page
            result = PageResult(regions=windows_regions, width=w, height=h, page=page)
            return Inference(
                result,
                self.key,
                "cpu",
                (time.perf_counter() - start) * 1000.0,
                degraded=True,
                extra={"regions": len(windows_regions), "path": "windows-ocr"},
            )

        return Inference(
            PageResult(regions=[], width=w, height=h, page=page),
            self.key,
            "stub",
            (time.perf_counter() - start) * 1000.0,
            degraded=True,
            extra={"regions": 0, "path": "stub", "hint": "run scripts/fetch_models.py"},
        )


def _connected_boxes(mask: np.ndarray, min_area: int = 40) -> list[tuple[int, int, int, int]]:
    """Row-run based connected components.

    A dependency-free stand-in for cv2.connectedComponents: group horizontal runs of the
    binarised probability map, merge runs that touch vertically, and emit bounding boxes.
    Adequate for the DB detector's output, and it keeps OpenCV off the ARM64 wheel list.
    """
    if mask.ndim != 2 or not mask.any():
        return []
    h, w = mask.shape
    boxes: list[list[int]] = []
    previous: list[tuple[int, int, int]] = []  # (x0, x1, box_index)

    for y in range(h):
        row = mask[y]
        runs: list[tuple[int, int]] = []
        x = 0
        while x < w:
            if row[x]:
                x0 = x
                while x < w and row[x]:
                    x += 1
                runs.append((x0, x))
            else:
                x += 1

        current: list[tuple[int, int, int]] = []
        for x0, x1 in runs:
            hit = None
            for px0, px1, idx in previous:
                if x0 < px1 and px0 < x1:
                    hit = idx
                    break
            if hit is None:
                boxes.append([x0, y, x1, y + 1])
                hit = len(boxes) - 1
            else:
                box = boxes[hit]
                box[0] = min(box[0], x0)
                box[1] = min(box[1], y)
                box[2] = max(box[2], x1)
                box[3] = max(box[3], y + 1)
            current.append((x0, x1, hit))
        previous = current

    return [
        (b[0], b[1], b[2], b[3])
        for b in boxes
        if (b[2] - b[0]) * (b[3] - b[1]) >= min_area
    ]


def _ctc_decode(logits: np.ndarray, charset: list[str]) -> tuple[str, float]:
    """Greedy CTC: argmax per timestep, collapse repeats, drop blanks."""
    if logits.ndim != 2:
        logits = logits.reshape(-1, logits.shape[-1])
    ids = logits.argmax(axis=-1)
    probs = logits.max(axis=-1)
    chars: list[str] = []
    kept: list[float] = []
    previous = -1
    for idx, prob in zip(ids.tolist(), probs.tolist()):
        if idx != previous and idx != 0 and idx < len(charset):
            chars.append(charset[idx])
            kept.append(float(prob))
        previous = idx
    confidence = float(np.mean(kept)) if kept else 0.0
    return "".join(chars), confidence
