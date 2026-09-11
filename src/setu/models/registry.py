"""Model catalogue.

One entry per model with its provenance, licence and expected on-disk artefacts.
``scripts/fetch_models.py`` reads this; so does ``/api/system``, so the running app can
tell a judge exactly which asset is loaded and where it came from.

Asset identifiers on Qualcomm AI Hub are versioned. Rather than hard-code a slug that
may drift, each entry records the AI Hub model *name* plus the export recipe used, and
``fetch_models.py`` resolves it through the ``qai_hub_models`` package at build time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelCard:
    key: str                      # SETU's internal model id, matches ModelSpec.name
    display: str
    source: str                   # "qualcomm-ai-hub" | "open-source"
    upstream: str                 # AI Hub model name or HF repo id
    licence: str
    precision: str
    runtime: str                  # "onnx" | "genie" | "python"
    files: tuple[str, ...] = ()
    notes: str = ""
    #: shell/python recipe used to produce the artefacts, recorded for reproducibility
    export_recipe: str = ""
    languages: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "display": self.display,
            "source": self.source,
            "upstream": self.upstream,
            "licence": self.licence,
            "precision": self.precision,
            "runtime": self.runtime,
            "files": list(self.files),
            "notes": self.notes,
        }


CATALOGUE: dict[str, ModelCard] = {
    "vad": ModelCard(
        key="vad",
        display="Silero VAD",
        source="open-source",
        upstream="snakers4/silero-vad",
        licence="MIT",
        precision="int8",
        runtime="onnx",
        files=("vad.onnx",),
        notes="Gates the ASR model so Whisper only wakes on real speech. This gate is "
        "what makes an all-day always-listening workload affordable.",
        export_recipe="python -m onnxruntime.quantization.preprocess + dynamic int8",
    ),
    "asr": ModelCard(
        key="asr",
        display="Whisper (Small / Base)",
        source="qualcomm-ai-hub",
        upstream="whisper_small_v2",
        licence="MIT (model) / see AI Hub terms for the optimised export",
        precision="int8 encoder + int8 decoder",
        runtime="onnx",
        files=("asr_encoder.onnx", "asr_decoder.onnx", "tokenizer.json"),
        notes="Exported from Qualcomm AI Hub for the Snapdragon X HTP target. Encoder and "
        "decoder are separate graphs so the router can place them independently.",
        export_recipe="python -m qai_hub_models.models.whisper_small_v2.export "
        "--target-runtime onnx --device 'Snapdragon X Elite CRD'",
        languages=("en", "hi", "te", "ta", "bn", "mr", "kn", "ml", "gu", "pa", "ur"),
    ),
    "lid": ModelCard(
        key="lid",
        display="Compact language identifier",
        source="open-source",
        upstream="fasttext lid.176 (compressed)",
        licence="MIT",
        precision="fp16",
        runtime="python",
        files=("lid.176.ftz",),
        notes="Runs on CPU; it is microseconds of work and not worth an NPU context switch. "
        "The router encodes exactly that in its ModelSpec.",
    ),
    "ocr_detect": ModelCard(
        key="ocr_detect",
        display="PaddleOCR DB text detector",
        source="open-source",
        upstream="PaddlePaddle/PaddleOCR ch_PP-OCRv4_det",
        licence="Apache-2.0",
        precision="int8",
        runtime="onnx",
        files=("ocr_detect.onnx",),
        notes="Differentiable-binarisation detector. Fixed 960x960 input so the QNN graph "
        "is fully static, which is what keeps it entirely on the HTP.",
        export_recipe="paddle2onnx --opset 13 then static int8 PTQ with 300 calibration pages",
    ),
    "ocr_recognize": ModelCard(
        key="ocr_recognize",
        display="Multilingual CRNN recogniser",
        source="open-source",
        upstream="PaddleOCR PP-OCRv4 rec (Devanagari + Latin heads)",
        licence="Apache-2.0",
        precision="int8",
        runtime="onnx",
        files=("ocr_recognize.onnx", "charset.txt"),
        notes="Two heads: Latin and Indic. Crops are batched to a fixed 32x320 so a whole "
        "page recognises in a handful of NPU submissions instead of one per line.",
        languages=("en", "hi", "mr", "ta", "te", "bn", "gu", "kn", "ml", "pa", "or"),
    ),
    "embed": ModelCard(
        key="embed",
        display="all-MiniLM-L6-v2",
        source="open-source",
        upstream="sentence-transformers/all-MiniLM-L6-v2",
        licence="Apache-2.0",
        precision="int8",
        runtime="onnx",
        files=("embed.onnx", "tokenizer.json"),
        notes="384-d embeddings for the local RAG store. Indexing runs at BACKGROUND "
        "priority so the router is free to push it to whichever engine is idle.",
        export_recipe="optimum-cli export onnx + static int8 PTQ",
    ),
    "translate": ModelCard(
        key="translate",
        display="OPUS-MT en-mul",
        source="open-source",
        upstream="Xenova/opus-mt-en-mul (Helsinki-NLP)",
        licence="Apache-2.0",
        precision="int8",
        runtime="onnx",
        files=("translate_encoder.onnx", "translate_decoder.onnx", "tokenizer.json"),
        notes="One 111 MB model covering every shipped target language. A distilled "
        "specialist is far cheaper per turn than asking the 3B chat model and more "
        "faithful on clinical register. NLLB-200 and IndicTrans2 are the quality upgrade "
        "path and are documented in docs/MODELS.md.",
        languages=tuple("en hi te ta bn mr kn ml gu pa or ur".split()),
    ),
    "llm": ModelCard(
        key="llm",
        display="Llama 3.2 3B Instruct",
        source="qualcomm-ai-hub",
        upstream="llama_v3_2_3b_chat_quantized",
        licence="Llama 3.2 Community License",
        precision="W4A16",
        runtime="genie",
        files=("genie_config.json", "llama_v3_2_3b.bin", "tokenizer.json"),
        notes="Deployed through the Genie runtime (QAIRT), not ONNX Runtime: Genie owns "
        "the KV cache and the HTP context binary for autoregressive decode. SETU talks to "
        "it through the LlmBackend interface, so the same prompts run on llama.cpp on any "
        "reviewer machine.",
        export_recipe="python -m qai_hub_models.models.llama_v3_2_3b_chat_quantized.export "
        "--device 'Snapdragon X Elite CRD' --skip-inferencing",
    ),
    "tts": ModelCard(
        key="tts",
        display="Piper VITS (Indic voices)",
        source="open-source",
        upstream="rhasspy/piper-voices",
        licence="MIT",
        precision="fp16",
        runtime="onnx",
        files=("tts.onnx", "tts.onnx.json"),
        notes="Voice output is the accessibility story: a user who cannot read the form "
        "also cannot read our answer about the form.",
        languages=("en", "hi", "ne", "mr"),
    ),
}


def card(key: str) -> ModelCard | None:
    return CATALOGUE.get(key)


def catalogue_dicts() -> list[dict[str, object]]:
    return [c.as_dict() for c in CATALOGUE.values()]
