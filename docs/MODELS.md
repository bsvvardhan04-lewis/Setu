# Models

Every model SETU uses, where it came from, what licence it carries, and how to reproduce
the artefact. The machine-readable version of this table is
[`src/setu/models/registry.py`](../src/setu/models/registry.py), which the running app
serves at `/api/system` — so what the app reports and what this document claims cannot
drift apart.

## From Qualcomm AI Hub

### `asr` — Whisper Small v2

- **AI Hub model:** `whisper_small_v2`
- **Licence:** MIT (OpenAI Whisper); the optimised export is subject to AI Hub's terms
- **Precision:** INT8 encoder + INT8 decoder
- **Runtime:** ONNX Runtime with the QNN execution provider

```bash
python -m qai_hub_models.models.whisper_small_v2.export \
    --device "Snapdragon X Elite CRD" --target-runtime onnx
```

Encoder and decoder are kept as separate graphs so Hexa-Router can place them
independently — the encoder is the dominant cost and benefits most from the NPU, while the
decoder is small and latency-sensitive.

**Known weakness, stated plainly.** Whisper's multilingual quality is good for Hindi and
Hinglish code-switching, noticeably weaker for Telugu, Kannada and Odia. The mitigation is
**IndicWhisper** (AI4Bharat, Vistaar project) — a Whisper fine-tune on Indian-language
corpora. It shares Whisper's architecture, so it exports through the same AI Hub recipe with
the checkpoint swapped. This is exactly the "significantly modified to add AI models from
Qualcomm AI Hub or other open-source platforms" path the challenge describes.

### `llm` — Llama 3.2 3B Instruct

- **AI Hub model:** `llama_v3_2_3b_chat_quantized`
- **Licence:** Llama 3.2 Community License
- **Precision:** W4A16
- **Runtime:** **Genie (QAIRT)**, not ONNX Runtime

```bash
python -m qai_hub_models.models.llama_v3_2_3b_chat_quantized.export \
    --device "Snapdragon X Elite CRD" --skip-inferencing
```

Autoregressive decode needs a runtime that owns the KV cache and a pre-compiled HTP context
binary, which is what Genie provides. SETU talks to it behind the `LlmBackend` interface, so
the same prompts run unchanged on llama.cpp on any reviewer's machine.

### The portable Whisper, and what it costs

`scripts/fetch_models.py --model asr` fetches **whisper-base multilingual INT8** from
`onnx-community/whisper-base` instead. That is deliberate: it means the speech path is
genuinely running on any reviewer's machine, not just architecturally ready for one.

Verified round-trip on synthesised speech (`assets/sample_doctor.wav`, produced by Windows
TTS so the fixture carries no third-party audio):

> **spoken:** "Take the amlodipine tablet once at night. If you get chest pain, come
> immediately to the emergency."
> **transcribed:** "Take the amlo**typing** tablet once at night. If you get chest pain,
> come immediately to the emergency."

Everything clinically load-bearing survives — the timing, the red flag, the instruction —
and **the drug name does not.** That is exactly the failure profile you would predict for
a *base* model on domain-specific vocabulary, and it is the concrete argument for the
larger `whisper_small_v2` AI Hub export on Snapdragon: the NPU is what makes affording the
bigger model possible in a sustained, all-day workload.

Three bugs in this path were invisible until real audio ran through it, and all three are
now regression-tested in `tests/test_asr.py`:

1. the mel spectrogram produced 2998 frames where the encoder demands exactly 3000
2. the session cache keyed on model name alone, handing the decoder the encoder's session
3. the filterbank used the HTK mel scale rather than Whisper's Slaney scale — which runs
   without any error and transcribes noise

## Open source

| Key | Model | Upstream | Licence | Precision |
| --- | --- | --- | --- | --- |
| `vad` | Silero VAD | `onnx-community/silero-vad` | MIT | INT8 |
| `embed` | all-MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | INT8 |
| `translate` | OPUS-MT en-mul | `Xenova/opus-mt-en-mul` | Apache-2.0 | INT8 |
| `translate_hi` | OPUS-MT en-hi | `Xenova/opus-mt-en-hi` | Apache-2.0 | INT8 |
| `ocr_detect` | PP-OCRv4 DB detector | PaddlePaddle/PaddleOCR | Apache-2.0 | INT8 |
| `ocr_recognize` | PP-OCRv4 recogniser | PaddlePaddle/PaddleOCR | Apache-2.0 | INT8 |
| `tts` | Piper VITS (Indic voices) | `rhasspy/piper-voices` | MIT | FP16 |

Fetch them with:

```bash
python scripts/fetch_models.py --list
python scripts/fetch_models.py --all
```

### Translation: what works, and what does not

Measured across all eleven target languages on the same two-sentence input:

| Works | Does not |
| --- | --- |
| Hindi (dedicated `en-hi`), Telugu, Tamil, Kannada, Malayalam, Gujarati, Odia, Urdu | Marathi, Bengali, Punjabi |

The failures are the instructive part. `opus-mt-en-mul` **accepts** `>>mar<<`, `>>ben<<`
and `>>pan<<` and returns fluent-looking Devanagari word salad, or echoes the source
unchanged. Nothing raises, and the output passes a script check — so it is excluded by
hand, and the UI says "(no translation)" rather than letting a live demo find out.

Quality on the languages that do work is good on structure and weak on medical nouns:
"tablet" becomes "table" in several of them. That is the same failure profile as Whisper
on "amlodipine", and the same argument: a bigger model is what fixes it, and the NPU is
what makes a bigger model affordable in an all-day workload.

**NLLB-200 and IndicTrans2** are the quality upgrade path. NLLB-200-distilled-600M covers
all twelve properly; it was not shipped here because it is 660 MB and non-commercial
(CC-BY-NC-4.0). IndicTrans2 (AI4Bharat, MIT) is the right choice for a commercial
deployment and needs an ONNX export step.

### Why a separate translator when we already have an LLM

A 200M distilled specialist beats asking a 3B chat model to translate: lower latency, far
lower power, and more faithful on official and clinical register. A bank notice or a
discharge summary is not conversational Hindi, and a general chat model tends to smooth it
into something friendlier and less exact. For a care plan, exactness is the product.

### `lid` — language identification

Not a downloaded model. Dominant-Unicode-script detection, implemented in
[`models/lang.py`](../src/setu/models/lang.py). It is microseconds of work and correct for
the overwhelming majority of Indian document and transcript text. Hexa-Router encodes
exactly this in its `ModelSpec`: `lid` is CPU-only and never placed on an accelerator,
because the context switch would cost more than the operation.

## Quantisation notes

Static shapes are not incidental. A dynamic axis is the most common reason a layer silently
falls back off the NPU, which is why:

- the OCR detector is frozen at 960×960
- line crops are batched to a fixed 32×32×320 (one NPU submission per 32 lines, not 32
  submissions)
- the Whisper encoder takes a fixed 30-second mel window

`scripts/aihub_profile.py` exists to catch it when this goes wrong: it reports the fraction
of layers that actually stayed on the NPU. A model that compiles but scatters half its
layers back to CPU is not running on the NPU in any meaningful sense, and the whole premise
of this project depends on knowing the difference.

## Reproducing every artefact

```bash
pip install "qai-hub-models[whisper_small_v2]" huggingface-hub onnx onnxruntime
python scripts/fetch_models.py --all --device "Snapdragon X Elite CRD"
python -m setu.cli doctor            # confirm the app can see them
python scripts/aihub_profile.py --all  # profile on real Snapdragon silicon
```
