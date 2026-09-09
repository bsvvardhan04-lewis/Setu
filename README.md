<div align="center">

# SETU — Snapdragon Edge Translation & Understanding

**An offline "Jan Seva Kendra in a laptop."**
Point it at a government letter, a hospital prescription, or a bank notice.
It reads the page, explains it in your language, answers your questions, and fills the
form back for you — **entirely on the Snapdragon NPU, with the Wi-Fi turned off.**

*setu* (सेतु) — Sanskrit for **bridge**.

</div>

---

## 1. The problem

India runs on paper that most Indians cannot read.

- ~**1.05 billion** people do not speak English, yet the overwhelming majority of official
  forms, insurance policies, loan documents, and prescriptions are English-first.
- Rural connectivity is intermittent, and cloud AI is **useless in a village office at 11:40 AM
  when the link drops mid-form.**
- The documents people most need help with — Aadhaar, land records, medical reports, salary
  slips — are exactly the documents you must **never** upload to a cloud API.

The existing answer is a human intermediary at a Common Service Centre, at ₹30–₹100 per document,
with a queue, and with your private papers in a stranger's hands.

## 2. The solution

SETU is a fully local, multimodal, multilingual document + voice agent for
**Snapdragon-powered HP PCs**. Everything below happens with the network adapter disabled:

| You do | SETU does | Runs on |
| --- | --- | --- |
| Put a page under the webcam, or drop a PDF | Detects + reads text, incl. Devanagari/Telugu/Tamil | **NPU** (detector + recognizer) |
| Say "इसमें क्या लिखा है?" | Wake → VAD → ASR → language ID | **NPU** (Whisper) |
| Ask anything about the page | Grounded RAG answer with page-region citations | **NPU** (embeddings) + **NPU** (LLM) |
| Say "मेरा नाम रमेश कुमार है" | Fills the matching field, transliterates, validates | LLM + rule layer |
| Listen | Speaks the answer back in your language | CPU/NPU (TTS) |

Nine models. One pipeline. Zero bytes leave the machine.

## 3. Why this *needs* Snapdragon (and would not work otherwise)

This is the part most submissions hand-wave. Ours is measured, not asserted.

1. **The workload is sustained, not bursty.** A CSC operator runs this 6–8 hours a day on
   battery. Continuous ASR + OCR + LLM on the CPU flattens an X-series battery in ~2 hours and
   thermally throttles the whole machine. On the Hexagon NPU the same pipeline is a background
   citizen. `docs/BENCHMARKS.md` reports the measured mW.
2. **The privacy constraint is absolute.** Land records and medical reports legally and
   ethically cannot be sent to a cloud endpoint. On-device is not an optimization here — it is
   the product requirement.
3. **We exploit heterogeneity, not just "the NPU".** SETU ships **Hexa-Router**, a
   power- and thermal-aware scheduler that places each of the nine models on NPU / GPU / CPU
   per-inference, learns each model's real cost on *this* machine with an EWMA cost model, and
   re-plans when you unplug the charger. See `src/setu/runtime/router.py`.
4. **Pipeline overlap.** OCR of page *n+1* runs on the NPU while the LLM prefills page *n*,
   because they are different engines. Wall-clock for a 6-page document drops materially versus
   naive sequential execution.

## 4. Architecture

```
                       ┌──────────────────────────────────────────┐
  webcam / PDF ───────►│  DocAgent    detect → recognize → layout │
  microphone  ───────►│  VoiceAgent  VAD → ASR → LID             │──┐
  typed text  ───────►│  FormAgent   schema → slot-fill → verify │  │
                       └──────────────────────────────────────────┘  │
                                       │                             │
                              ┌────────▼─────────┐          ┌────────▼────────┐
                              │  Local RAG store │          │  LLM (Genie /   │
                              │  sqlite + MiniLM │◄────────►│  ORT-GenAI QNN) │
                              └──────────────────┘          └─────────────────┘
                                       │
                       ┌───────────────▼────────────────────────────┐
                       │   HEXA-ROUTER   placement + telemetry      │
                       │   QNN(HTP) │ DirectML(Adreno) │ CPU(oneDNN)│
                       └────────────────────────────────────────────┘
```

Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## 5. Models

Nine models, all from **Qualcomm AI Hub** or permissively-licensed open source.
Exact assets, licences, quantisation and provenance: [`docs/MODELS.md`](docs/MODELS.md).

| Role | Model | Source | Precision |
| --- | --- | --- | --- |
| Speech recognition | Whisper Small / Base | Qualcomm AI Hub | INT8 encoder + decoder |
| Voice activity | Silero VAD | open source (MIT) | INT8 |
| Text detection | PaddleOCR DB detector | open source (Apache-2.0) | INT8 |
| Text recognition | PaddleOCR / TrOCR recogniser | AI Hub / open source | INT8 |
| Language ID | fastText-lid compact | open source (MIT) | FP16 |
| Reasoning / chat | Llama 3.2 3B Instruct | Qualcomm AI Hub (Genie) | W4A16 |
| Translation | IndicTrans2 distilled | AI4Bharat (MIT) | INT8 |
| Embeddings | all-MiniLM-L6-v2 | open source (Apache-2.0) | INT8 |
| Speech synthesis | Piper VITS (Indic voices) | open source (MIT) | FP16 |

## 6. Run it

```bash
git clone <this-repo> && cd setu
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
python -m setu.server.app
```

Open <http://127.0.0.1:8756>.

SETU runs on **any** Windows/macOS/Linux machine out of the box — the runtime layer degrades
gracefully to CPU and to deterministic stub backends, so reviewers can exercise the whole app
without a Snapdragon device. On a Snapdragon X / X2 PC with QAIRT installed it lights up the
NPU automatically; `GET /api/system` tells you exactly which engine every model landed on.

Snapdragon first-time setup: [`scripts/setup_windows_arm64.ps1`](scripts/setup_windows_arm64.ps1).

## 7. Benchmarks

```bash
python scripts/run_bench.py --iters 30 --report docs/BENCHMARKS.md
```

The harness measures per-model p50/p90 latency **and real energy per inference**, sampled from
the Windows `root\WMI BatteryStatus.DischargeRate` counter — actual milliwatts off the battery,
not a estimate. It emits the NPU-vs-CPU comparison table that backs every claim in this README.

## 8. Repository map

| Path | What lives there |
| --- | --- |
| `src/setu/runtime/` | EP discovery, **Hexa-Router**, power telemetry, session cache |
| `src/setu/models/` | One adapter per model role, each with a graceful fallback |
| `src/setu/pipeline/` | DocAgent / VoiceAgent / FormAgent orchestration |
| `src/setu/store/` | Local sqlite vector store (no server, no network) |
| `src/setu/server/` | FastAPI backend + static local UI |
| `src/setu/bench/` | Measurement harness |
| `docs/` | Architecture, models, benchmarks, demo script, pitch |

## 9. Licence

MIT. See `LICENSE`. Model licences are itemised per-model in `docs/MODELS.md`.
