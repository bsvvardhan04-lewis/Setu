<div align="center">

# SETU

### The doctor speaks English. The patient thinks in Hindi.
### SETU makes sure they actually understood — offline, on the NPU.

*setu* (सेतु) — Sanskrit for **bridge**

Built for the Snapdragon® AI Lab Build & Present Challenge

</div>

---

## The problem nobody measures

A patient at a district hospital is told, in English, that they have hypertension, to take
amlodipine 5 mg *od*, to get a fasting lipid profile, and to come back immediately if they
get chest pain.

They nod. They say *"haan ji."* They go home.

They did not understand "od". They did not understand "fasting". And they have no idea
that chest pain means *return now* rather than *wait for the follow-up*.

This is not a translation problem — a phone can translate. It is a **comprehension**
problem, and nobody in the room finds out it happened. The WHO puts adherence to long-term
therapy in developing countries at around **50%**, and "the patient never understood the
instruction" is a large, unglamorous share of that.

## What SETU does

A laptop sits on the doctor's desk and listens to the whole consultation. With the Wi-Fi off.

| Stage | What happens |
| --- | --- |
| **Listen** | VAD gates the mic; only real speech reaches Whisper. Continuous, all day, on battery. |
| **Bridge** | Each turn is transcribed, language-identified, and shown to the patient in their own language. |
| **Flag jargon** | "od", "fasting", "lipid profile", "chronic" — caught as spoken, glossed in plain words the doctor can read aloud. |
| **Build the plan** | Instructions are extracted as they are given: medicines, tests, follow-up, red flags, lifestyle. |
| **Teach back** | The patient is asked to repeat the plan in their own words. SETU checks it against what was actually said. |
| **Take-home card** | Printed, in their language, ordered by what will hurt them if they forget it. |

The moment that matters is the fifth one:

> **2 of 5 understood**
> The patient did not repeat these back, and they are the ones that matter:
> ✗ *If you get chest pain or breathlessness, come immediately to the emergency.*

The doctor learns this **while the patient is still in the room.**

## Why this needs Snapdragon

Three reasons, and only the third is a performance argument.

**1. It is legally and ethically un-cloudable.** A consultation is protected health
information. "Send the audio to a data centre" is not a design choice we get to make.
On-device is the requirement, not the optimisation.

**2. It has to work where the connection does not.** A PHC with an intermittent link is
the normal case, not the edge case.

**3. It is a sustained workload, not a bursty one.** Continuous ASR for a six-hour clinic
day is precisely the thing an NPU exists for and a CPU cannot afford. This is measurable,
and we measure it rather than asserting it — see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
and [`docs/AIHUB_PROFILE.md`](docs/AIHUB_PROFILE.md).

## Hexa-Router

Most on-device demos hard-code *"put everything on the NPU."* That is wrong for a real
workload: the NPU is shared and finite, some ops fall back regardless, and the right answer
changes the moment the charger comes out.

**Hexa-Router** ([`src/setu/runtime/router.py`](src/setu/runtime/router.py)) decides
placement per model, per request, from four inputs:

- what each model can run on at all, and its rough cost shape
- live power state — AC vs battery, charge level, thermal pressure
- request priority — a human waiting for a caption, or a background index
- **a cost model it learns on your machine**, from measured latency and energy, so routing
  improves the longer the app runs

Placement is sticky: rebuilding a QNN session costs hundreds of milliseconds, so migration
needs to clear a margin before it is worth paying for. The UI shows every decision live.

## Models

Nine models. Sources, licences, quantisation and export recipes are itemised in
[`docs/MODELS.md`](docs/MODELS.md).

| Role | Model | Source | Precision |
| --- | --- | --- | --- |
| Speech recognition | Whisper Small v2 | **Qualcomm AI Hub** | INT8 encoder + decoder |
| Reasoning | Llama 3.2 3B Instruct | **Qualcomm AI Hub** (Genie) | W4A16 |
| Voice activity | Silero VAD | open source (MIT) | INT8 |
| Translation | IndicTrans2 distilled 200M | AI4Bharat (MIT) | INT8 |
| Embeddings | all-MiniLM-L6-v2 | open source (Apache-2.0) | INT8 |
| Text detection | PaddleOCR DB | open source (Apache-2.0) | INT8 |
| Text recognition | PP-OCRv4 rec | open source (Apache-2.0) | INT8 |
| Language ID | Unicode-script identifier | in-repo | — |
| Speech synthesis | Piper VITS | open source (MIT) | FP16 |

Twelve languages: Hindi, Telugu, Tamil, Bengali, Marathi, Kannada, Malayalam, Gujarati,
Punjabi, Odia, Urdu, English.

## Run it

```bash
git clone <this-repo> && cd setu
python -m venv .venv && .venv\Scripts\activate
pip install -e .
python -m setu.server.app
```

Open <http://127.0.0.1:8756> and press **Replay sample visit**.

**It runs on any machine.** Every model has a labelled fallback, so the whole product —
transcript, jargon glossing, care plan, teach-back, take-home card — works on a reviewer's
laptop with no weights downloaded and no Snapdragon hardware. Anything running on a
fallback is marked `degraded` in the API and on screen. SETU never passes a stub off as a
real inference.

On a Snapdragon PC, `scripts\setup_windows_arm64.ps1` installs the QNN execution provider
and the NPU lights up automatically. `python -m setu.cli doctor` prints exactly what your
machine can do.

## Verify the claims

```bash
python -m setu.cli doctor           # what this machine actually has
python scripts/run_bench.py         # local latency + marginal energy per inference
python scripts/aihub_profile.py --all   # real Snapdragon hardware, via AI Hub
```

`aihub_profile.py` compiles each graph as a QNN context binary, runs it on physical
Snapdragon silicon in Qualcomm's cloud, and reports what fraction of layers actually stayed
on the NPU rather than falling back to CPU — **with a link to each job on Qualcomm's own
site**, so every number in this repo can be checked independently.

Energy is measured from the Windows `BatteryStatus.DischargeRate` counter with a measured
idle baseline subtracted, so the figures are a model's *marginal* cost, not the whole
laptop's draw.

## Repository

| Path | What lives there |
| --- | --- |
| `src/setu/runtime/` | Hexa-Router, EP discovery, power telemetry, session cache |
| `src/setu/models/` | Nine adapters, each with a labelled fallback |
| `src/setu/pipeline/` | ConsultAgent, take-home card, plus document and form agents |
| `src/setu/server/` | FastAPI backend and the single-file offline UI |
| `src/setu/bench/` | Latency and energy harness |
| `scripts/` | AI Hub profiling, model fetch, Snapdragon setup |
| `docs/` | Architecture, model provenance, benchmarks, demo script |

## Honest limitations

- **Whisper is weaker on Telugu, Kannada and Odia than on Hindi.** The mitigation is
  AI4Bharat's IndicWhisper through the same AI Hub export path; see `docs/MODELS.md`.
- **Teach-back across languages needs the translation model.** Without it SETU reports
  *"could not verify"* rather than scoring the patient as having understood nothing —
  a false negative here would send a doctor away with the wrong conclusion.
- **This is a comprehension aid, not a medical device.** It does not diagnose, does not
  recommend treatment, and never overrides the clinician. Every care-plan item is extracted
  verbatim from what the doctor said.
- The jargon lexicon is curated for an Indian OPD and is deliberately small; the LLM only
  fills gaps for terms actually spoken.

## Licence

MIT — see [`LICENSE`](LICENSE). Model licences are itemised per model in `docs/MODELS.md`.
