<div align="center">

# SETU

### The doctor speaks English. The patient thinks in Hindi.
### SETU checks they actually understood — offline, on the NPU.

**One engine, three settings: clinic, classroom, government counter.**

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
| **Or scan** | Photograph a prescription or notice — the extracted text joins the session exactly as speech does, and produces the same take-home card. |
| **Bridge** | Each turn is transcribed, language-identified, and shown to the patient in their own language. |
| **Flag jargon** | "od", "fasting", "lipid profile", "chronic" — caught as spoken, glossed in plain words the doctor can read aloud. |
| **Build the plan** | Instructions are extracted as they are given: medicines, tests, follow-up, red flags, lifestyle. |
| **Teach back** | The patient is asked to repeat the plan in their own words. SETU checks it against what was actually said. |
| **Take-home card** | A printable page at `/card/<visit>` — large type, high contrast, red flags first, with a read-aloud button. The patient's only artefact, so it is typeset rather than dumped as text. |

The moment that matters is the fifth one:

> **2 of 5 understood**
> The patient did not repeat these back, and they are the ones that matter:
> ✗ *If you get chest pain or breathlessness, come immediately to the emergency.*

The doctor learns this **while the patient is still in the room.**

## Three settings, one engine

The clinic is the flagship, not the limit. Strip the vocabulary away and every one of
these is the same problem:

> Two people. A language or knowledge gap. One side issues instructions.
> The other is expected to act on them. **Nobody checks that the instructions landed.**

So SETU ships one engine and three domain profiles
([`domains.py`](src/setu/pipeline/domains.py)). Switch the mode in the header:

| Setting | Expert → learner | What it catches | Ordered by |
| --- | --- | --- | --- |
| **Clinic** | doctor → patient | "od", "fasting", "lipid profile" | red flags first |
| **Classroom** | teacher → student | "weightage", "internal", "plagiarism" | exam dates first |
| **Counter** | officer → citizen | "self attested", "BPL", "acknowledgement" | deadline first |

Each profile supplies a jargon lexicon, instruction categories ordered by consequence,
teach-back questions, and card headings. **Adding a fourth setting — a courtroom, a bank,
a panchayat — means writing one `Domain`, not another agent.** That is the whole argument
for the abstraction, and `tests/test_domains.py` enforces it: if the clinic's vocabulary
ever leaks into the classroom, the suite fails.

The results speak for themselves. Same code, three sessions:

- **Clinic** — patient repeated the tablet, missed *"chest pain means come immediately"*
- **Classroom** — student repeated the reading, missed the **assignment deadline**, the
  **internal test**, and the **plagiarism warning**
- **Counter** — citizen repeated the ₹50 fee, missed the **last date** and the **documents**

## Who owns the laptop

**Not the patient.** The device belongs to the clinic, the school, or the office. The
person being helped never touches it, installs nothing, and needs no phone — their
interface is **a printed card**, which is exactly why the take-home output is printable
text rather than an app screen.

The economics work because one machine serves everyone who walks in. A clinic seeing ~40
people a day over a three-year life is roughly 36,000 sessions; a ₹90,000 laptop across
that is about **₹2.50 per session** (illustrative, but the order of magnitude holds).
A Common Service Centre agent charges ₹30–100 to help with a single document.
**The device was never the expensive part. The human intermediary is.**

Realistic buyers today: private clinics in tier-2/3 cities, district hospitals, medical
and engineering colleges, NGO and CSR health programmes, and telemedicine providers — all
of whom already buy PCs, and all of whom face the language gap daily.

## Does it actually work?

Every other measurement in this repo is about speed. This one asks whether the product's
central claim is true — and it is the question most submissions never ask about themselves.

`python scripts/run_eval.py --compare` runs the real pipeline over a labelled corpus of
sessions ([`corpus.py`](src/setu/bench/corpus.py)) and reports how well the comprehension
check separates a learner who understood from one who did not.

| Metric | Result | Why it is the right metric |
| --- | ---: | --- |
| **Dangerous misses** | **0** | Said "understood" when they had not. The error that sends a patient home. |
| Recall on missed items | **100%** | Of instructions truly not restated, how many we caught. |
| Precision | 86.7% | Of what we flagged, how much was genuinely missed. |

The two errors are not symmetric, so they are not averaged together. A false alarm costs a
doctor five seconds of repetition. A dangerous miss costs a patient. **Thresholds were
chosen at the operating point that keeps dangerous misses at zero, not the one that
maximises F1** — at a lower threshold F1 is higher and one real miss slips through.

The evaluation also **overturned a design decision.** Replacing keyword matching with
sentence embeddings made the check *worse* (precision 81% → 76%): cosine scores an
instruction against a restatement lower than expected, because an imperative and a
first-person promise are grammatically asymmetric. The two signals failed on different
items, so the shipped scorer unions them — **86.7%, better than either alone.** Full
working: [`docs/EVALUATION.md`](docs/EVALUATION.md).

## What is kept, and where

Consultations are written to a sqlite file under your own data directory after every turn,
so a restart does not lose a visit and a take-home card URL handed to a patient keeps
working.

That deserves stating precisely, because it changes the privacy claim. *Nothing leaves the
device* stays true — there is no outbound call in any path. But **"does not leave" is not
"is not kept"**, and a consultation is protected health information sitting on a laptop.
So:

- everything is in **one file** you can delete with the file manager, no tooling required
- **deleting a visit cascades** — turns, plan items and glosses go with it, no orphans
- **retention is bounded by default**: 90 days, then automatic deletion. A clinic that
  never thinks about retention does not silently accumulate years of patient conversations.
  `SETU_RETENTION_DAYS=0` keeps everything, which is a decision someone has to make on
  purpose.
- **Past visits** in the header lists what is stored and deletes any of it in one click

`GET /api/system` reports the file path, the retention window and how many consultations
are held, so what is kept is inspectable rather than implied.

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
| Translation | OPUS-MT en-mul + en-hi | Helsinki-NLP (Apache-2.0) | INT8 |
| Embeddings | all-MiniLM-L6-v2 | open source (Apache-2.0) | INT8 |
| Text detection | PaddleOCR DB | open source (Apache-2.0) | INT8 |
| Text recognition | PP-OCRv4 rec | open source (Apache-2.0) | INT8 |
| Language ID | Unicode-script identifier | in-repo | — |
| Speech synthesis | Piper VITS | open source (MIT) | FP16 |

**Twelve languages are supported; eight of them translate.** Hindi, Telugu, Tamil,
Kannada, Malayalam, Gujarati, Odia and Urdu have a working translation checkpoint.
Marathi, Bengali and Punjabi are transcribed, language-identified, jargon-glossed and
plan-extracted like the rest, but the available checkpoints return word salad for them, so
they are excluded from the translatable set and the UI labels them **"(no translation)"**.
Shipping a language that silently returns English would be worse than not offering it.

Hindi routes to a dedicated `en-hi` checkpoint because the multilingual model cannot do it
— a failure that emits perfectly well-formed Devanagari and therefore passes any automated
"is this the right script?" check. It had to be read to be caught.

## Run it

**Double-click `START_SETU.bat`.** It creates the environment on first run, prints what
your machine can actually do, and opens the app. Nothing to install but Python.

Or by hand:

```bash
git clone <this-repo> && cd setu
python -m venv .venv && .venv\Scriptsctivate
pip install -e . "onnxruntime>=1.22"
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
- **Speech synthesis refuses rather than guesses.** The Piper voice is on disk, but Indic
  Piper voices are `espeak`-typed: they expect IPA phonemes from espeak-ng, not characters.
  Driving one from characters does not fail — it speaks confident, fluent-sounding noise.
  For a tool whose purpose is that a patient understood an instruction, that is worse than
  silence, so SETU delegates to the client's own on-device voice and says why.
- **Document capture uses the OS engine, not ours.** Our quantised OCR graphs are not
  exported yet, so scanning falls to `Windows.Media.Ocr` — genuinely offline and accurate,
  but CPU-only rather than NPU-accelerated, and limited to the OCR language packs the
  machine has installed. It is reported as such rather than claimed as our own.
- **This is a comprehension aid, not a medical device.** It does not diagnose, does not
  recommend treatment, and never overrides the clinician. Every care-plan item is extracted
  verbatim from what the doctor said.
- The jargon lexicon is curated for an Indian OPD and is deliberately small; the LLM only
  fills gaps for terms actually spoken.

## Licence

MIT — see [`LICENSE`](LICENSE). Model licences are itemised per model in `docs/MODELS.md`.
