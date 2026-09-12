# Submission — Snapdragon® AI Lab Build & Present Challenge

**Project:** SETU — an offline consultation companion that verifies patient comprehension
**Category:** AI use case development for Snapdragon-powered HP PCs
**Deadline:** 30 September 2026, 11:59 PM IST

---

## 1. The problem

Indian clinical consultations routinely happen across a language gap. Doctors are trained
and think in English; most patients are not. The patient nods, goes home, and takes the
medicine wrong — or misses the warning sign that should have brought them straight back.

This is not a translation problem. A phone can translate. It is a **comprehension** problem,
and its defining feature is that **nobody in the room finds out it happened.** The WHO puts
adherence to long-term therapy in developing countries at roughly 50%; "the patient never
understood the instruction" is a large and unglamorous share of that.

## 2. The solution

A laptop on the doctor's desk listens to the consultation with the network disabled, and:

0. **reads the page** — a photographed prescription or notice goes through the same
   pipeline as speech, producing the same care plan and take-home card
1. **transcribes and bridges** each turn into the patient's language
2. **flags jargon** as it is spoken — "od", "fasting", "lipid profile" — with plain glosses
3. **extracts the care plan** — medicines, tests, follow-up, red flags, lifestyle
4. **runs a teach-back check** — the patient repeats the plan, SETU verifies it against what
   was actually said
5. **prints a take-home card** in the patient's language, ordered by consequence

The output that matters is step 4, delivered while the patient is still in the room:

> **2 of 5 understood.** The patient did not repeat back: *"if you get chest pain or
> breathlessness, come immediately to the emergency."*

## 3. Technical implementation

### Nine models, four from a heterogeneous placement policy

Whisper Small v2 and Llama 3.2 3B come from **Qualcomm AI Hub**; Silero VAD, MiniLM,
IndicTrans2, PaddleOCR and Piper from open source. Full provenance, licences and export
recipes: [`MODELS.md`](MODELS.md).

### Hexa-Router — the contribution

Most on-device work hard-codes "everything on the NPU". Hexa-Router
([`router.py`](../src/setu/runtime/router.py)) decides placement **per model, per request**
from: the model's deployment envelope, live power and thermal state, request priority
(interactive / streaming / background), and **a cost model learned on the specific machine**
from measured latency and marginal energy.

It applies hysteresis, because rebuilding a QNN session costs hundreds of milliseconds and
thrashing on noise would cost more than any placement gain. It steers work off a throttling
NPU. It selects the QNN HTP performance mode per placement — `burst` for interactive turns,
`sustained_high_performance` for all-day streaming ASR, `power_saver` under battery
pressure.

### Measurement that does not lie

Two independent evidence paths:

- **[`AIHUB_PROFILE.md`](AIHUB_PROFILE.md)** — each graph compiled as a QNN context binary
  and profiled on **real Snapdragon silicon** via AI Hub's cloud devices, reporting
  inference time, peak memory, and the fraction of layers that stayed on the NPU rather than
  falling back. **Every row links to its job on Qualcomm's own site.**
- **[`BENCHMARKS.md`](BENCHMARKS.md)** — local latency and energy, sampled from the Windows
  battery discharge counter with a **measured idle baseline subtracted**, so the figure is a
  model's marginal cost rather than the whole laptop's draw.

Two bugs found by measuring rather than asserting are documented in
[`ARCHITECTURE.md`](ARCHITECTURE.md): the telemetry that was adding 450 ms to every
inference, and the energy column that was overstating model cost by an order of magnitude.
Both are fixed and both have regression tests.

## 4. Application use case and innovation

The innovation is not translation. It is **comprehension verification**, and the design
principle behind it is distrust of the model:

- Jargon comes from a curated lexicon and cannot hallucinate a term that was not spoken.
- Care-plan extraction is deterministic; the LLM runs afterwards and **may only add, never
  delete**. A dropped red-flag instruction is the one failure mode that could hurt someone.
- Teach-back inverts this: the LLM may only **downgrade** an item to missed, never upgrade a
  missed item to covered. Optimism there would defeat the feature.
- When the patient answers in a language the system cannot bridge, SETU reports **"could not
  verify"** rather than scoring them as having understood nothing — a false negative would
  send a doctor away with the wrong conclusion.

## 5. Deployment and accessibility

- **Fully offline.** Loopback-bound server, no CDN in the UI, no outbound call in any path.
- **Durable, and honest about it.** Consultations persist to a local sqlite file so a
  restart does not lose a visit. Because that means protected health information is at
  rest on the machine, retention is bounded to 90 days by default, deletion is a
  first-class control in the interface, and `/api/system` reports exactly what is held
  and where.
- **Runs on any machine.** Every model has a labelled fallback; the entire product is
  exercisable on a reviewer's laptop with zero weights downloaded. Anything on a fallback
  path is marked `degraded` in the API and on screen — a stub is never passed off as real.
- **One command on Snapdragon.** `scripts\setup_windows_arm64.ps1` installs the QNN
  execution provider and verifies the result; `python -m setu.cli doctor` prints exactly
  what the machine can do.
- **Accessible by design.** The users are people who cannot read the form in front of them.
  Everything is voice-in and voice-out, and the take-home card prints to a clinic's existing
  thermal printer.

## 6. Verify this submission

```bash
pip install -e .
pytest -q                              # 65 tests, no assets required
python -m setu.cli doctor              # what this machine has
python -m setu.server.app              # then open 127.0.0.1:8756
python scripts/run_bench.py            # local latency + marginal energy
python scripts/aihub_profile.py --all  # real Snapdragon hardware
```

## 7. What is honestly not finished

- Whisper is weaker on Telugu, Kannada and Odia than on Hindi. The stated mitigation is
  IndicWhisper through the same AI Hub export recipe.
- Speech synthesis does not run the Piper voice, because Indic Piper voices are
  `espeak`-typed and no espeak-ng phonemiser is installable here; driving one from
  characters produces fluent noise rather than an error. Synthesis therefore uses Windows
  SAPI — real, offline, OS-provided — for languages with an installed voice, and delegates
  to the client's own voice otherwise. Both the engine and the per-language coverage are
  reported rather than implied.
- IndicTrans2 needs an ONNX export step that is documented but not scripted.
- Document capture works, but through `Windows.Media.Ocr` rather than our own quantised
  graphs: genuinely offline and accurate, CPU-only rather than NPU-accelerated, and limited
  to the OCR language packs installed on the machine. The ONNX detector and recogniser are
  specified in the registry but not yet exported.
- Marathi, Bengali and Punjabi have no working translation checkpoint. They are excluded
  from the translatable set and the interface labels them "(no translation)"; everything
  else in the pipeline still serves them.
- SETU is a comprehension aid, **not a medical device**. It does not diagnose, does not
  recommend treatment, and never overrides the clinician.
