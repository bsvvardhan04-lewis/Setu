# Architecture

SETU is four layers. Each one exists to solve a problem the layer below it cannot.

```
  microphone ──► VoiceAgent ──► ConsultAgent ──► TakeHomeCard
                  VAD gate       jargon · plan
                  ASR · LID      teach-back
                                      │
                                      ▼
                            ┌───────────────────┐
                            │   MODEL ADAPTERS  │  nine, each with a
                            │  asr vad ocr lid  │  labelled fallback
                            │  embed llm tts …  │
                            └─────────┬─────────┘
                                      │
                            ┌─────────▼─────────┐
                            │    HEXA-ROUTER    │  placement policy
                            │  + SessionCache   │  + learned cost model
                            │  + PowerTelemetry │
                            └─────────┬─────────┘
                                      │
              ┌───────────────┬───────┴───────┬───────────────┐
           QNN / HTP      DirectML          CPU            stub
          (Hexagon NPU)   (Adreno)      (oneDNN/ARM64)   (no hardware)
```

## Layer 1 — Runtime

### `providers.py` — what this machine can actually do

SETU never assumes a Snapdragon device. It asks onnxruntime what execution providers exist
and maps them onto four logical devices: `NPU`, `GPU`, `CPU`, `STUB`. The list always ends
with `STUB`, so no call site ever has to handle "nothing available".

This is what lets the same wheel run on a reviewer's x64 ThinkPad and on an HP OmniBook
with a Hexagon NPU, with no conditional code above this layer.

### `router.py` — Hexa-Router

Placement is a decision, not a constant. Inputs:

| Input | Why it matters |
| --- | --- |
| `ModelSpec.allowed` | Language ID is microseconds of work; an NPU context switch costs more than the op |
| `PowerState` | On battery at 15%, latency stops being the thing worth optimising |
| `Priority` | INTERACTIVE (human waiting) · STREAMING (must keep up) · BACKGROUND (nearly free) |
| Learned cost | EWMA over measured latency and marginal energy, **per model per device, on this machine** |

The learned term is the interesting one. Priors are conservative guesses; within a handful
of real inferences the EWMA replaces them. If a model's NPU path turns out slower than the
prior claimed — quantisation fallbacks, an unsupported op — the router notices and moves it,
without anyone editing a table.

**Hysteresis.** Rebuilding a QNN session means re-finalising the graph, which costs hundreds
of milliseconds. So a candidate must beat the incumbent by a margin (20%, or 45% for
resident models like the LLM) before migration is worth paying for. Without this the router
would thrash between engines on noise.

**Thermal backstop.** Above 0.75 thermal pressure, work is steered off the NPU entirely
unless nothing else can run it.

**Performance mode.** On the NPU the router also picks the QNN HTP mode: `burst` for
interactive turns, `sustained_high_performance` for the streaming ASR that runs all day,
`power_saver` when the battery or the thermals say so.

### `telemetry.py` — measuring without perturbing

Reading the Windows `BatteryStatus.DischargeRate` counter means spawning PowerShell:
**~450 ms**. Hexa-Router consults power state on every placement.

The naive implementation therefore put half a second of shell startup in front of every
single inference — and produced benchmarks where a stub doing no work measured at 445 ms.
The instrument was destroying the thing it measured.

So: a daemon thread samples the counter on its own slow cadence, `read_power_state()` never
blocks and returns the last known value, and the assembled state is cached for a second
because battery level does not change between two inferences. `EnergyProbe` is the
deliberate exception — it blocks for a fresh reading, because it is an instrument and its
callers time *around* it rather than through it.

`tests/test_telemetry.py` fails if anyone reintroduces a blocking read into the hot path.

### `session.py` — one place that talks to onnxruntime

Builds and caches an `InferenceSession` per (model, device), applies the router's chosen HTP
performance mode, and enables QNN context caching so a second launch skips graph
compilation. If a device rejects a graph it falls back one rung and retries once. Every run
is timed and fed back into the router's cost model.

## Layer 2 — Model adapters

Nine adapters, one contract:

```
real path  →  SessionCache returns a live session, we run it
stub path  →  no asset on disk, return a deterministic, labelled result
```

The stub path is not decoration. It means the full product runs on any machine, CI is
hermetic, and a judge can exercise everything before downloading a gigabyte. Every stubbed
result carries `degraded=True` all the way to the UI, so nothing ever silently passes a
fallback off as a real inference.

The LLM is special: autoregressive decode needs a runtime that owns the KV cache and a
pre-compiled HTP context binary, so it has four backends tried in order — **Genie** (QAIRT,
the NPU path), **onnxruntime-genai** (also NPU), **llama.cpp** (CPU, universal), and a
**template** backend that answers extractively from context with no weights at all.

## Layer 3 — ConsultAgent

Four jobs, in order: transcript → jargon → care plan → teach-back.

**Rules first, model second, everywhere.** The jargon lexicon is curated and cannot
hallucinate a term that was not spoken. Care-plan extraction is regex-based; the LLM runs
afterwards and **may only add**, never delete. A dropped red-flag instruction is the one
failure mode that could actually hurt a patient, so the model is not permitted to cause it.

Teach-back inverts the same asymmetry: term overlap decides coverage, and the LLM may only
*downgrade* an item to missed, never upgrade a missed item to covered. Optimism there would
defeat the point of the feature.

**Cross-lingual honesty.** The patient answers in their own language while the plan is
recorded in the doctor's. Comparing them directly scores everything as missed — worse than
not checking, because it would send a doctor away believing the patient understood nothing.
When the languages cannot be bridged, SETU reports *"could not verify"* instead.

## Layer 4 — Server and UI

FastAPI bound to loopback, serving a single-file UI with no CDN dependency — an offline app
that needs a CDN is not offline. The UI's bottom strip shows live Hexa-Router placement per
model and live battery draw, so the compute story is visible rather than claimed.

`/api/system` returns the full transparency payload: host, providers, every placement, the
learned cost model, which assets are loaded and which are missing.

## Demo replay

`pipeline/demo_script.py` replays a scripted consultation through the *real* pipeline — same
LID, jargon scan, extraction and teach-back. Only audio capture is bypassed, and the UI says
so on screen. A demo that depends on room audio in front of judges is a demo that fails.
