# Demo video — 3 minute shot list

The video is the single most-weighted artefact in the submission. Judges will watch it
before they read a line of code, and many will decide from it alone.

**Rules for the shoot**
- Turn Wi-Fi **off on camera**, deliberately, before anything else. That one gesture is the
  entire pitch.
- Use **Replay sample visit**, not a live microphone. Room audio in front of judges fails.
  The UI states on screen that the audio is replayed — say so out loud too. Being caught
  hiding it costs far more than admitting it.
- Record at 1920×1080. Keep the browser at 100% zoom so the bottom compute strip is legible.
- No music. Speak plainly.

---

## 0:00–0:25 — The problem, as a scene

**Visual:** the sample transcript, paused on the doctor's line about chest pain.

> "A patient at a district hospital is told, in English, that they have hypertension. Take
> amlodipine 5 mg *od*. Get a fasting lipid profile. Come back immediately if you get chest
> pain. They nod. They say *haan ji*. They go home.
>
> They didn't understand 'od'. They didn't understand 'fasting'. And they don't know that
> chest pain means come back *now*.
>
> Nobody in that room finds out."

## 0:25–0:40 — Turn off the network

**Visual:** click the Windows network icon, enable airplane mode. Hold for two seconds.
Then the SETU header, showing `on-device · no network`.

> "Everything you're about to see runs on this laptop. A consultation is protected health
> information — sending it to a cloud API isn't a design choice we get to make."

## 0:40–1:30 — The consultation

**Visual:** press **Replay sample visit**. Let it run. Do not talk over the whole thing.

Point at things as they appear:

> "Each turn is transcribed and shown in the patient's language."
>
> *(as jargon highlights appear)* "It's flagging the words the patient almost certainly
> didn't parse. 'od' means once a day. 'Fasting' means nothing to eat for eight hours.
> Notice it glosses standalone 'od' — and leaves 'amlodipine' alone."
>
> *(as the care plan builds)* "And it's pulling out the actual instructions as they're
> given. Red flags first, because those are what kill people."

## 1:30–2:05 — The moment

**Visual:** press **Run teach-back**. Enter the patient's partial restatement. Let the red
panel land, and **hold the shot for three full seconds** before speaking.

> "Now we ask the patient to say the plan back in their own words.
>
> Two of five. They remembered the tablet. They did not remember that chest pain means come
> back immediately.
>
> The doctor learns this while the patient is still in the room."

Then the take-home card:

> "And they leave with this. In their language. Ordered by what will hurt them if they
> forget it."

## 2:05–2:40 — Why Snapdragon

**Visual:** the bottom compute strip. Zoom in on the chips and the live wattage.

> "Every model, and where it's running right now. This is Hexa-Router — it places each
> model on the NPU, GPU or CPU per request, reading live battery and thermal state, and it
> learns each model's real cost on this specific machine.
>
> That matters because this is a *sustained* workload. A six-hour clinic day of continuous
> speech recognition. On a CPU the laptop is hot and dead by lunch. On the Hexagon NPU it's
> a background process."

**Visual:** cut to `docs/AIHUB_PROFILE.md`, scroll the table, hover a job link.

> "And these aren't my numbers. Every model was compiled and profiled on real Snapdragon
> silicon through Qualcomm AI Hub. Each row links to the job — you can verify it yourself."

## 2:40–3:00 — Close

**Visual:** terminal, `python -m setu.cli doctor`, then the passing test suite.

> "It runs on any machine — every model has a labelled fallback, and nothing is ever passed
> off as a real inference when it isn't. On a Snapdragon PC the NPU lights up automatically.
>
> It's not a translator. It's a check that the patient understood. Offline, because it has
> to be."

---

## Checklist before recording

- [ ] `python -m setu.cli doctor` runs clean
- [ ] `pytest -q` passes, terminal ready for the closing shot
- [ ] `docs/AIHUB_PROFILE.md` generated with real job links
- [ ] `docs/BENCHMARKS.md` regenerated **on battery** so the energy column is populated
- [ ] Browser at 100% zoom, bookmarks bar hidden, no other tabs
- [ ] Notifications silenced (Focus assist on)
- [ ] Wi-Fi actually off, not just visually
