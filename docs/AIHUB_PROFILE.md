# On-device profiling (Qualcomm AI Hub)

> **Not yet generated.** This file is written by `scripts/aihub_profile.py`. It is checked
> in as a placeholder so the links from the README and the submission resolve, and so it is
> obvious that the numbers are absent rather than invented.

## How to produce it

```bash
pip install qai-hub
qai-hub configure --api_token <token from https://aihub.qualcomm.com>

python scripts/aihub_profile.py --list-devices   # see what hardware is available
python scripts/aihub_profile.py --all            # compile + profile every graph
```

This requires the model assets to be present — run `python scripts/fetch_models.py --all`
first. Models that are missing are reported as skipped rather than silently omitted.

## What it will contain

For each ONNX graph, compiled as a **QNN context binary** and executed on **physical
Snapdragon silicon** provisioned by AI Hub:

| Column | Meaning |
| --- | --- |
| Inference | Measured on-device latency |
| Peak memory | Measured peak inference memory |
| **NPU layers** | How many layers actually ran on the Hexagon NPU, out of the total |
| Verify | Link to the job on Qualcomm's own site |

**The NPU layers column is the one that matters.** A model can compile successfully and
still scatter half its layers back onto the CPU — usually because of a dynamic axis or an
unsupported op. That model is not meaningfully "running on the NPU", and the entire premise
of this project depends on telling the difference. This is why the figure is reported per
model rather than as a single headline number.

Nothing here is estimated, and nothing is copied from a datasheet. Every row carries a job
URL so a reader can check it independently rather than taking the claim on trust.
