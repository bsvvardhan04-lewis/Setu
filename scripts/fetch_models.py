"""Fetch and export SETU's model assets.

Two sources, two mechanisms:

* **Qualcomm AI Hub** (Whisper, Llama) - exported through `qai_hub_models`, which
  compiles the model for a chosen Snapdragon target and downloads the artefact.
* **Open source** (Silero VAD, MiniLM, PaddleOCR, IndicTrans2, Piper) - downloaded
  from their upstream repos and quantised locally.

Everything lands under `models/<key>/` with the filenames `src/setu/models/registry.py`
declares, so the app finds them with no further configuration.

This script deliberately does NOT run at install time. Model weights are large, their
licences differ, and a build step that silently downloads gigabytes is a bad neighbour.
Run it explicitly:

    pip install "qai-hub-models[whisper]" huggingface-hub onnx onnxruntime
    python scripts/fetch_models.py --list
    python scripts/fetch_models.py --model embed --model vad
    python scripts/fetch_models.py --all
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"

# --------------------------------------------------------------------------- recipes

#: Models exported through qai_hub_models. `module` is the package path; the export
#: writes a compiled artefact we then move into place. Names on AI Hub are versioned,
#: so `--list` prints them and any mismatch is reported rather than silently ignored.
AIHUB_RECIPES: dict[str, dict] = {
    "asr": {
        "module": "qai_hub_models.models.whisper_small_v2",
        "extra": "whisper_small_v2",
        "produces": ["asr_encoder.onnx", "asr_decoder.onnx"],
        "licence": "MIT (OpenAI Whisper)",
    },
    "llm": {
        "module": "qai_hub_models.models.llama_v3_2_3b_chat_quantized",
        "extra": "llama_v3_2_3b_chat_quantized",
        "produces": ["genie_config.json", "llama_v3_2_3b.bin"],
        "licence": "Llama 3.2 Community License",
        "note": "Large. Deploys through Genie/QAIRT, not ONNX Runtime.",
    },
}

#: Models pulled straight from their upstream open-source home.
HF_RECIPES: dict[str, dict] = {
    "vad": {
        "repo": "onnx-community/silero-vad",
        "files": {"onnx/model.onnx": "vad.onnx"},
        "licence": "MIT",
    },
    "embed": {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "files": {
            "onnx/model.onnx": "embed.onnx",
            "tokenizer.json": "tokenizer.json",
        },
        "licence": "Apache-2.0",
    },
    "translate": {
        "repo": "ai4bharat/indictrans2-en-indic-dist-200M",
        "files": {},
        "licence": "MIT",
        "note": "Needs an ONNX export step; see docs/MODELS.md. Placeholder entry.",
    },
    "tts": {
        "repo": "rhasspy/piper-voices",
        "files": {
            "hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx": "tts.onnx",
            "hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx.json": "tts.onnx.json",
        },
        "licence": "MIT",
    },
}


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _skip(msg: str) -> None:
    print(f"  [skip] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def list_recipes() -> None:
    print("Qualcomm AI Hub exports")
    for key, recipe in AIHUB_RECIPES.items():
        print(f"  {key:<12} {recipe['module']}")
        print(f"  {'':<12} licence: {recipe['licence']}")
    print("\nOpen-source downloads")
    for key, recipe in HF_RECIPES.items():
        print(f"  {key:<12} {recipe['repo']}")
        print(f"  {'':<12} licence: {recipe['licence']}")
    print("\nAlready present under models/:")
    for path in sorted(MODELS.glob("*/*")) if MODELS.exists() else []:
        if path.is_file():
            size = path.stat().st_size / (1024 * 1024)
            print(f"  {path.relative_to(MODELS)}  ({size:.1f} MB)")


def fetch_hf(key: str, recipe: dict, force: bool) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        _fail("huggingface-hub not installed: pip install huggingface-hub")
        return False

    if not recipe["files"]:
        _skip(f"{key}: {recipe.get('note', 'no direct download defined')}")
        return False

    dest_dir = MODELS / key
    dest_dir.mkdir(parents=True, exist_ok=True)
    got_any = False
    for remote, local in recipe["files"].items():
        dest = dest_dir / local
        if dest.is_file() and not force:
            _skip(f"{key}/{local} already present")
            got_any = True
            continue
        try:
            src = hf_hub_download(repo_id=recipe["repo"], filename=remote)
        except Exception as exc:
            _fail(f"{key}/{local}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        shutil.copyfile(src, dest)
        _ok(f"{key}/{local}  <- {recipe['repo']}/{remote}")
        got_any = True
    return got_any


def fetch_aihub(key: str, recipe: dict, device: str, force: bool) -> bool:
    dest_dir = MODELS / key
    if all((dest_dir / f).is_file() for f in recipe["produces"]) and not force:
        _skip(f"{key} already present")
        return True

    try:
        import qai_hub_models  # noqa: F401
    except ImportError:
        _fail(
            f"{key}: qai-hub-models not installed.\n"
            f'         pip install "qai-hub-models[{recipe["extra"]}]"'
        )
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        f"{recipe['module']}.export",
        "--device",
        device,
        "--output-dir",
        str(dest_dir),
    ]
    print(f"  running: {' '.join(cmd[2:])}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except Exception as exc:
        _fail(f"{key}: {exc}")
        return False

    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-4:]
        _fail(f"{key}: export failed\n         " + "\n         ".join(tail))
        return False

    _ok(f"{key} exported to models/{key}/")
    missing = [f for f in recipe["produces"] if not (dest_dir / f).is_file()]
    if missing:
        print(
            f"         note: expected {missing} - AI Hub asset names are versioned, "
            "so rename what landed to match src/setu/models/registry.py"
        )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="show recipes and what is present")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", action="append", help="fetch only these (repeatable)")
    parser.add_argument("--device", default="Snapdragon X Elite CRD")
    parser.add_argument("--force", action="store_true", help="re-download existing assets")
    args = parser.parse_args(argv)

    if args.list:
        list_recipes()
        return 0
    if not args.all and not args.model:
        parser.error("pass --all, or --model NAME (repeatable), or --list")

    wanted = set(args.model or []) if not args.all else set(AIHUB_RECIPES) | set(HF_RECIPES)
    unknown = wanted - set(AIHUB_RECIPES) - set(HF_RECIPES)
    if unknown:
        parser.error(f"unknown model(s): {', '.join(sorted(unknown))}")

    MODELS.mkdir(parents=True, exist_ok=True)
    print("Open-source downloads")
    for key in sorted(wanted & set(HF_RECIPES)):
        fetch_hf(key, HF_RECIPES[key], args.force)

    aihub_wanted = sorted(wanted & set(AIHUB_RECIPES))
    if aihub_wanted:
        print(f"\nQualcomm AI Hub exports (target: {args.device})")
        for key in aihub_wanted:
            fetch_aihub(key, AIHUB_RECIPES[key], args.device, args.force)

    print("\nCheck what the app can now see:  python -m setu.cli doctor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
