"""Configuration. Everything is overridable by env var so the demo machine and the
CI runner can differ without editing files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(key: str, default: str) -> str:
    return os.environ.get(f"SETU_{key}", default)


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(f"SETU_{key}")
    return Path(raw) if raw else default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(f"SETU_{key}")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


#: Languages the pipeline is wired for. Codes are ISO-639-1 / BCP-47 where they exist.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "हिन्दी (Hindi)",
    "te": "తెలుగు (Telugu)",
    "ta": "தமிழ் (Tamil)",
    "bn": "বাংলা (Bengali)",
    "mr": "मराठी (Marathi)",
    "kn": "ಕನ್ನಡ (Kannada)",
    "ml": "മലയാളം (Malayalam)",
    "gu": "ગુજરાતી (Gujarati)",
    "pa": "ਪੰਜਾਬੀ (Punjabi)",
    "or": "ଓଡ଼ିଆ (Odia)",
    "ur": "اردو (Urdu)",
}


@dataclass
class Settings:
    # storage
    model_root: Path = field(default_factory=lambda: _env_path("MODEL_ROOT", REPO_ROOT / "models"))
    data_dir: Path = field(default_factory=lambda: _env_path("DATA_DIR", REPO_ROOT / ".setu_data"))
    db_path: Path = field(default_factory=lambda: _env_path("DB", REPO_ROOT / ".setu_data" / "setu.db"))
    sessions_db: Path = field(
        default_factory=lambda: _env_path("SESSIONS_DB", REPO_ROOT / ".setu_data" / "consultations.db")
    )

    # server
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8756")))

    # behaviour
    default_language: str = field(default_factory=lambda: _env("LANG", "hi"))
    #: Hard guarantee for the demo: refuse to make any outbound network call.
    offline_only: bool = field(default_factory=lambda: _env_bool("OFFLINE_ONLY", True))
    #: Force a device for A/B benchmarking, e.g. SETU_FORCE_DEVICE=cpu
    force_device: str | None = field(default_factory=lambda: os.environ.get("SETU_FORCE_DEVICE"))

    #: How long a consultation is kept before `prune` drops it. Bounded by default,
    #: because a clinic that never thinks about retention should not silently accumulate
    #: years of patient conversations on a laptop. Set SETU_RETENTION_DAYS=0 to keep
    #: everything, which is a decision someone has to make deliberately.
    retention_days: int = field(default_factory=lambda: int(_env("RETENTION_DAYS", "90")))

    # retrieval
    chunk_chars: int = 900
    chunk_overlap: int = 150
    top_k: int = 6
    embed_dim: int = 384

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_root.mkdir(parents=True, exist_ok=True)


settings = Settings()
