"""Shared adapter base.

Every model adapter follows the same contract:

    real path  -> SessionCache gives us a live ORT/Genie session, we run it
    stub path  -> no asset on disk, we return a deterministic, clearly-labelled result

The stub path is not decoration. It means the full pipeline, UI, tests and CI run on any
machine, and it means a judge can click through the entire product before a single
gigabyte of weights has been downloaded. Every stubbed result carries ``degraded=True``
so nothing in the UI ever silently pretends a stub is a real inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..runtime import Priority, SessionCache
from .registry import ModelCard, card


@dataclass
class Inference:
    """Uniform result envelope for every adapter."""

    value: Any
    model: str
    device: str
    latency_ms: float
    degraded: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "device": self.device,
            "latency_ms": round(self.latency_ms, 2),
            "degraded": self.degraded,
            **self.extra,
        }


class Adapter:
    """Base class holding the session cache and the model's catalogue card."""

    key: str = "base"
    priority: Priority = Priority.INTERACTIVE
    #: The asset that decides whether this adapter is really loaded. Models that ship as
    #: two graphs under one key (Whisper, the translator) must name one, or `doctor`
    #: reports them missing while they are demonstrably transcribing.
    primary_asset: str | None = None
    #: Some adapters have no model file at all - language ID is a Unicode-script scan.
    #: Reporting those as "missing" tells a user to download something that does not exist.
    needs_asset: bool = True

    def __init__(self, cache: SessionCache) -> None:
        self.cache = cache

    @property
    def card(self) -> ModelCard | None:
        return card(self.key)

    def available(self) -> bool:
        """True when a real asset is loaded (as opposed to the stub path)."""
        if not self.needs_asset:
            return True
        return (
            self.cache.get(self.key, filename=self.primary_asset, priority=self.priority)
            is not None
        )

    def status(self) -> dict[str, Any]:
        placement = self.cache.router.place(self.key, priority=self.priority)
        return {
            "key": self.key,
            "loaded": self.available(),
            "asset": self.primary_asset or f"{self.key}.onnx",
            "placement": placement.as_dict(),
            "card": self.card.as_dict() if self.card else None,
        }
