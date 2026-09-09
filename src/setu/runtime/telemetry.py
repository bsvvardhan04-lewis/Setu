"""Power and thermal telemetry.

The interesting claim in this project is "the NPU makes an 8-hour battery workload
possible". Claims like that are worth nothing unless you measure them, so this module
reads *actual milliwatts off the battery* rather than estimating from utilisation.

On Windows the ``root\\WMI`` namespace exposes ``BatteryStatus.DischargeRate`` in mW.
Sampling it around a workload gives energy per inference directly. Everywhere else we
degrade to wall-clock only and say so honestly in the report.

**Reading that counter costs ~450 ms**, because it means spawning PowerShell. Hexa-Router
consults the power state on every placement decision, so a naive implementation puts half
a second of shell startup in front of every single inference - which both destroys the
latency it is trying to optimise and corrupts the benchmark measuring it. So:

* a daemon thread samples the counter in the background, on its own slow cadence
* ``read_power_state`` never blocks on it; it reads the last known value
* the assembled ``PowerState`` is itself cached briefly, since battery level does not
  meaningfully change between two inferences

``EnergyProbe`` is the deliberate exception: it blocks for a fresh reading, because it is
a measurement instrument and its callers time around it, not through it.
"""

from __future__ import annotations

import logging
import platform
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_WMI_QUERY = (
    "Get-CimInstance -Namespace root/WMI -ClassName BatteryStatus "
    "| Select-Object -First 1 -ExpandProperty DischargeRate"
)

#: How often the background thread refreshes the discharge counter.
_SAMPLE_INTERVAL = 5.0
#: How long an assembled PowerState stays good enough for a placement decision.
_STATE_TTL = 1.0


@dataclass(frozen=True)
class PowerState:
    """A snapshot of how much headroom the machine has right now."""

    on_ac: bool
    battery_percent: float | None
    discharge_mw: float | None
    cpu_percent: float
    thermal_pressure: float  # 0.0 cool .. 1.0 throttling

    @property
    def energy_constrained(self) -> bool:
        """Should the router trade latency for milliwatts?"""
        if self.on_ac:
            return False
        if self.battery_percent is None:
            return True  # unknown battery on DC: assume we must be careful
        return self.battery_percent < 40.0

    @property
    def thermally_constrained(self) -> bool:
        return self.thermal_pressure > 0.75


# --------------------------------------------------------------- discharge counter


def _read_discharge_mw_blocking() -> float | None:
    """Read the battery discharge counter. Costs ~450 ms; never call this in a hot path."""
    if platform.system() != "Windows":
        return None
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _WMI_QUERY],
            capture_output=True,
            text=True,
            timeout=6,
        )
    except Exception:
        return None
    text = out.stdout.strip()
    if not text:
        return None
    try:
        value = float(text.splitlines()[0].strip())
    except ValueError:
        return None
    # 0 means "not discharging" (i.e. on AC), which is not a measurement.
    return value if value > 0 else None


class _DischargeSampler:
    """Keeps a recent discharge reading available without ever blocking a caller."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: float | None = None
        self._at: float = 0.0
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while True:
            value = _read_discharge_mw_blocking()
            with self._lock:
                self._value = value
                self._at = time.monotonic()
            time.sleep(_SAMPLE_INTERVAL)

    def _ensure_running(self) -> None:
        if self._thread is not None:
            return
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run, name="setu-power-sampler", daemon=True
            )
            self._thread.start()

    def latest(self) -> float | None:
        """Last known milliwatts, or None until the first sample lands."""
        self._ensure_running()
        with self._lock:
            return self._value

    def force(self) -> float | None:
        """Blocking read, for measurement contexts only."""
        value = _read_discharge_mw_blocking()
        with self._lock:
            self._value = value
            self._at = time.monotonic()
        return value


_sampler = _DischargeSampler()


# ------------------------------------------------------------------ cheap sensors


def _read_battery() -> tuple[bool, float | None]:
    try:
        import psutil

        batt = psutil.sensors_battery()
    except Exception:
        return True, None
    if batt is None:
        return True, None
    return bool(batt.power_plugged), float(batt.percent)


def _read_cpu_percent() -> float:
    try:
        import psutil

        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return 0.0


def _read_thermal_pressure() -> float:
    """Best-effort 0..1 thermal signal.

    No portable Windows API exposes SoC junction temperature to user space, so we use
    sustained CPU load as a proxy and let callers treat it as a hint, not a truth.
    """
    try:
        import psutil

        temps = psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {}
        readings = [t.current for group in temps.values() for t in group if t.current]
        if readings:
            hottest = max(readings)
            return max(0.0, min(1.0, (hottest - 45.0) / 40.0))
    except Exception:
        pass
    return max(0.0, min(1.0, _read_cpu_percent() / 100.0))


# ------------------------------------------------------------------- public read

_state_lock = threading.Lock()
_state_cache: PowerState | None = None
_state_at: float = 0.0


def read_power_state(max_age: float = _STATE_TTL) -> PowerState:
    """Current power state. Cheap and non-blocking; safe to call per inference.

    Pass ``max_age=0`` to force a rebuild from the cheap sensors (this still does not
    block on the discharge counter - use ``EnergyProbe`` when you need a fresh one).
    """
    global _state_cache, _state_at

    now = time.monotonic()
    with _state_lock:
        if _state_cache is not None and (now - _state_at) < max_age:
            return _state_cache

    on_ac, pct = _read_battery()
    state = PowerState(
        on_ac=on_ac,
        battery_percent=pct,
        discharge_mw=None if on_ac else _sampler.latest(),
        cpu_percent=_read_cpu_percent(),
        thermal_pressure=_read_thermal_pressure(),
    )
    with _state_lock:
        _state_cache = state
        _state_at = now
    return state


# ------------------------------------------------------------------ energy probe


@dataclass
class EnergyReading:
    seconds: float
    samples_mw: list[float] = field(default_factory=list)

    @property
    def mean_mw(self) -> float | None:
        return statistics.fmean(self.samples_mw) if self.samples_mw else None

    @property
    def energy_mwh(self) -> float | None:
        mw = self.mean_mw
        return None if mw is None else mw * (self.seconds / 3600.0)

    def as_dict(self) -> dict[str, object]:
        return {
            "seconds": round(self.seconds, 4),
            "mean_mw": None if self.mean_mw is None else round(self.mean_mw, 1),
            "energy_mwh": None if self.energy_mwh is None else round(self.energy_mwh, 5),
            "n_power_samples": len(self.samples_mw),
        }


class EnergyProbe:
    """Times a block and samples battery draw across it.

    ``with EnergyProbe() as probe: run_workload()`` then read ``probe.reading``.

    Entry and exit each block for ~450 ms on a real counter read. That is deliberate and
    correct *provided callers time the workload inside the block rather than timing the
    block itself* - which is what the benchmark harness does. Do not use this inside a
    request path.
    """

    def __init__(self, mid_samples: int = 0) -> None:
        self.mid_samples = mid_samples
        self.reading = EnergyReading(seconds=0.0)
        self._t0 = 0.0

    def __enter__(self) -> "EnergyProbe":
        first = _sampler.force()
        if first is not None:
            self.reading.samples_mw.append(first)
        self._t0 = time.perf_counter()
        return self

    def sample(self) -> None:
        mw = _sampler.force()
        if mw is not None:
            self.reading.samples_mw.append(mw)

    def __exit__(self, *exc) -> bool:
        self.reading.seconds = time.perf_counter() - self._t0
        last = _sampler.force()
        if last is not None:
            self.reading.samples_mw.append(last)
        return False
