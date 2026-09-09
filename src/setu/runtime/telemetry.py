"""Power and thermal telemetry.

The interesting claim in this project is "the NPU makes an 8-hour battery workload
possible". Claims like that are worth nothing unless you measure them, so this module
reads *actual milliwatts off the battery* rather than estimating from utilisation.

On Windows the ``root\WMI`` namespace exposes ``BatteryStatus.DischargeRate`` in mW.
Sampling it around a workload gives energy per inference directly. Everywhere else we
degrade to wall-clock only and say so honestly in the report.
"""

from __future__ import annotations

import logging
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_WMI_QUERY = (
    "Get-CimInstance -Namespace root/WMI -ClassName BatteryStatus "
    "| Select-Object -First 1 -ExpandProperty DischargeRate"
)


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


def _read_discharge_mw() -> float | None:
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


def read_power_state() -> PowerState:
    on_ac, pct = _read_battery()
    return PowerState(
        on_ac=on_ac,
        battery_percent=pct,
        discharge_mw=None if on_ac else _read_discharge_mw(),
        cpu_percent=_read_cpu_percent(),
        thermal_pressure=_read_thermal_pressure(),
    )


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
    """Context manager that times a block and samples battery draw across it.

    ``with EnergyProbe() as probe: run_workload()`` then read ``probe.reading``.
    Sampling is deliberately cheap (one WMI call at entry, one at exit, plus optional
    mid-samples) because the probe must not perturb what it measures.
    """

    def __init__(self, mid_samples: int = 0) -> None:
        self.mid_samples = mid_samples
        self.reading = EnergyReading(seconds=0.0)
        self._t0 = 0.0

    def __enter__(self) -> "EnergyProbe":
        first = _read_discharge_mw()
        if first is not None:
            self.reading.samples_mw.append(first)
        self._t0 = time.perf_counter()
        return self

    def sample(self) -> None:
        mw = _read_discharge_mw()
        if mw is not None:
            self.reading.samples_mw.append(mw)

    def __exit__(self, *exc) -> bool:
        self.reading.seconds = time.perf_counter() - self._t0
        last = _read_discharge_mw()
        if last is not None:
            self.reading.samples_mw.append(last)
        return False
