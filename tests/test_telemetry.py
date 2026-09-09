"""Power telemetry must be cheap enough to sit in the placement hot path.

This exists because it once was not: reading the Windows battery counter spawns
PowerShell (~450 ms), and Hexa-Router consults the power state on every placement, so a
blocking read added half a second to every inference and corrupted the benchmarks that
were supposed to prove the project's central claim.
"""

from __future__ import annotations

import time

from setu.runtime import telemetry
from setu.runtime.telemetry import EnergyProbe, PowerState, read_power_state


def test_read_power_state_is_fast_enough_for_the_hot_path():
    read_power_state()  # prime the cache and start the sampler
    start = time.perf_counter()
    for _ in range(50):
        read_power_state()
    per_call_ms = (time.perf_counter() - start) * 1000.0 / 50
    assert per_call_ms < 5.0, f"{per_call_ms:.1f} ms per call is too slow for placement"


def test_state_is_cached_within_its_ttl(monkeypatch):
    calls = {"n": 0}

    def counted():
        calls["n"] += 1
        return True, 55.0

    monkeypatch.setattr(telemetry, "_read_battery", counted)
    read_power_state(max_age=0)  # force one rebuild
    before = calls["n"]
    for _ in range(10):
        read_power_state()
    assert calls["n"] == before, "cached reads must not re-poll the sensors"


def test_never_blocks_on_the_discharge_counter(monkeypatch):
    def explode():
        raise AssertionError("read_power_state must not make a blocking counter read")

    monkeypatch.setattr(telemetry, "_read_discharge_mw_blocking", explode)
    read_power_state(max_age=0)


def test_power_state_policy_flags():
    on_ac = PowerState(True, 20.0, None, 10.0, 0.1)
    assert on_ac.energy_constrained is False

    low = PowerState(False, 15.0, 6000.0, 10.0, 0.1)
    assert low.energy_constrained is True

    unknown = PowerState(False, None, None, 10.0, 0.1)
    assert unknown.energy_constrained is True

    hot = PowerState(True, 90.0, None, 99.0, 0.9)
    assert hot.thermally_constrained is True


def test_energy_probe_reports_a_duration():
    with EnergyProbe() as probe:
        time.sleep(0.01)
    assert probe.reading.seconds >= 0.01
    payload = probe.reading.as_dict()
    assert "energy_mwh" in payload and "n_power_samples" in payload
