"""Hexa-Router placement policy."""

from __future__ import annotations

import pytest

from setu.runtime import Device, HexaRouter, ModelSpec, PowerState, Priority
from setu.runtime.router import DEFAULT_SPECS, build_default_router


def power(on_ac=True, pct=100.0, thermal=0.1) -> PowerState:
    return PowerState(
        on_ac=on_ac,
        battery_percent=pct,
        discharge_mw=None if on_ac else 5000.0,
        cpu_percent=10.0,
        thermal_pressure=thermal,
    )


@pytest.fixture
def router() -> HexaRouter:
    r = build_default_router()
    # Pretend the machine has every engine, so policy is what is under test.
    r._devices = {Device.NPU, Device.GPU, Device.CPU, Device.STUB}
    return r


def test_registers_every_shipped_model():
    r = build_default_router()
    assert {s.name for s in DEFAULT_SPECS} <= set(r._specs)


def test_prefers_npu_for_interactive_work(router):
    placement = router.place("asr", priority=Priority.INTERACTIVE, power=power())
    assert placement.device is Device.NPU
    assert placement.perf_mode == "burst"


def test_streaming_uses_sustained_mode(router):
    placement = router.place("asr", priority=Priority.STREAMING, power=power())
    assert placement.perf_mode == "sustained_high_performance"


def test_low_battery_switches_npu_to_power_saver(router):
    placement = router.place(
        "asr", priority=Priority.BACKGROUND, power=power(on_ac=False, pct=15.0)
    )
    assert placement.perf_mode == "power_saver"


def test_thermal_pressure_steers_off_the_npu(router):
    hot = power(thermal=0.95)
    placement = router.place("ocr_detect", priority=Priority.INTERACTIVE, power=hot)
    assert placement.device is not Device.NPU


def test_cpu_only_model_never_lands_on_an_accelerator(router):
    placement = router.place("lid", power=power())
    assert placement.device is Device.CPU


def test_falls_back_to_stub_when_no_device_qualifies():
    r = HexaRouter()
    r.register(ModelSpec(name="ghost", role="test", allowed=(Device.GPU,)))
    r._devices = {Device.CPU, Device.STUB}
    assert r.place("ghost", power=power()).device is Device.STUB


def test_hysteresis_holds_a_placement_against_a_marginal_win(router):
    first = router.place("embed", priority=Priority.INTERACTIVE, power=power())
    # Report the NPU as only slightly better than CPU: not worth a session rebuild.
    router.observe("embed", Device.CPU, latency_ms=100.0)
    router.observe("embed", Device.NPU, latency_ms=95.0)
    second = router.place("embed", priority=Priority.INTERACTIVE, power=power())
    assert second.device == first.device


def test_learned_cost_overrides_an_optimistic_prior(router):
    # Prior says the NPU is 3x faster for embeddings; measurement says otherwise.
    for _ in range(5):
        router.observe("embed", Device.NPU, latency_ms=900.0)
        router.observe("embed", Device.CPU, latency_ms=30.0)
    placement = router.place("embed", priority=Priority.INTERACTIVE, power=power())
    assert placement.device is Device.CPU


def test_snapshot_is_json_shaped(router):
    router.place("llm", power=power())
    router.observe("llm", Device.NPU, latency_ms=120.0, energy_mwh=0.02)
    snap = router.snapshot()
    assert "placements" in snap and "cost_model" in snap
    assert snap["cost_model"]["llm@npu"]["samples"] == 1
