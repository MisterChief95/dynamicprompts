"""
Shared fixtures for benchmarks.

Run benchmarks with:
    pytest tests/benchmarks/ --benchmark-only
    pytest tests/benchmarks/ --benchmark-only --benchmark-sort=mean
    pytest tests/benchmarks/ --benchmark-only --benchmark-histogram

Compare runs (save then compare):
    pytest tests/benchmarks/ --benchmark-save=baseline
    pytest tests/benchmarks/ --benchmark-compare=baseline
"""
from pathlib import Path

import pytest
from dynamicprompts.commands.base import SamplingMethod
from dynamicprompts.sampling_context import SamplingContext
from dynamicprompts.wildcards import WildcardManager

WILDCARD_DATA_DIR = Path(__file__).parent.parent / "test_data" / "wildcards"


@pytest.fixture(scope="session")
def wildcard_manager() -> WildcardManager:
    return WildcardManager(path=WILDCARD_DATA_DIR)


@pytest.fixture(scope="session")
def random_context(wildcard_manager: WildcardManager) -> SamplingContext:
    return SamplingContext(
        wildcard_manager=wildcard_manager,
        default_sampling_method=SamplingMethod.RANDOM,
    )


@pytest.fixture(scope="session")
def combinatorial_context(wildcard_manager: WildcardManager) -> SamplingContext:
    return SamplingContext(
        wildcard_manager=wildcard_manager,
        default_sampling_method=SamplingMethod.COMBINATORIAL,
    )


@pytest.fixture(scope="session")
def cyclical_context(wildcard_manager: WildcardManager) -> SamplingContext:
    return SamplingContext(
        wildcard_manager=wildcard_manager,
        default_sampling_method=SamplingMethod.CYCLICAL,
    )


def pytest_benchmark_update_machine_info(config, machine_info):
    """Strip host-identifying fields from saved benchmark results.

    pytest-benchmark embeds the hostname and full CPU/OS fingerprint in every
    ``--benchmark-json`` / ``--benchmark-save`` payload. Keep the fields that
    make results comparable, drop the ones that identify the machine.
    """
    machine_info["node"] = "anonymous"
    cpu = machine_info.get("cpu")
    if cpu:
        for key in ("brand_raw", "hardware_raw", "vendor_id_raw", "serial"):
            cpu.pop(key, None)
