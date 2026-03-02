"""
Pytest configuration and fixtures for The Switchboard tests.
"""

import os

import pytest

# Ensure tests know they're running in test mode
os.environ["PYTEST_CURRENT_TEST"] = "1"


@pytest.fixture(autouse=True)
def reset_prometheus_registry():
    """Reset Prometheus registry between tests to avoid conflicts."""
    from prometheus_client import REGISTRY

    # Store original collectors
    yield

    # After test: unregister custom collectors to allow re-registration
    # This is a workaround for Prometheus not supporting multiple app instances
    collectors_to_remove = []
    for collector in list(REGISTRY._names_to_collectors.values()):
        name = getattr(collector, '_name', '')
        if name.startswith('switchboard_') or name.startswith('http_'):
            collectors_to_remove.append(collector)

    for collector in collectors_to_remove:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
