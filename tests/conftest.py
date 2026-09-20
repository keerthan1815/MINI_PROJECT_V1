"""
conftest.py – Shared pytest fixtures for network_simulator tests.

Provides pre-built NetworkSimulator instances for each device type so
individual test functions don't repeat boilerplate construction.
"""

import sys
import os

import pytest

# Ensure the project root is on sys.path so imports work when pytest is run
# from any directory (e.g. `pytest` from repo root or `python -m pytest`).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from network_simulator import NetworkSimulator


# ---------------------------------------------------------------------------
# Per-device simulators (function-scoped: each test gets a fresh instance)
# ---------------------------------------------------------------------------

@pytest.fixture()
def router_sim():
    """A fresh Router simulator."""
    return NetworkSimulator("Router")


@pytest.fixture()
def switch_sim():
    """A fresh Switch simulator."""
    return NetworkSimulator("Switch")


@pytest.fixture()
def firewall_sim():
    """A fresh Firewall simulator."""
    return NetworkSimulator("Firewall")


# ---------------------------------------------------------------------------
# Convenience: all three at once
# ---------------------------------------------------------------------------

@pytest.fixture()
def all_sims(router_sim, switch_sim, firewall_sim):
    """Dict of {device_name: simulator} for multi-device tests."""
    return {
        "Router": router_sim,
        "Switch": switch_sim,
        "Firewall": firewall_sim,
    }


# ---------------------------------------------------------------------------
# Helper: bulk readings generator (available as a fixture factory)
# ---------------------------------------------------------------------------

@pytest.fixture()
def bulk_readings():
    """
    Factory fixture: returns a callable ``get(sim, n)`` that collects
    *n* readings from *sim* and returns them as a list of dicts.
    """
    def _get(sim: NetworkSimulator, n: int = 100):
        return [sim.next_reading() for _ in range(n)]
    return _get
