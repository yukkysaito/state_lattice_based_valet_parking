"""Shared fixtures for the auto-parking unit tests."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from vehicle import VehicleInfo, default_vehicle  # noqa: E402


@pytest.fixture
def vehicle() -> VehicleInfo:
    return default_vehicle()


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
