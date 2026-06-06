from __future__ import annotations

import numpy as np
import pytest

from tanagerspec import bands


def test_strictly_increasing_accepts_increasing():
    bands.assert_strictly_increasing(np.array([400.0, 500.0, 600.0]))


def test_strictly_increasing_rejects_flat_step():
    with pytest.raises(ValueError, match="strictly increasing"):
        bands.assert_strictly_increasing(np.array([400.0, 400.0, 600.0]))


def test_strictly_increasing_rejects_decreasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        bands.assert_strictly_increasing(np.array([400.0, 600.0, 500.0]))


def test_strictly_increasing_rejects_2d():
    with pytest.raises(ValueError, match="1-D"):
        bands.assert_strictly_increasing(np.zeros((3, 3)))


def test_indices_in_windows_selects_correct_bands():
    wl = np.array([400.0, 760.0, 1000.0, 1400.0, 2000.0])
    mask = bands.indices_in_windows(wl, [(755.0, 770.0), (1350.0, 1450.0)])
    assert mask.tolist() == [False, True, False, True, False]


def test_indices_in_windows_empty_windows():
    wl = np.array([400.0, 760.0])
    mask = bands.indices_in_windows(wl, [])
    assert not mask.any()
