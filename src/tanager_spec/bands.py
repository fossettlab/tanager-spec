"""Wavelength-axis utilities shared across spectral operations.

The whole workspace assumes band order is strictly increasing in wavelength.
These helpers enforce that assumption and translate wavelength windows into
band indices.
"""

from __future__ import annotations

import numpy as np


def assert_strictly_increasing(wavelengths: np.ndarray) -> None:
    """Validate that a wavelength vector is strictly increasing.

    Parameters
    ----------
    wavelengths : np.ndarray
        1-D array of band-center wavelengths (nm).

    Raises
    ------
    ValueError
        If the array is not 1-D, or not strictly increasing.
    """
    wl = np.asarray(wavelengths)
    if wl.ndim != 1:
        raise ValueError(f"wavelengths must be 1-D, got shape {wl.shape}")
    diffs = np.diff(wl)
    if not np.all(diffs > 0):
        bad = int(np.argmin(diffs))
        raise ValueError(
            "wavelengths must be strictly increasing; "
            f"first non-increasing step at index {bad} "
            f"({wl[bad]:.3f} -> {wl[bad + 1]:.3f} nm)"
        )


def indices_in_windows(
    wavelengths: np.ndarray,
    windows: list[tuple[float, float]],
) -> np.ndarray:
    """Return a boolean mask of bands falling inside any wavelength window.

    Parameters
    ----------
    wavelengths : np.ndarray
        1-D array of band-center wavelengths (nm).
    windows : list of (float, float)
        ``(low_nm, high_nm)`` inclusive intervals.

    Returns
    -------
    np.ndarray
        Boolean array, shape ``(n_bands,)``. ``True`` where the band center
        falls within at least one window.
    """
    wl = np.asarray(wavelengths)
    in_window = np.zeros(wl.shape, dtype=bool)
    for low, high in windows:
        in_window |= (wl >= low) & (wl <= high)
    return in_window
