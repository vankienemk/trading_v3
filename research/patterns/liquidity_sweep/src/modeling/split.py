"""Time-based train/validation/test split with purging and embargo.

Implements guide sections 22.1-22.5: chronological contiguous splits
(never random), purge of events whose label horizon reaches into
validation, and an embargo gap between train and validation/test.

Contract (INTERFACES.md §8):
- ``time_split(positions, ...)`` returns three positional index arrays
  ``(train_idx, val_idx, test_idx)``.
- ``apply_purge_and_embargo(...)`` takes event positions and horizon,
  returns a boolean mask over the **training** set only.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class SplitError(ValueError):
    """Raised when split parameters are invalid."""


def time_split(
    positions: np.ndarray | int,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return chronological contiguous positional indices for train/val/test.

    Parameters
    ----------
    positions : np.ndarray or int
        Either an array of positional indices (e.g. ``np.arange(n)``) or an
        integer count ``n``.  If an array, its length determines the split;
        the returned arrays are subsets of ``positions``.
    train_fraction : float
        Fraction for training (default 0.60).
    validation_fraction : float
        Fraction for validation (default 0.20).
    test_fraction : float
        Fraction for test (default 0.20).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(train_idx, val_idx, test_idx)`` — 1-D arrays of integer positions,
        each strictly ascending with step 1 (contiguous blocks).

    Raises
    ------
    SplitError
        If fractions don't sum to 1.0 or leave an empty split.
    """
    # Accept either an integer count or a positions array
    if isinstance(positions, np.ndarray):
        n_rows = len(positions)
        pos_array = positions
    else:
        n_rows = int(positions)
        pos_array = np.arange(n_rows)

    if n_rows <= 0:
        raise SplitError("n_rows must be > 0")
    if not math.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise SplitError("split fractions must sum to 1.0")

    train_end = round(n_rows * train_fraction)
    val_end = train_end + round(n_rows * validation_fraction)

    if not (0 < train_end < val_end < n_rows):
        raise SplitError(
            f"fractions leave an empty split for n_rows={n_rows} "
            f"(train_end={train_end}, val_end={val_end})"
        )

    train_idx = pos_array[:train_end].copy()
    val_idx = pos_array[train_end:val_end].copy()
    test_idx = pos_array[val_end:].copy()

    return train_idx, val_idx, test_idx


def apply_purge(
    train_positions: np.ndarray,
    horizon_bars: int,
    validation_start_pos: int,
) -> np.ndarray:
    """Return boolean mask: True where event's label window stays inside train.

    An event at position ``p`` uses future bars ``p+1 .. p+horizon_bars`` to
    form its label.  It leaks into validation if ``p + horizon_bars >=
    validation_start_pos``; such events are purged (mask=False).

    Parameters
    ----------
    train_positions : np.ndarray
        Integer positions of training events (must be < validation_start_pos).
    horizon_bars : int
        Label horizon in bars (e.g. max_horizon from labeling config).
    validation_start_pos : int
        First position of the validation set.

    Returns
    -------
    np.ndarray
        Boolean mask same length as ``train_positions``; True = keep.
    """
    if len(train_positions) == 0:
        return np.array([], dtype=bool)
    if horizon_bars < 0:
        raise SplitError("horizon_bars must be >= 0")
    return (train_positions + horizon_bars) < validation_start_pos


def apply_embargo(
    train_positions: np.ndarray,
    validation_start_pos: int,
    embargo_bars: int,
) -> np.ndarray:
    """Return boolean mask: True where event is outside the embargo gap.

    Events at ``position >= validation_start_pos - embargo_bars`` are dropped
    so no training sample's context abuts the validation window.

    Parameters
    ----------
    train_positions : np.ndarray
        Integer positions of training events.
    validation_start_pos : int
        First position of the validation set.
    embargo_bars : int
        Gap size in bars (baseline 32 per guide §22.5).

    Returns
    -------
    np.ndarray
        Boolean mask same length as ``train_positions``; True = keep.
    """
    if len(train_positions) == 0:
        return np.array([], dtype=bool)
    if embargo_bars < 0:
        raise SplitError("embargo_bars must be >= 0")
    embargo_boundary = validation_start_pos - embargo_bars
    return train_positions < embargo_boundary


def apply_purge_and_embargo(
    train_positions: np.ndarray,
    horizon_bars: int,
    validation_start_pos: int,
    embargo_bars: int,
) -> np.ndarray:
    """Combined purge + embargo mask for training events.

    Parameters
    ----------
    train_positions : np.ndarray
        Integer positions of training events.
    horizon_bars : int
        Maximum label horizon (purge threshold).
    validation_start_pos : int
        First position of the validation set.
    embargo_bars : int
        Embargo gap size.

    Returns
    -------
    np.ndarray
        Boolean mask; True = event survives both purge and embargo.
    """
    purge_mask = apply_purge(train_positions, horizon_bars, validation_start_pos)
    embargo_mask = apply_embargo(train_positions, validation_start_pos, embargo_bars)
    return purge_mask & embargo_mask


def build_train_val_test_splits(
    n_events: int,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """High-level split builder using config values.

    Parameters
    ----------
    n_events : int
        Total number of labeled events.
    config : dict
        Merged config with keys under ``splitting``:
        ``train_fraction``, ``validation_fraction``, ``test_fraction``,
        ``embargo_bars``.

    Returns
    -------
    dict
        ``{"train": idx_array, "validation": idx_array, "test": idx_array}``
        with raw chronological splits (no purge/embargo applied yet — use
        :func:`apply_purge_and_embargo` on the train indices separately).
    """
    splitting = config.get("splitting", {})
    train_frac = float(splitting.get("train_fraction", 0.60))
    val_frac = float(splitting.get("validation_fraction", 0.20))
    test_frac = float(splitting.get("test_fraction", 0.20))

    train_idx, val_idx, test_idx = time_split(
        n_events, train_frac, val_frac, test_frac
    )
    return {"train": train_idx, "validation": val_idx, "test": test_idx}


def walk_forward_folds(
    n_events: int,
    initial_train_size: int,
    step_size: int,
    validation_size: int,
    embargo_bars: int = 32,
    horizon_bars: int = 32,
) -> list[dict[str, np.ndarray]]:
    """Generate expanding-window walk-forward folds per guide §22.3.

    Fold k:
    - Train: events [0 .. initial_train_size + (k-1)*step_size)
    - Validation: next ``validation_size`` events after embargo gap
    - Purge + embargo applied to training set before each fold

    Example (initial_train=300, step=100, val=100):
    - Fold 0: train=[0..300), val=[332..432)  (embargo 32 bars between)
    - Fold 1: train=[0..400), val=[432..532)
    - Fold 2: train=[0..500), val=[532..632)

    Parameters
    ----------
    n_events : int
        Total number of events in the dataset.
    initial_train_size : int
        Size of the initial training window.
    step_size : int
        Number of events to expand training by per fold.
    validation_size : int
        Size of each validation window.
    embargo_bars : int
        Gap between train end and validation start (default 32).
    horizon_bars : int
        Label horizon for purging (default 32).

    Returns
    -------
    list[dict]
        Each dict has keys:
        - "fold": int (fold index)
        - "train": np.ndarray of training positions (after purge+embargo)
        - "validation": np.ndarray of validation positions
        - "train_raw_end": int (raw train boundary before purge/embargo)
        - "val_start": int (first validation position)

    Raises
    ------
    SplitError
        If parameters would produce empty folds or exceed dataset bounds.
    """
    if n_events <= 0:
        raise SplitError("n_events must be > 0")
    if initial_train_size <= 0:
        raise SplitError("initial_train_size must be > 0")
    if step_size <= 0:
        raise SplitError("step_size must be > 0")
    if validation_size <= 0:
        raise SplitError("validation_size must be > 0")
    if embargo_bars < 0:
        raise SplitError("embargo_bars must be >= 0")
    if horizon_bars < 0:
        raise SplitError("horizon_bars must be >= 0")

    folds = []
    fold_idx = 0

    while True:
        # Compute raw train boundary (expanding window)
        train_end = initial_train_size + fold_idx * step_size
        val_start = train_end + embargo_bars
        val_end = val_start + validation_size

        # Check bounds
        if val_end > n_events:
            break  # No more room for a full validation window

        if train_end >= val_start:
            raise SplitError(
                f"Fold {fold_idx}: train_end ({train_end}) >= val_start ({val_start}); "
                "increase embargo_bars or reduce train expansion"
            )

        # Raw train indices
        raw_train = np.arange(0, train_end)

        # Apply purge + embargo to get clean training set
        if len(raw_train) > 0:
            mask = apply_purge_and_embargo(
                raw_train, horizon_bars, val_start, embargo_bars
            )
            clean_train = raw_train[mask]
        else:
            clean_train = np.array([], dtype=int)

        # Validation indices
        val_idx = np.arange(val_start, val_end)

        folds.append({
            "fold": fold_idx,
            "train": clean_train,
            "validation": val_idx,
            "train_raw_end": train_end,
            "val_start": val_start,
        })

        fold_idx += 1

    if len(folds) == 0:
        raise SplitError(
            f"No valid walk-forward folds generated for n_events={n_events}, "
            f"initial_train={initial_train_size}, step={step_size}, "
            f"val={validation_size}, embargo={embargo_bars}"
        )

    return folds

