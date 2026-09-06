from __future__ import annotations

import math

CE_RANGE_LABELS = (
    "non-finite",
    "min-to-q1",
    "q1-to-median",
    "median-to-q3",
    "q3-to-max",
)


def tensorboard_label(value: str) -> str:
    """Sanitize a group label so it is safe to use as a TensorBoard tag segment."""
    return value.strip().replace("/", "∕").replace(" ", "_")


def ce_range_label(value: float, *, q1: float, median: float, q3: float) -> str:
    """Bin a collision-energy value into one of ``CE_RANGE_LABELS``.

    ``q1``/``median``/``q3`` are quartile cuts computed over whatever
    collection of collision-energy values the caller is grouping (a training
    batch or a validation split); the same value can therefore fall into a
    different bucket depending on the surrounding population.
    """
    if not math.isfinite(value):
        return "non-finite"
    bin_index = sum(value > cut for cut in (q1, median, q3))
    return CE_RANGE_LABELS[bin_index + 1]
