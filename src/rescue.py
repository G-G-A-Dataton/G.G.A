"""Validated prediction rules for leaderboard rescue candidates."""

import numpy as np


def threshold_predictions(probabilities, threshold):
    """Convert an aligned probability vector to binary predictions."""
    probabilities = _probability_vector(probabilities, "probabilities")
    threshold = _threshold(threshold, "threshold")
    return (probabilities >= threshold).astype(np.int8)


def band_override_predictions(
    base_probabilities,
    base_threshold,
    band_probabilities,
    band_mask,
    band_threshold,
):
    """Override a base decision only on rows covered by a gated model."""
    base_probabilities = _probability_vector(
        base_probabilities, "base_probabilities"
    )
    band_probabilities = _probability_vector(
        band_probabilities, "band_probabilities"
    )
    band_mask = np.asarray(band_mask)
    if band_mask.ndim != 1 or len(band_mask) != len(base_probabilities):
        raise ValueError("band_mask must align with base_probabilities")
    if band_mask.dtype != np.bool_:
        raise TypeError("band_mask must be a boolean vector")
    if len(band_probabilities) != int(band_mask.sum()):
        raise ValueError(
            "band_probabilities must cover every selected row exactly once"
        )

    predictions = threshold_predictions(base_probabilities, base_threshold)
    predictions[band_mask] = threshold_predictions(
        band_probabilities, band_threshold
    )
    return predictions


def prediction_summary(predictions):
    """Return stable class-balance diagnostics for a binary vector."""
    predictions = np.asarray(predictions)
    if predictions.ndim != 1 or predictions.dtype.kind not in "biu":
        raise TypeError("predictions must be a one-dimensional integer vector")
    if not np.isin(predictions, [0, 1]).all():
        raise ValueError("predictions must contain only 0 and 1")
    positives = int(np.count_nonzero(predictions))
    rows = int(len(predictions))
    return {
        "rows": rows,
        "positive_rows": positives,
        "negative_rows": rows - positives,
        "positive_rate": positives / rows if rows else 0.0,
    }


def _probability_vector(values, name):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"{name} must contain finite values in [0, 1]")
    return values


def _threshold(value, name):
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return value
