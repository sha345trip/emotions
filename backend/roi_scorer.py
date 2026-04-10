"""
roi_scorer.py — ROI Scoring Layer (Phase 3)
============================================
Maps a vertex-wise cortical activation vector (shape 20,484) to a probability
distribution over the five brain regions tracked by Emotional Weight.

Public API
----------
score_regions(vertex_activations)  →  {region: probability}   (sums to 1.0)
classify(vertex_activations)        →  (winning_region, confidence)

Design notes
------------
• Uses **softmax** normalisation so each region's score is a proper probability.
  Softmax is preferred over min-max here because it:
    - Is invariant to additive shifts (handles negative BOLD predictions)
    - Amplifies differences between regions rather than linearly compressing them
    - Matches the convention used in neural encoding benchmarks

• A sentence is labelled "Neutral" when the winning region's probability does
  not exceed 1/N_regions + NEUTRAL_MARGIN, i.e. when the distribution is nearly
  uniform across all tracked regions.

• Regions with no vertex indices (empty list) receive a raw score of 0.0 and
  are included in the softmax denominator so they never win.

References
----------
  d'Ascoli et al. (2026). TRIBE v2. Meta FAIR. (CC-BY-NC-4.0)
  Glasser et al. (2016). Nature 536, 171–178.
"""

import os
import sys
import numpy as np

# Allow this module to be imported from any working directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.roi_map import REGION_VERTICES  # {region_name: [vertex_indices]}

# ── Constants ─────────────────────────────────────────────────────────────

# Regions in a fixed order so numpy operations are deterministic
_REGIONS: list[str] = list(REGION_VERTICES.keys())
_N_REGIONS: int = len(_REGIONS)

# Vertex index arrays, pre-converted to numpy for fast slicing
_REGION_INDEX_ARRAYS: dict[str, np.ndarray] = {
    region: np.array(indices, dtype=np.int32)
    for region, indices in REGION_VERTICES.items()
}

# A sentence is "Neutral" if its winning softmax probability does not exceed
# the uniform baseline by at least this margin.
# uniform baseline = 1 / N_REGIONS  (≈ 0.20 for 5 regions)
NEUTRAL_MARGIN: float = 0.08   # winning region must score ≥ 0.28 to be labelled


# ── Core functions ────────────────────────────────────────────────────────

def _raw_region_scores(vertex_activations: np.ndarray) -> np.ndarray:
    """
    Compute the mean activation across each region's vertices.

    Parameters
    ----------
    vertex_activations : np.ndarray, shape (20484,)
        Mean BOLD prediction per fsaverage5 vertex, as returned by
        run_tribe_on_text().

    Returns
    -------
    np.ndarray, shape (N_REGIONS,)
        Raw mean activation value per region, in _REGIONS order.
    """
    raw = np.empty(_N_REGIONS, dtype=np.float64)
    for i, region in enumerate(_REGIONS):
        idx = _REGION_INDEX_ARRAYS[region]
        if idx.size == 0:
            raw[i] = 0.0
        else:
            raw[i] = vertex_activations[idx].mean()
    return raw


def _softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.

    Shifts by max(x) before exponentiating to avoid overflow with large values
    from TRIBE v2 BOLD predictions.
    """
    e = np.exp(x - x.max())
    return e / e.sum()


def score_regions(vertex_activations: np.ndarray) -> dict[str, float]:
    """
    Map a vertex activation vector to a probability distribution over regions.

    Parameters
    ----------
    vertex_activations : np.ndarray, shape (20484,)
        Mean BOLD prediction per fsaverage5 vertex (float32 or float64).

    Returns
    -------
    dict[str, float]
        {region_name: softmax_probability}
        Values are in (0, 1) and sum to exactly 1.0.

    Example
    -------
    >>> import numpy as np
    >>> act = np.zeros(20484, dtype=np.float32)
    >>> scores = score_regions(act)
    >>> assert abs(sum(scores.values()) - 1.0) < 1e-6
    >>> assert set(scores.keys()) == {'TPJ', 'MTG', 'Broca', 'STS', 'DMN'}
    """
    if vertex_activations.ndim != 1 or vertex_activations.shape[0] < max(
        idx.max() for idx in _REGION_INDEX_ARRAYS.values() if idx.size > 0
    ):
        raise ValueError(
            f"vertex_activations must be 1-D with length ≥ max vertex index. "
            f"Got shape {vertex_activations.shape}."
        )

    raw = _raw_region_scores(vertex_activations.astype(np.float64))
    probs = _softmax(raw)

    return {region: float(probs[i]) for i, region in enumerate(_REGIONS)}


def classify(vertex_activations: np.ndarray) -> tuple[str, float]:
    """
    Return the winning brain region and its confidence for a vertex activation
    vector.

    A sentence is labelled "Neutral" when no region's probability exceeds the
    uniform baseline (1 / N_REGIONS) by more than NEUTRAL_MARGIN.

    Parameters
    ----------
    vertex_activations : np.ndarray, shape (20484,)

    Returns
    -------
    (region_name, confidence) : (str, float)
        region_name  — one of 'TPJ', 'MTG', 'Broca', 'STS', 'DMN', 'Neutral'
        confidence   — softmax probability of the winning region (0.0–1.0),
                       or that of the runner-up if Neutral (for debugging)

    Example
    -------
    >>> import numpy as np
    >>> act = np.zeros(20484)
    >>> region, conf = classify(act)
    >>> region
    'Neutral'
    """
    scores = score_regions(vertex_activations)

    winning_region = max(scores, key=scores.__getitem__)
    winning_score  = scores[winning_region]

    uniform_baseline = 1.0 / _N_REGIONS
    if winning_score < uniform_baseline + NEUTRAL_MARGIN:
        return "Neutral", round(winning_score, 6)

    return winning_region, round(winning_score, 6)
