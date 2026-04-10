"""
tests/test_roi_scorer.py — Unit tests for backend/roi_scorer.py
================================================================
Run with:
    cd C:\\Users\\Shashank\\Desktop\\emotions
    .\\backend\\venv\\Scripts\\python.exe -m pytest tests/ -v

These tests use only random / synthetic activation arrays — no TRIBE v2
model required. They verify the mathematical correctness of the scoring
and classification logic.
"""

import sys
import os

import numpy as np
import pytest

# ── Path setup ────────────────────────────────────────────────────────────
# Ensure the project root is on sys.path so `data` and `backend` are importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.roi_scorer import (
    score_regions,
    classify,
    _REGIONS,
    _N_REGIONS,
    NEUTRAL_MARGIN,
    _REGION_INDEX_ARRAYS,
)
from data.roi_map import REGION_VERTICES

# ── Fixtures ──────────────────────────────────────────────────────────────

N_VERTS = 20_484

@pytest.fixture
def zero_activations():
    """All-zero activation vector — uniform across all regions."""
    return np.zeros(N_VERTS, dtype=np.float32)


@pytest.fixture
def random_activations():
    """Random activation vector with fixed seed for reproducibility."""
    rng = np.random.default_rng(seed=2026)
    return rng.random(N_VERTS).astype(np.float32)


@pytest.fixture
def boosted_broca():
    """
    Activation vector where Broca vertices are set to 1.0 and all others to 0.0.
    Broca should win decisively.
    """
    act = np.zeros(N_VERTS, dtype=np.float32)
    broca_idx = _REGION_INDEX_ARRAYS["Broca"]
    act[broca_idx] = 1.0
    return act


@pytest.fixture
def boosted_dmn():
    """Activation vector that strongly favours the DMN."""
    act = np.zeros(N_VERTS, dtype=np.float32)
    dmn_idx = _REGION_INDEX_ARRAYS["DMN"]
    act[dmn_idx] = 1.0
    return act


# ── score_regions tests ───────────────────────────────────────────────────

class TestScoreRegions:

    def test_returns_all_regions(self, random_activations):
        """All five tracked regions must appear in the output dict."""
        scores = score_regions(random_activations)
        assert set(scores.keys()) == set(_REGIONS), (
            f"Expected regions {_REGIONS}, got {list(scores.keys())}"
        )

    def test_scores_sum_to_one(self, random_activations):
        """Softmax output must sum to 1.0 within floating-point tolerance."""
        scores = score_regions(random_activations)
        total = sum(scores.values())
        assert abs(total - 1.0) < 1e-5, f"Scores sum to {total}, expected ~1.0"

    def test_scores_sum_to_one_zero_input(self, zero_activations):
        """Even with all-zero input, softmax probabilities must sum to 1."""
        scores = score_regions(zero_activations)
        total = sum(scores.values())
        assert abs(total - 1.0) < 1e-5, f"Scores sum to {total} for zero input"

    def test_scores_all_positive(self, random_activations):
        """Softmax values must always be strictly positive."""
        scores = score_regions(random_activations)
        for region, score in scores.items():
            assert score > 0.0, f"Region {region} has non-positive score {score}"

    def test_scores_all_positive_zero_input(self, zero_activations):
        """Softmax of all-equal inputs gives uniform distribution > 0."""
        scores = score_regions(zero_activations)
        for region, score in scores.items():
            assert score > 0.0, f"Region {region} has non-positive score for zero input"

    def test_zero_input_uniform_distribution(self, zero_activations):
        """
        All-zero activation → all regions have exactly the same mean (0.0)
        → softmax gives a uniform distribution of 1/N_REGIONS each.
        """
        scores = score_regions(zero_activations)
        expected = 1.0 / _N_REGIONS
        for region, score in scores.items():
            assert abs(score - expected) < 1e-6, (
                f"Region {region}: expected {expected:.4f}, got {score:.4f}"
            )

    def test_boosted_broca_wins(self, boosted_broca):
        """When only Broca vertices are activated, Broca must have highest score."""
        scores = score_regions(boosted_broca)
        winner = max(scores, key=scores.__getitem__)
        assert winner == "Broca", (
            f"Expected Broca to win, got {winner}. Scores: {scores}"
        )

    def test_boosted_dmn_wins(self, boosted_dmn):
        """When only DMN vertices are activated, DMN must have highest score."""
        scores = score_regions(boosted_dmn)
        winner = max(scores, key=scores.__getitem__)
        assert winner == "DMN", (
            f"Expected DMN to win, got {winner}. Scores: {scores}"
        )

    def test_negative_activations_handled(self):
        """
        Negative BOLD values (plausible from TRIBE v2) must not cause NaN/Inf.
        Softmax is shift-invariant, so negatives are fine.
        """
        rng = np.random.default_rng(seed=99)
        act = (rng.random(N_VERTS) - 0.5).astype(np.float32)  # values in [-0.5, 0.5]
        scores = score_regions(act)
        for region, score in scores.items():
            assert np.isfinite(score), f"Region {region} got non-finite score {score}"
        assert abs(sum(scores.values()) - 1.0) < 1e-5

    def test_large_activations_handled(self):
        """Large activation values must not cause overflow in softmax."""
        act = np.full(N_VERTS, 1e6, dtype=np.float32)
        scores = score_regions(act)
        total = sum(scores.values())
        assert abs(total - 1.0) < 1e-5, f"Scores sum to {total} for large input"
        for score in scores.values():
            assert np.isfinite(score)

    def test_invalid_shape_raises(self):
        """Wrong-length input should raise ValueError."""
        bad_input = np.zeros(100, dtype=np.float32)
        with pytest.raises((ValueError, IndexError)):
            score_regions(bad_input)


# ── classify tests ────────────────────────────────────────────────────────

class TestClassify:

    def test_returns_tuple(self, random_activations):
        """classify() must return a 2-tuple."""
        result = classify(random_activations)
        assert isinstance(result, tuple) and len(result) == 2

    def test_region_is_valid(self, random_activations):
        """
        Returned region must be one of the known regions or 'Neutral'.
        """
        valid_regions = set(_REGIONS) | {"Neutral"}
        region, _ = classify(random_activations)
        assert region in valid_regions, f"Unknown region returned: {region}"

    def test_confidence_in_range(self, random_activations):
        """Confidence must be in [0, 1]."""
        _, confidence = classify(random_activations)
        assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"

    def test_zero_input_returns_neutral(self, zero_activations):
        """
        All-zero input → uniform distribution → winning score = 1/N_REGIONS.
        This is below the NEUTRAL_MARGIN threshold, so result should be 'Neutral'.
        """
        region, _ = classify(zero_activations)
        uniform_score = 1.0 / _N_REGIONS
        neutral_threshold = uniform_score + NEUTRAL_MARGIN
        # Only assert Neutral if uniform score truly < threshold
        if uniform_score < neutral_threshold:
            assert region == "Neutral", (
                f"Expected 'Neutral' for zero input, got {region}"
            )

    def test_boosted_broca_classified_correctly(self, boosted_broca):
        """Decisively boosted Broca should not be labelled Neutral."""
        region, confidence = classify(boosted_broca)
        assert region == "Broca", f"Expected 'Broca', got '{region}'"
        assert confidence > 1.0 / _N_REGIONS + NEUTRAL_MARGIN, (
            f"Confidence {confidence} too low for clearly boosted region"
        )

    def test_boosted_dmn_classified_correctly(self, boosted_dmn):
        """Decisively boosted DMN should not be labelled Neutral."""
        region, confidence = classify(boosted_dmn)
        assert region == "DMN", f"Expected 'DMN', got '{region}'"

    def test_confidence_matches_score_regions(self, random_activations):
        """
        The confidence returned by classify() must equal the score for the
        winning region as returned by score_regions().
        """
        region, confidence = classify(random_activations)
        if region != "Neutral":
            scores = score_regions(random_activations)
            expected_conf = round(scores[region], 6)
            assert abs(confidence - expected_conf) < 1e-5, (
                f"classify() confidence {confidence} != score_regions()[{region}] {expected_conf}"
            )


# ── Vertex atlas sanity checks ────────────────────────────────────────────

class TestROIAtlas:

    def test_all_regions_have_vertices(self):
        """Every region in REGION_VERTICES must have at least one vertex."""
        for region, vertices in REGION_VERTICES.items():
            assert len(vertices) > 0, f"Region '{region}' has no vertex indices"

    def test_vertex_indices_in_range(self):
        """All vertex indices must be within fsaverage5 bounds [0, 20483]."""
        for region, vertices in REGION_VERTICES.items():
            arr = np.array(vertices)
            assert arr.min() >= 0,      f"Region '{region}' has negative index {arr.min()}"
            assert arr.max() <= 20483,  f"Region '{region}' index {arr.max()} exceeds 20483"

    def test_no_duplicate_vertices_within_region(self):
        """A single region must not list the same vertex twice."""
        for region, vertices in REGION_VERTICES.items():
            assert len(vertices) == len(set(vertices)), (
                f"Region '{region}' has duplicate vertex indices"
            )

    def test_language_rois_are_left_hemisphere(self):
        """Broca, STS, MTG parcels must only contain left-hemisphere indices (< 10242)."""
        left_only = ["Broca", "STS", "MTG"]
        for region in left_only:
            arr = np.array(REGION_VERTICES[region])
            rh_verts = arr[arr >= 10242]
            assert len(rh_verts) == 0, (
                f"Region '{region}' should be LH-only but has RH indices: {rh_verts[:5]}"
            )

    def test_bilateral_rois_have_both_hemispheres(self):
        """TPJ and DMN must have vertices from both hemispheres."""
        bilateral = ["TPJ", "DMN"]
        for region in bilateral:
            arr = np.array(REGION_VERTICES[region])
            has_lh = (arr < 10242).any()
            has_rh = (arr >= 10242).any()
            assert has_lh, f"Region '{region}' has no LH vertices"
            assert has_rh, f"Region '{region}' has no RH vertices"
