"""Tests for camera motion analysis and extended motion features."""

import numpy as np
import pytest

from media_engine.extractors.motion import (
    MOTION_FEATURES_VERSION,
    MotionAnalysis,
    MotionFeatures,
    MotionSegment,
    MotionType,
    analyze_motion,
    compute_motion_features,
    motion_result_to_dict,
)

SAMPLE_FPS = 5.0


def _constant_series(n: int = 50, magnitude: float = 3.0, angle: float = 0.0):
    """A series with constant magnitude and constant direction."""
    magnitudes = np.full(n, magnitude)
    mean_xs = magnitudes * np.cos(angle)
    mean_ys = magnitudes * np.sin(angle)
    return magnitudes, mean_xs, mean_ys


# --- compute_motion_features (fast, synthetic) ---


def test_constant_motion_features():
    """Constant magnitude + direction: no variance, full consistency, no jerk."""
    magnitudes, mean_xs, mean_ys = _constant_series()
    f = compute_motion_features(magnitudes, mean_xs, mean_ys, SAMPLE_FPS)

    assert f.magnitude_mean == pytest.approx(3.0)
    assert f.magnitude_std == pytest.approx(0.0)
    assert f.magnitude_p90 == pytest.approx(3.0)
    assert f.magnitude_max == pytest.approx(3.0)
    assert f.direction_consistency == pytest.approx(1.0)
    assert f.direction_reversals_per_sec == pytest.approx(0.0)
    assert f.acceleration_mean == pytest.approx(0.0)
    assert f.jerk_max == pytest.approx(0.0)
    # Zero-variance series has no spectral energy - degenerate split
    assert f.hf_energy + f.lf_energy == pytest.approx(1.0)
    assert f.hf_lf_ratio == pytest.approx(f.hf_energy)


def test_reversing_motion_features():
    """Direction flipping 180 degrees every frame: low consistency, high reversals."""
    n = 50
    magnitudes = np.full(n, 3.0)
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    mean_xs = 3.0 * signs
    mean_ys = np.zeros(n)

    f = compute_motion_features(magnitudes, mean_xs, mean_ys, SAMPLE_FPS)

    assert f.direction_consistency < 0.1
    # Every consecutive pair reverses: (n-1) reversals over n/fps seconds
    assert f.direction_reversals_per_sec == pytest.approx((n - 1) / (n / SAMPLE_FPS))


def test_frequency_band_split():
    """Low-frequency oscillation lands in lf_energy; high-frequency in hf_energy."""
    n = 100
    t = np.arange(n) / SAMPLE_FPS

    # 0.5 Hz oscillation (below the 2 Hz split)
    lf_magnitudes = 3.0 + np.sin(2 * np.pi * 0.5 * t)
    _, mean_xs, mean_ys = _constant_series(n)
    f_lf = compute_motion_features(lf_magnitudes, mean_xs, mean_ys, SAMPLE_FPS)
    assert f_lf.lf_energy > 0.9
    assert f_lf.hf_lf_ratio < 0.1

    # 2.4 Hz oscillation (above the split, below Nyquist 2.5 Hz)
    hf_magnitudes = 3.0 + np.sin(2 * np.pi * 2.4 * t)
    f_hf = compute_motion_features(hf_magnitudes, mean_xs, mean_ys, SAMPLE_FPS)
    assert f_hf.hf_energy > 0.9
    assert f_hf.hf_lf_ratio > 0.9

    # Energies always normalized
    assert f_lf.hf_energy + f_lf.lf_energy == pytest.approx(1.0)
    assert f_hf.hf_energy + f_hf.lf_energy == pytest.approx(1.0)


def test_jerk_detects_spike():
    """A sudden magnitude spike produces nonzero acceleration and jerk."""
    magnitudes, mean_xs, mean_ys = _constant_series(50, magnitude=1.0)
    magnitudes = magnitudes.copy()
    magnitudes[25:30] = 8.0  # sudden yank

    f = compute_motion_features(magnitudes, mean_xs, mean_ys, SAMPLE_FPS)

    assert f.acceleration_mean > 0.0
    assert f.jerk_max > 0.0
    assert f.magnitude_max == pytest.approx(8.0)
    assert f.magnitude_std > 0.0


def test_short_series_degenerate_defaults():
    """Series below MIN_FEATURE_SAMPLES gets documented deterministic defaults."""
    magnitudes, mean_xs, mean_ys = _constant_series(3, magnitude=2.0)
    f = compute_motion_features(magnitudes, mean_xs, mean_ys, SAMPLE_FPS)

    assert f.magnitude_mean == pytest.approx(2.0)
    assert f.magnitude_std == 0.0
    assert f.magnitude_p90 == pytest.approx(2.0)
    assert f.magnitude_max == pytest.approx(2.0)
    assert f.direction_consistency == 1.0
    assert f.direction_reversals_per_sec == 0.0
    assert f.acceleration_mean == 0.0
    assert f.jerk_max == 0.0
    assert f.hf_energy == 0.0
    assert f.lf_energy == 1.0
    assert f.hf_lf_ratio == 0.0


def test_magnitude_mean_override():
    """The magnitude_mean override keeps parity with a segment's existing intensity."""
    magnitudes, mean_xs, mean_ys = _constant_series()
    f = compute_motion_features(magnitudes, mean_xs, mean_ys, SAMPLE_FPS, magnitude_mean=2.5)
    assert f.magnitude_mean == pytest.approx(2.5)


def test_features_deterministic():
    """Identical input produces identical output (no random state)."""
    rng = np.random.default_rng(42)
    magnitudes = rng.uniform(0.5, 6.0, 200)
    mean_xs = rng.uniform(-3, 3, 200)
    mean_ys = rng.uniform(-3, 3, 200)

    f1 = compute_motion_features(magnitudes, mean_xs, mean_ys, SAMPLE_FPS)
    f2 = compute_motion_features(magnitudes.copy(), mean_xs.copy(), mean_ys.copy(), SAMPLE_FPS)

    assert f1 == f2


def test_bounded_ranges():
    """direction_consistency and band energies stay in [0, 1] on noisy input."""
    rng = np.random.default_rng(7)
    magnitudes = rng.uniform(0.1, 10.0, 300)
    mean_xs = rng.uniform(-5, 5, 300)
    mean_ys = rng.uniform(-5, 5, 300)

    f = compute_motion_features(magnitudes, mean_xs, mean_ys, SAMPLE_FPS)

    assert 0.0 <= f.direction_consistency <= 1.0
    assert 0.0 <= f.hf_energy <= 1.0
    assert 0.0 <= f.lf_energy <= 1.0
    assert 0.0 <= f.hf_lf_ratio <= 1.0
    assert f.hf_energy + f.lf_energy == pytest.approx(1.0)


# --- motion_result_to_dict (fast, synthetic) ---


def _synthetic_analysis() -> MotionAnalysis:
    magnitudes, mean_xs, mean_ys = _constant_series()
    features = compute_motion_features(magnitudes, mean_xs, mean_ys, SAMPLE_FPS, magnitude_mean=3.0)
    return MotionAnalysis(
        duration=10.0,
        fps=25.0,
        primary_motion=MotionType.PAN_LEFT,
        segments=[MotionSegment(start=0.0, end=10.0, motion_type=MotionType.PAN_LEFT, intensity=3.0, features=features)],
        avg_intensity=3.0,
        is_stable=False,
        magnitude_p90_overall=3.0,
        jerk_max_overall=0.0,
        hf_lf_ratio_overall=0.0,
    )


def test_result_dict_includes_features():
    result = motion_result_to_dict(_synthetic_analysis(), include_features=True)

    assert result["features_version"] == MOTION_FEATURES_VERSION
    assert "magnitude_p90_overall" in result
    assert "jerk_max_overall" in result
    assert "hf_lf_ratio_overall" in result

    seg = result["segments"][0]
    # Existing fields unchanged
    assert seg["motion_type"] == "pan_left"
    assert seg["intensity"] == pytest.approx(3.0)
    # New fields present, magnitude_mean mirrors intensity
    assert seg["magnitude_mean"] == pytest.approx(seg["intensity"])
    for field in (
        "magnitude_std",
        "magnitude_p90",
        "magnitude_max",
        "direction_consistency",
        "direction_reversals_per_sec",
        "acceleration_mean",
        "jerk_max",
        "hf_energy",
        "lf_energy",
        "hf_lf_ratio",
        "features_version",
    ):
        assert field in seg


def test_result_dict_features_disabled():
    """include_features=False yields exactly the pre-1.1 shape."""
    result = motion_result_to_dict(_synthetic_analysis(), include_features=False)

    assert set(result.keys()) == {"duration", "fps", "primary_motion", "avg_intensity", "is_stable", "segments"}
    assert set(result["segments"][0].keys()) == {"start", "end", "motion_type", "intensity"}


# --- analyze_motion end to end (slow, needs video) ---


@pytest.mark.slow
def test_analyze_motion_features(test_video_path):
    """All segments carry extended features with valid ranges."""
    result = analyze_motion(test_video_path)

    assert result.features_version == MOTION_FEATURES_VERSION
    assert result.magnitude_p90_overall >= 0.0
    assert result.jerk_max_overall >= 0.0
    assert 0.0 <= result.hf_lf_ratio_overall <= 1.0

    assert len(result.segments) > 0
    for seg in result.segments:
        f = seg.features
        assert isinstance(f, MotionFeatures)
        assert f.magnitude_mean == pytest.approx(seg.intensity)
        assert 0.0 <= f.direction_consistency <= 1.0
        assert f.hf_energy + f.lf_energy == pytest.approx(1.0)
        assert f.magnitude_max >= f.magnitude_p90 >= 0.0


@pytest.mark.slow
def test_analyze_motion_deterministic(test_video_path):
    """Two runs on the same file produce identical feature values."""
    r1 = analyze_motion(test_video_path)
    r2 = analyze_motion(test_video_path)

    assert r1.magnitude_p90_overall == pytest.approx(r2.magnitude_p90_overall, abs=1e-6)
    assert r1.jerk_max_overall == pytest.approx(r2.jerk_max_overall, abs=1e-6)
    assert r1.hf_lf_ratio_overall == pytest.approx(r2.hf_lf_ratio_overall, abs=1e-6)
    assert len(r1.segments) == len(r2.segments)
    for s1, s2 in zip(r1.segments, r2.segments):
        assert s1.features is not None and s2.features is not None
        assert s1.features.magnitude_mean == pytest.approx(s2.features.magnitude_mean, abs=1e-6)
        assert s1.features.jerk_max == pytest.approx(s2.features.jerk_max, abs=1e-6)
        assert s1.features.hf_lf_ratio == pytest.approx(s2.features.hf_lf_ratio, abs=1e-6)
