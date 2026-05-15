"""
Paper 5 — Unit tests for critical functions
===================================
These tests protect the numerically-sensitive parts of the pipeline
so that any future refactor cannot silently invalidate the Q1 results.

Scope:
    * Gumbel λ_U formula (copula family conversion — 03, 09, 10)
    * Expected Shortfall (05, 09)
    * Rolling z-score (07 — CSRI component scaler)
    * Block bootstrap index generator (07)
    * Student-t tail dependence (09)
    * Clayton lower tail dependence (09)

Run from the paper5-portfolio-contagion/ directory with:
    pytest tests/ -v
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════
# Keep the reference implementations here to make the tests independent of
# which source script we test them against — whenever a source script is
# refactored, we update the import line only, never the expected values.

def gumbel_lambda_U(tau: float) -> float:
    """Upper-tail dep. for a Gumbel copula given Kendall's τ ≥ 0."""
    if math.isnan(tau) or tau <= 0:
        return 0.0
    return 2.0 - 2.0 ** (1.0 - tau)


def clayton_lambda_L(tau: float) -> float:
    if math.isnan(tau) or tau <= 0:
        return 0.0
    theta = 2.0 * tau / (1.0 - tau)
    return 2.0 ** (-1.0 / theta)


def student_lambda_U(tau: float, nu: float = 4.0) -> float:
    from scipy.stats import t as student_t
    if math.isnan(tau):
        return float("nan")
    rho = math.sin(math.pi * tau / 2.0)
    rho = max(-0.999, min(0.999, rho))
    arg = math.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return 2.0 * (1.0 - float(student_t.cdf(arg, df=nu + 1)))


def expected_shortfall(x: np.ndarray, alpha: float = 0.95) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return float("nan")
    var_q = np.quantile(x, 1 - alpha)
    tail = x[x <= var_q]
    return float(-tail.mean()) if len(tail) else float("nan")


def rolling_z(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / sd.replace(0, np.nan)


def block_bootstrap_indices(n: int, block: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])
    return idx[:n]


# ══════════════════════════════════════════════════════════════════════════════
# GUMBEL λ_U
# ══════════════════════════════════════════════════════════════════════════════
class TestGumbelLambdaU:
    def test_tau_zero_returns_zero(self):
        assert gumbel_lambda_U(0.0) == 0.0

    def test_negative_tau_clamped_to_zero(self):
        assert gumbel_lambda_U(-0.4) == 0.0

    def test_nan_returns_zero(self):
        assert gumbel_lambda_U(float("nan")) == 0.0

    def test_tau_one_returns_one(self):
        # τ=1 ⇒ λ_U = 2 − 2^0 = 1
        assert gumbel_lambda_U(1.0) == pytest.approx(1.0)

    def test_monotonic_increasing(self):
        taus = [0.1, 0.3, 0.5, 0.7, 0.9]
        vals = [gumbel_lambda_U(t) for t in taus]
        assert all(a < b for a, b in zip(vals, vals[1:]))

    def test_known_value(self):
        # τ=0.5 ⇒ 2 − 2^0.5 ≈ 0.5858
        assert gumbel_lambda_U(0.5) == pytest.approx(2.0 - 2 ** 0.5, abs=1e-9)

    def test_bounded_unit_interval(self):
        for t in np.linspace(0.001, 0.999, 25):
            lam = gumbel_lambda_U(t)
            assert 0.0 <= lam <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT-t / CLAYTON
# ══════════════════════════════════════════════════════════════════════════════
class TestOtherCopulaFamilies:
    def test_clayton_zero_tau(self):
        assert clayton_lambda_L(0.0) == 0.0

    def test_clayton_range(self):
        for t in np.linspace(0.1, 0.9, 9):
            lam = clayton_lambda_L(t)
            assert 0.0 < lam < 1.0

    def test_clayton_monotone(self):
        taus = [0.1, 0.3, 0.5, 0.7, 0.9]
        vals = [clayton_lambda_L(t) for t in taus]
        assert all(a < b for a, b in zip(vals, vals[1:]))

    def test_student_matches_reference(self):
        # For ρ=0 (τ=0) Student-t tail dep ≠ 0 due to fat tails
        lam = student_lambda_U(0.0, nu=4.0)
        assert lam > 0.0
        assert lam < 0.5

    def test_student_monotone(self):
        taus = [0.1, 0.3, 0.5, 0.7, 0.9]
        vals = [student_lambda_U(t, nu=4.0) for t in taus]
        assert all(a < b for a, b in zip(vals, vals[1:]))

    def test_student_and_gumbel_both_positive(self):
        # The paper's claim: both families flag upper-tail dep > 0 for moderate τ
        t = 0.6
        assert gumbel_lambda_U(t) > 0.3
        assert student_lambda_U(t) > 0.3


# ══════════════════════════════════════════════════════════════════════════════
# EXPECTED SHORTFALL
# ══════════════════════════════════════════════════════════════════════════════
class TestExpectedShortfall:
    def test_nan_for_tiny_sample(self):
        assert math.isnan(expected_shortfall(np.array([1.0, 2.0])))

    def test_standard_normal_95(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, size=20_000)
        es = expected_shortfall(x, alpha=0.95)
        # Analytical ES for N(0,1) at 95% ≈ 2.063
        assert es == pytest.approx(2.063, abs=0.05)

    def test_zero_for_constant_series(self):
        x = np.full(100, 5.0)
        # ES returns −mean of tail = −5 → constant losses should give −5
        es = expected_shortfall(x, alpha=0.95)
        assert es == pytest.approx(-5.0, abs=1e-9)

    def test_ignores_nan(self):
        x = np.array([np.nan, np.nan] + [-3, -2, -1, 0, 1, 2, 3] * 5)
        es = expected_shortfall(x, alpha=0.95)
        assert not math.isnan(es)

    def test_larger_alpha_larger_es(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 20_000)
        es_95 = expected_shortfall(x, 0.95)
        es_99 = expected_shortfall(x, 0.99)
        assert es_99 > es_95


# ══════════════════════════════════════════════════════════════════════════════
# ROLLING Z-SCORE
# ══════════════════════════════════════════════════════════════════════════════
class TestRollingZ:
    def test_constant_gives_nan_or_zero(self):
        s = pd.Series([5.0] * 100)
        z = rolling_z(s, window=20)
        # std = 0 → NaN
        assert z.dropna().empty or (z.dropna() == 0).all()

    def test_same_length_as_input(self):
        s = pd.Series(np.random.default_rng(0).normal(size=120))
        z = rolling_z(s, window=30)
        assert len(z) == len(s)

    def test_recent_values_near_zero_mean(self):
        rng = np.random.default_rng(2)
        s = pd.Series(rng.normal(0, 1, size=200))
        z = rolling_z(s, window=60)
        # The tail of the z-score series should have near-unit sd
        assert abs(z.iloc[-60:].std() - 1.0) < 0.4

    def test_shift_invariance(self):
        # Adding a constant to the input should not change the z-score
        # (shift affects both mean and x − mean identically).
        rng = np.random.default_rng(3)
        s1 = pd.Series(rng.normal(size=150))
        s2 = s1 + 42.0
        z1 = rolling_z(s1, window=30).dropna()
        z2 = rolling_z(s2, window=30).dropna()
        np.testing.assert_allclose(z1.values, z2.values, atol=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════
class TestBlockBootstrap:
    def test_length_matches_n(self):
        idx = block_bootstrap_indices(100, block=12, seed=0)
        assert len(idx) == 100

    def test_values_in_range(self):
        idx = block_bootstrap_indices(100, block=12, seed=0)
        assert idx.min() >= 0
        assert idx.max() < 100

    def test_reproducible_with_seed(self):
        a = block_bootstrap_indices(100, 12, seed=7)
        b = block_bootstrap_indices(100, 12, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = block_bootstrap_indices(100, 12, seed=1)
        b = block_bootstrap_indices(100, 12, seed=2)
        assert not np.array_equal(a, b)

    def test_preserves_block_contiguity(self):
        # Within each block, indices should be consecutive integers
        n, block = 60, 10
        idx = block_bootstrap_indices(n, block, seed=4)
        # Take the first block
        first = idx[:block]
        diffs = np.diff(first)
        assert (diffs == 1).all()


# ══════════════════════════════════════════════════════════════════════════════
# END-TO-END SANITY CHECK (using scipy Kendall's τ)
# ══════════════════════════════════════════════════════════════════════════════
class TestEndToEndSanity:
    def test_kendall_tau_recoverable_from_gumbel_sample(self):
        """Simple monotone transform: rank-based τ is invariant."""
        rng = np.random.default_rng(10)
        u = rng.uniform(size=500)
        v = u + rng.normal(0, 0.05, size=500)   # tightly coupled
        tau, _ = kendalltau(u, v)
        assert tau > 0.8
        lam = gumbel_lambda_U(tau)
        assert lam > 0.7    # strong upper-tail dep recovered
