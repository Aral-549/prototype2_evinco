"""Unit tests for the statistical protocol (Spec Sections 2.5 and 2.7)."""

import numpy as np
import pytest

from app.paimana_statistics import (
    continuous_cost_overrun_report,
    delong_test_paired_auc,
    mean_absolute_error,
    quantile_pinball_loss,
    r_squared,
    root_mean_squared_error,
)
from scipy import stats as scipy_stats


class TestDeLongPairedAUC:
    def test_identical_scores_zero_z_one_p(self):
        rng = np.random.default_rng(42)
        y = np.array([1] * 20 + [0] * 30)
        scores = rng.random(50)
        res = delong_test_paired_auc(y, scores, scores.copy())
        assert res.auc_1 == pytest.approx(res.auc_2)
        assert res.z_statistic == 0.0
        assert res.p_value == pytest.approx(1.0)

    def test_perfect_separation_auc_one(self):
        y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        res = delong_test_paired_auc(y, scores, np.zeros(10))
        assert res.auc_1 == pytest.approx(1.0)
        assert res.z_statistic > 0

    def test_matches_mann_whitney_auc(self):
        rng = np.random.default_rng(7)
        y = np.array([1] * 15 + [0] * 25)
        s1 = rng.random(40)
        res = delong_test_paired_auc(y, s1, s1 + 0.01)
        # Empirical AUC must equal the Mann-Whitney U statistic / (m*n)
        from scipy.stats import mannwhitneyu

        u = mannwhitneyu(s1[y == 1], s1[y == 0], alternative="two-sided").statistic
        assert res.auc_1 == pytest.approx(u / (15 * 25), rel=1e-9)

    def test_z_statistic_matches_two_sample_construction(self):
        # For independent (uncorrelated) score vectors with disjoint samples,
        # DeLong must recover a sane Z consistent with a normal approximation.
        rng = np.random.default_rng(3)
        y = np.array([1] * 30 + [0] * 50)
        s_good = np.where(y == 1, rng.random(80) + 0.5, rng.random(80))
        s_bad = rng.random(80)
        res = delong_test_paired_auc(y, s_good, s_bad)
        assert res.auc_1 > res.auc_2
        assert res.z_statistic > 2.0
        expected_p = 2 * (1 - scipy_stats.norm.cdf(abs(res.z_statistic)))
        assert res.p_value == pytest.approx(expected_p, rel=1e-9)

    def test_variance_components_sanity(self):
        rng = np.random.default_rng(11)
        y = np.array([1] * 25 + [0] * 40)
        s = rng.random(65)
        res = delong_test_paired_auc(y, s, s * 0.99)
        assert res.var_1 > 0
        assert res.var_2 > 0
        # Identical ranking structure -> near-perfect positive covariance
        assert res.cov_12 > 0

    def test_requires_both_classes(self):
        y = np.ones(10, dtype=int)
        s = np.random.default_rng(1).random(10)
        with pytest.raises(ValueError, match="positive.*negative"):
            delong_test_paired_auc(y, s, s)

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="identical shapes"):
            delong_test_paired_auc(
                np.array([1, 0, 1]),
                np.array([0.5, 0.6]),
                np.array([0.5, 0.6, 0.7]),
            )

    def test_ties_contribute_half(self):
        # m = n = 1, scores equal -> psi = 0.5 -> AUC 0.5
        res = delong_test_paired_auc(np.array([1, 0]), np.array([0.7, 0.7]), np.array([0.1, 0.2]))
        assert res.auc_1 == pytest.approx(0.5)


class TestContinuousMetrics:
    def test_rmse(self):
        y = np.array([1.0, 2.0, 3.0])
        p = np.array([1.0, 2.0, 5.0])
        assert root_mean_squared_error(y, p) == pytest.approx(np.sqrt(4.0 / 3.0))

    def test_mae(self):
        y = np.array([1.0, 2.0, 3.0])
        p = np.array([2.0, 2.0, 2.0])
        assert mean_absolute_error(y, p) == pytest.approx(2.0 / 3.0)

    def test_r_squared_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r_squared(y, y) == pytest.approx(1.0)

    def test_r_squared_constant_baseline_is_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.full(3, 2.0)  # mean predictor
        assert r_squared(y, pred) == pytest.approx(0.0, abs=1e-12)

    def test_r_squared_worse_than_mean_negative(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([10.0, 10.0, 10.0])
        assert r_squared(y, pred) < 0.0

    def test_pinball_loss_median_underprediction(self):
        # tau=0.5: loss = 0.5 * |y - q| -> MAE/2
        y = np.array([10.0])
        q = np.array([6.0])
        assert quantile_pinball_loss(y, q, 0.5) == pytest.approx(2.0)

    def test_pinball_loss_asymmetry(self):
        y = np.array([10.0])
        # Under-prediction at tau=0.9 penalized more than over-prediction
        under = quantile_pinball_loss(y, np.array([5.0]), 0.9)
        over = quantile_pinball_loss(y, np.array([15.0]), 0.9)
        assert under == pytest.approx(0.9 * 5.0)
        assert over == pytest.approx(0.1 * 5.0)
        assert under > over

    def test_pinball_invalid_tau_rejected(self):
        with pytest.raises(ValueError, match="tau"):
            quantile_pinball_loss(np.array([1.0]), np.array([1.0]), 1.5)

    def test_full_report_with_quantiles(self):
        rng = np.random.default_rng(5)
        y = rng.normal(20.0, 10.0, size=200)
        pred = y + rng.normal(0, 5.0, size=200)
        report = continuous_cost_overrun_report(
            y, pred, quantile_predictions={0.10: pred - 8, 0.50: pred, 0.90: pred + 8}
        )
        assert set(report) == {"rmse_pct", "mae_pct", "r_squared", "pinball_loss_tau_0.10", "pinball_loss_tau_0.50", "pinball_loss_tau_0.90"}
        assert report["rmse_pct"] > report["mae_pct"]
        assert 0.0 < report["pinball_loss_tau_0.50"]
