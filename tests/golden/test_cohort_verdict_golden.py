"""FROZEN golden cases for the actionable-cohort verdict (contracts/model_training.md).

The cohort analysis exists to answer "isn't this just a deadline rule?" — which
means it is only worth anything if it is capable of answering **yes**. A check
that can only return good news is decoration.

These cases pin that the verdict text distinguishes three outcomes honestly:
a real advantage, a marginal one, and the model losing to the trivial rule.

Add cases; do not edit or delete.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))

from train_model import _cohort_verdict, actionable_cohort_analysis  # noqa: E402


def cohort_row(model_auc, rule_auc, n=6000, name="not_yet_overdue"):
    return {
        "cohort": name,
        "n": n,
        "positive_rate": 0.11,
        "auc_xgboost": model_auc,
        "auc_deadline_continuous": rule_auc,
    }


# ------------------------------------------------------------ verdict wording

def test_golden_clear_advantage_is_reported_as_such():
    v = _cohort_verdict([cohort_row(0.80, 0.73)])
    assert "beats the deadline rule" in v
    assert "0.07" in v
    assert "execution dynamics" in v


def test_golden_marginal_advantage_is_not_oversold():
    """A 0.01 AUC edge must not be described the same way as a 0.07 edge."""
    v = _cohort_verdict([cohort_row(0.74, 0.73)])
    assert "only" in v
    assert "rules engine would" in v
    assert "execution dynamics" not in v


def test_golden_model_losing_to_the_trivial_rule_is_stated_plainly():
    """The check must be able to fail. This is the whole point of it."""
    v = _cohort_verdict([cohort_row(0.70, 0.76)])
    assert "does NOT beat the deadline rule" in v
    assert "already overdue" in v
    assert "a deterministic rule would serve as well" in v


def test_golden_exact_tie_counts_as_not_beating():
    v = _cohort_verdict([cohort_row(0.7300, 0.7300)])
    assert "does NOT beat" in v


def test_golden_missing_cohort_is_reported_not_assumed():
    assert "Not evaluable" in _cohort_verdict([])
    assert "Not evaluable" in _cohort_verdict(
        [{"cohort": "not_yet_overdue", "n": 10, "auc_xgboost": None}]
    )


def test_golden_constant_rule_in_cohort_is_handled():
    """If the deadline rule is constant inside a cohort it has no rank
    information; report the model's AUC and say so rather than dividing."""
    v = _cohort_verdict(
        [{"cohort": "not_yet_overdue", "n": 500, "auc_xgboost": 0.81,
          "auc_deadline_continuous": None}]
    )
    assert "0.81" in v
    assert "constant" in v


# -------------------------------------------------------- cohort partitioning

def _frame(months_left, progress=40.0, n=None):
    n = n or len(months_left)
    return pd.DataFrame(
        {
            "months_to_revised_date": months_left,
            "physical_progress": [progress] * n,
            "n_snapshots": [8] * n,
        }
    )


def test_golden_cohorts_split_on_the_declared_deadline():
    """A project exactly at its declared date counts as overdue, matching F6."""
    rng = np.random.default_rng(7)
    months = np.concatenate([np.full(200, -5.0), np.full(200, 0.0), np.full(200, 10.0)])
    df = _frame(list(months))
    y = rng.integers(0, 2, size=len(df))
    oof = {"xgboost": rng.random(len(df)), "deadline_continuous": rng.random(len(df))}

    out = actionable_cohort_analysis(df, oof, y)
    by = {c["cohort"]: c for c in out["cohorts"]}
    # -5 and 0 are overdue (condition is <= 0); only +10 is actionable.
    assert by["already_overdue"]["n"] == 400
    assert by["not_yet_overdue"]["n"] == 200
    assert by["runway_over_6_months"]["n"] == 200


def test_golden_small_cohort_is_refused_not_scored():
    """Fewer than 50 rows must not produce an authoritative-looking AUC."""
    rng = np.random.default_rng(3)
    df = _frame([20.0] * 10 + [-1.0] * 200)
    y = rng.integers(0, 2, size=len(df))
    oof = {"xgboost": rng.random(len(df)), "deadline_continuous": rng.random(len(df))}
    out = actionable_cohort_analysis(df, oof, y)
    by = {c["cohort"]: c for c in out["cohorts"]}
    assert by["not_yet_overdue"]["status"] == "insufficient_or_single_class"
    assert "auc_xgboost" not in by["not_yet_overdue"]


def test_golden_single_class_cohort_is_refused():
    """A cohort where nothing slipped has no AUC; it must say so."""
    df = _frame([20.0] * 300)
    y = np.zeros(300, dtype=int)
    oof = {"xgboost": np.random.default_rng(1).random(300),
           "deadline_continuous": np.random.default_rng(2).random(300)}
    out = actionable_cohort_analysis(df, oof, y)
    by = {c["cohort"]: c for c in out["cohorts"]}
    assert by["not_yet_overdue"]["status"] == "insufficient_or_single_class"


def test_golden_delta_is_model_minus_rule():
    """Sign convention: positive delta always means the model won."""
    rng = np.random.default_rng(11)
    n = 400
    df = _frame([12.0] * n)
    y = rng.integers(0, 2, size=n)
    # Give xgboost a genuine edge by leaking a little signal into its scores.
    strong = y + rng.normal(0, 0.6, size=n)
    weak = rng.random(n)
    out = actionable_cohort_analysis(df, {"xgboost": strong, "deadline_continuous": weak}, y)
    row = next(c for c in out["cohorts"] if c["cohort"] == "not_yet_overdue")
    assert row["xgboost_minus_deadline_rule"] == pytest.approx(
        row["auc_xgboost"] - row["auc_deadline_continuous"], abs=1e-9
    )
    assert row["xgboost_minus_deadline_rule"] > 0
