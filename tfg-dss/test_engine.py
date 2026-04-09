"""
Unit tests for the DSS simulation engine.

Run with:  pytest test_engine.py -v
"""

import math
import pytest

from dss.config import CompanyConfig
from dss.decisions import Decision, Scenario
from dss.engine import run_simulation


# ─── helpers ─────────────────────────────────────────────────────────────────

def make_baseline() -> Scenario:
    """Fresh baseline Scenario with no decisions."""
    return Scenario(name="Baseline", config=CompanyConfig())


# ─── 1. Determinism ──────────────────────────────────────────────────────────

def test_determinism():
    """Two independent runs of the baseline must produce bit-for-bit identical results."""
    r1 = run_simulation(make_baseline())
    r2 = run_simulation(make_baseline())
    assert r1.summary == r2.summary
    assert r1.monthly_df.equals(r2.monthly_df)


# ─── 2. Baseline values ──────────────────────────────────────────────────────

def test_baseline_values():
    """
    Regression check against known-good baseline KPIs:
      Cash₁₂ ≈ 313,721  |  Customers₁₂ ≈ 106  |  MRR₁₂ ≈ 52,856
    """
    s = run_simulation(make_baseline()).summary

    assert abs(s["cash_12"] - 313_721.18) < 1.0, f"cash_12={s['cash_12']}"
    assert abs(s["customers_12"] - 105.71) < 1.0, f"customers_12={s['customers_12']}"
    assert abs(s["mrr_12"] - 52_855.63) < 10.0, f"mrr_12={s['mrr_12']}"
    assert not s["insolvent"]


# ─── 3. Churn penalty ────────────────────────────────────────────────────────

def test_churn_penalty_at_threshold_no_penalty():
    """
    ARPU +5% equals the threshold (price_churn_threshold=0.05).
    The condition is strictly-greater-than (>), so NO penalty is applied.
    """
    d = Decision(month=1, arpu_change_pct=5.0)
    result = run_simulation(Scenario(name="price5", config=CompanyConfig(), decisions=[d]))
    churn_m1 = result.monthly_df.loc[result.monthly_df["month"] == 1, "churn_rate"].iloc[0]
    assert abs(churn_m1 - 0.03) < 1e-4, f"expected base churn 0.03, got {churn_m1}"


def test_churn_penalty_above_threshold():
    """
    ARPU +15% exceeds the 5% threshold.
    Expected penalty = sensitivity * (exp(0.15*2) - 1) * 0.05
    The engine rounds churn_rate to 4 decimal places, so tolerance = 1e-4.
    """
    cfg = CompanyConfig()
    d = Decision(month=1, arpu_change_pct=15.0)
    result = run_simulation(Scenario(name="price15", config=cfg, decisions=[d]))
    churn_m1 = result.monthly_df.loc[result.monthly_df["month"] == 1, "churn_rate"].iloc[0]

    expected_penalty = cfg.price_churn_sensitivity * (math.exp(0.15 * 2) - 1) * 0.05
    expected_churn = cfg.monthly_churn_rate + expected_penalty
    assert abs(churn_m1 - expected_churn) < 1e-4, (
        f"churn_rate={churn_m1:.6f}, expected≈{expected_churn:.6f}"
    )


def test_churn_penalty_extreme_capped_at_one():
    """Extreme ARPU increase (+500%) must never push churn above 1.0."""
    d = Decision(month=1, arpu_change_pct=500.0)
    result = run_simulation(Scenario(name="price500", config=CompanyConfig(), decisions=[d]))
    churn_m1 = result.monthly_df.loc[result.monthly_df["month"] == 1, "churn_rate"].iloc[0]
    assert churn_m1 <= 1.0, f"churn_rate={churn_m1} exceeds 1.0"


# ─── 4. Capacity cap ─────────────────────────────────────────────────────────

def test_capacity_cap():
    """
    With marketing_spend=€500k (far above saturation) and capacity=5,
    new_customers must never exceed 5 in any month.
    """
    cfg = CompanyConfig(
        marketing_spend_monthly=500_000,
        max_new_customers_capacity=5,
    )
    result = run_simulation(Scenario(name="cap_test", config=cfg))
    max_acquired = result.monthly_df["new_customers"].max()
    assert max_acquired <= 5.0, f"new_customers exceeded capacity: max={max_acquired}"


# ─── 5. One-time costs ───────────────────────────────────────────────────────

def test_one_time_cost_reduces_cash_not_ebitda():
    """
    A one-time cost of €10,000 in month 3 must:
      - Reduce Cash₁₂ by exactly €10,000 (extra-operational, not in total_costs).
      - Leave cumulative EBITDA unchanged (not counted in the income statement).
    """
    one_time_amount = 10_000.0

    r_base = run_simulation(make_baseline())
    d_ot = Decision(month=3, one_time_cost=one_time_amount)
    r_ot = run_simulation(
        Scenario(name="one_time", config=CompanyConfig(), decisions=[d_ot])
    )

    cash_diff = r_base.summary["cash_12"] - r_ot.summary["cash_12"]
    assert abs(cash_diff - one_time_amount) < 0.01, (
        f"Expected cash reduction of {one_time_amount}, got {cash_diff}"
    )

    ebitda_diff = abs(r_base.summary["total_ebitda"] - r_ot.summary["total_ebitda"])
    assert ebitda_diff < 0.01, (
        f"total_ebitda should be identical, diff={ebitda_diff}"
    )


# ─── 6. Cash accounting identity ─────────────────────────────────────────────

def _assert_cash_identity(scenario: Scenario, tol: float = 0.10) -> None:
    """
    Cash₁₂ = initial_cash + Σ(revenue - total_costs - one_time_cost).
    Tolerance of 0.10 accounts for accumulated floating-point rounding
    when summing 12 months of rounded column values.
    """
    result = run_simulation(scenario)
    df = result.monthly_df
    expected = (
        scenario.config.initial_cash
        + df["revenue"].sum()
        - df["total_costs"].sum()
        - df["one_time_cost"].sum()
    )
    actual = df["cash"].iloc[-1]
    assert abs(actual - expected) < tol, (
        f"Cash identity violated: actual={actual:.4f}, expected={expected:.4f}, "
        f"diff={abs(actual - expected):.6f}"
    )


def test_cash_identity_baseline():
    """Cash identity holds for the unmodified baseline."""
    _assert_cash_identity(make_baseline())


def test_cash_identity_with_decisions():
    """Cash identity holds when decisions modify ARPU, marketing spend, fixed costs, and one-time costs."""
    decisions = [
        Decision(month=1, arpu_change_pct=10.0, marketing_spend_change_pct=20.0),
        Decision(month=6, one_time_cost=5_000.0, fixed_cost_change_abs=1_000.0),
    ]
    _assert_cash_identity(
        Scenario(name="with_decisions", config=CompanyConfig(), decisions=decisions)
    )
