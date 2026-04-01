"""
Deterministic simulation engine (DSS core).

Algorithm — monthly loop t = 1..T:
  1. Apply decisions scheduled for month t.
  2. Calculate new customers with diminishing returns + capacity cap.
  3. Update customer stock.
  4. Calculate income statement and cash.
  5. Calculate derived KPIs (MRR, CAC, LTV, LTV/CAC).
"""

import math
import pandas as pd
from dss.config import CompanyConfig, SimulationResult
from dss.decisions import Scenario, DecisionChain


def run_simulation(scenario: Scenario) -> SimulationResult:
    """
    Runs the discrete simulation for the given scenario.
    Returns SimulationResult with monthly DataFrame and summary KPIs.
    """
    cfg = scenario.config
    cfg.validate()

    # Flatten DecisionChain into individual Decisions.
    # This allows Scenario.decisions to contain a mix of
    # simple Decisions and DecisionChains without changing the rest of the loop.
    _decisions = []
    for item in scenario.decisions:
        if isinstance(item, DecisionChain):
            _decisions.extend(item.decisions)
        else:
            _decisions.append(item)

    arpu            = cfg.initial_arpu
    churn           = cfg.monthly_churn_rate
    marketing_spend = cfg.marketing_spend_monthly
    capacity        = cfg.max_new_customers_capacity
    fixed_costs     = cfg.fixed_costs_monthly
    alpha           = cfg.marketing_efficiency_alpha
    beta            = cfg.acquisition_beta
    gross_margin    = cfg.gross_margin

    customers = float(cfg.initial_customers)
    cash      = cfg.initial_cash

    rows = []

    for t in range(1, cfg.horizon_months + 1):

        # 1. Apply decisions for this month
        for d in _decisions:
            if d.month == t:
                arpu            = arpu * (1 + d.arpu_change_pct / 100.0)
                marketing_spend = marketing_spend * (1 + d.marketing_spend_change_pct / 100.0)
                capacity        = capacity + d.capacity_change_abs
                fixed_costs     = fixed_costs + d.fixed_cost_change_abs
                churn           = max(0.0, min(1.0, churn + d.churn_change_abs))
                gross_margin    = max(0.0, min(1.0, gross_margin + d.gross_margin_change_abs))

                # Churn penalty for price increase (exponential model)
                # Only triggered if the increase is positive and exceeds the threshold.
                # Exponential formula: penalty = sensitivity * (exp(increase*2) - 1) * 0.05
                # Behaviour: small increases → mild penalty;
                #            large increases → severe penalty (non-linear).
                # Example: +5% → ~0.16pp | +20% → ~1.5pp | +90% → ~7.6pp
                arpu_increase_pct = d.arpu_change_pct / 100.0
                if arpu_increase_pct > cfg.price_churn_threshold:
                    churn_penalty = cfg.price_churn_sensitivity * (
                        math.exp(arpu_increase_pct * 2) - 1
                    ) * 0.05
                    churn = min(1.0, churn + churn_penalty)

        # Clamp to valid ranges
        arpu            = max(0.0, arpu)
        marketing_spend = max(0.0, marketing_spend)
        capacity        = max(0, capacity)
        fixed_costs     = max(0.0, fixed_costs)

        # 2. Acquisition with diminishing returns, capacity cap, and seasonality
        #
        # Formula: NewCustomers_t = min(alpha * spend^beta * seasonality[t-1], capacity)
        #
        # Why spend^beta instead of linear spend:
        #   - The first euros of marketing are the most efficient
        #   - Each additional euro yields slightly less (market saturates)
        #   - beta=0.8: doubling the budget → x2^0.8=x1.74 customers (not x2)
        #   - beta=1 restores the linear model (special case)
        # Seasonality:
        #   - Multiplies acquisition by the factor for month t (1-indexed)
        #   - With seasonality=[1.0]*12 and beta=1 the result is identical to original
        season            = cfg.seasonality_factors[t - 1]
        new_customers_raw = alpha * (marketing_spend ** beta) * season
        new_customers     = min(new_customers_raw, float(capacity))

        # 3. Update customer stock
        churned   = customers * churn
        # Assumption: customers acquired in month t do not churn that same month
        # — churn is applied only to the cohort existing at the start of the month.
        # Equivalent to a minimum 1-month contract, standard in SaaS B2B models.
        customers = customers * (1.0 - churn) + new_customers
        customers = max(0.0, customers)

        # 4. Financial statements
        revenue     = customers * arpu
        cogs        = revenue * (1.0 - gross_margin)
        total_costs = cogs + fixed_costs + marketing_spend
        ebitda      = revenue - total_costs

        # One-time costs (one_time_cost): direct cash deduction this month only.
        # Not counted in EBITDA or total_costs (extra-operational).
        one_time = sum(d.one_time_cost for d in _decisions if d.month == t)

        cash        = cash + revenue - total_costs - one_time

        # 5. Derived KPIs
        mrr = customers * arpu

        if new_customers > 0:
            cac = marketing_spend / new_customers
        else:
            cac = float("nan")

        if churn > 0:
            lifetime = 1.0 / churn
        else:
            lifetime = 9999.0

        ltv     = arpu * gross_margin * lifetime
        ltv_cac = ltv / cac if (cac > 0 and not pd.isna(cac)) else float("nan")

        rows.append({
            "month":           t,
            "customers":       round(customers, 2),
            "new_customers":   round(new_customers, 2),
            "churned":         round(churned, 2),
            "arpu":            round(arpu, 2),
            "churn_rate":      round(churn, 4),
            "mrr":             round(mrr, 2),
            "revenue":         round(revenue, 2),
            "cogs":            round(cogs, 2),
            "fixed_costs":     round(fixed_costs, 2),
            "marketing_spend": round(marketing_spend, 2),
            "total_costs":     round(total_costs, 2),
            "ebitda":          round(ebitda, 2),
            "one_time_cost":   round(one_time, 2),
            "cash":            round(cash, 2),
            "cac":             round(cac, 2) if not pd.isna(cac) else float("nan"),
            "ltv":             round(ltv, 2),
            "ltv_cac":         round(ltv_cac, 2) if not pd.isna(ltv_cac) else float("nan"),
        })

    df      = pd.DataFrame(rows)
    summary = _compute_summary(df, scenario.name)

    return SimulationResult(
        scenario_name=scenario.name,
        monthly_df=df,
        summary=summary,
    )


def _compute_summary(df, name):
    """
    Scalar summary KPIs from the monthly DataFrame.
    Uses column 'month' (1..12) not the index (0..11) for runway.
    """
    cash_series     = df["cash"]
    mask            = cash_series < 0
    negative_months = df.loc[mask, "month"].tolist()
    runway          = negative_months[0] if negative_months else int(df["month"].iloc[-1])
    final           = df.iloc[-1]

    # ── NRR (Net Revenue Retention) ─────────────────────────────────────────
    # Measures the initial cohort only: what fraction of its MRR is retained
    # after 12 months of churn and ARPU changes.
    # Formula: NRR = Π(1 - churn_t) × (arpu_12 / arpu_1) × 100
    # New customers acquired via marketing are NOT included in the calculation.
    arpu_initial = df["arpu"].iloc[0]
    arpu_final   = df["arpu"].iloc[-1]
    retention    = (1.0 - df["churn_rate"]).prod()
    if arpu_initial > 0:
        nrr = retention * (arpu_final / arpu_initial) * 100.0
    else:
        nrr = 100.0

    result = dict()
    result["scenario"]           = name
    result["cash_12"]            = final["cash"]
    result["customers_12"]       = final["customers"]
    result["mrr_12"]             = final["mrr"]
    result["ebitda_12"]          = final["ebitda"]
    result["total_revenue"]      = df["revenue"].sum()
    result["total_ebitda"]       = df["ebitda"].sum()
    result["avg_cac"]            = df["cac"].mean(skipna=True)
    result["avg_ltv_cac"]        = df["ltv_cac"].mean(skipna=True)
    result["cash_runway_months"] = runway
    result["insolvent"]          = bool((cash_series < 0).any())
    result["nrr"]                = round(nrr, 1)
    return result
