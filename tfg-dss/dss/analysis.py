"""
Scenario comparison, OAT sensitivity analysis, and Monte Carlo simulation.
"""

import copy
import numpy as np
import pandas as pd
from typing import List

from dss.config import SimulationResult
from dss.decisions import Scenario
from dss.engine import run_simulation


# ─────────────────────────────────────────────────────────────────────────────
# 1. SCENARIO COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_scenarios(results: List[SimulationResult]) -> pd.DataFrame:
    """
    Comparative KPI table across scenarios.
    The first in the list is the baseline; deltas are vs. it.
    """
    summaries = [r.summary for r in results]
    df        = pd.DataFrame(summaries).set_index("scenario")

    baseline_name = results[0].scenario_name
    numeric_cols  = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        df[f"Δ{col}"] = df[col] - df.loc[baseline_name, col]

    df.loc[baseline_name, [c for c in df.columns if c.startswith("Δ")]] = None
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. OAT SENSITIVITY (One-At-a-Time)
# ─────────────────────────────────────────────────────────────────────────────

OAT_PARAMS = {
    "monthly_churn_rate":         {"eps": 0.01,  "type": "absolute"},  # ±1 pp
    "initial_arpu":               {"eps": 0.05,  "type": "relative"},  # ±5%
    "marketing_efficiency_alpha": {"eps": 0.05,  "type": "relative"},  # ±5%
    "marketing_spend_monthly":    {"eps": 0.10,  "type": "relative"},  # ±10%
    "fixed_costs_monthly":        {"eps": 0.10,  "type": "relative"},  # ±10%
}


def sensitivity_analysis(baseline_scenario: Scenario):
    """
    OAT analysis: perturbs each parameter ±ε, re-runs the simulation,
    and records the impact on Cash_12 and runway.
    Returns (df_detail, df_ranking).
    """
    base_result = run_simulation(baseline_scenario)
    base_cash   = base_result.summary["cash_12"]
    base_runway = base_result.summary["cash_runway_months"]

    records = []

    for param, spec in OAT_PARAMS.items():
        for sign, label in [(+1, "+ε"), (-1, "−ε")]:

            cfg_copy = copy.deepcopy(baseline_scenario.config)
            original = getattr(cfg_copy, param)

            perturbed = (original * (1 + sign * spec["eps"])
                         if spec["type"] == "relative"
                         else original + sign * spec["eps"])

            if param == "monthly_churn_rate":
                perturbed = max(0.0, min(1.0, perturbed))

            setattr(cfg_copy, param, perturbed)

            sc = Scenario(
                name=f"{param}_{label}",
                config=cfg_copy,
                decisions=copy.deepcopy(baseline_scenario.decisions),
            )
            result = run_simulation(sc)

            records.append({
                "parameter":       param,
                "perturbation":    label,
                "perturbed_value": round(perturbed, 6),
                "cash_12":         round(result.summary["cash_12"], 2),
                "Δcash_12":        round(result.summary["cash_12"] - base_cash, 2),
                "runway_months":   result.summary["cash_runway_months"],
                "Δrunway":         result.summary["cash_runway_months"] - base_runway,
                "insolvent":       result.summary["insolvent"],
            })

    df = pd.DataFrame(records)
    df["abs_impact"] = df["Δcash_12"].abs()

    rank = (
        df.groupby("parameter")["abs_impact"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"abs_impact": "mean_abs_Δcash_12"})
    )
    rank["rank"] = range(1, len(rank) + 1)

    # Approximate local OAT elasticity — not a strict elasticity
    # but a finite-perturbation approximation.
    # Valid for relative comparison across drivers.
    _denom_cash = base_cash if abs(base_cash) > 1e-9 else 1e-9

    def _eps_rel(row):
        spec     = OAT_PARAMS[row["parameter"]]
        original = getattr(baseline_scenario.config, row["parameter"])
        if spec["type"] == "relative":
            return spec["eps"]
        else:
            return spec["eps"] / max(abs(original), 1e-9)

    rank["local_elasticity"] = rank.apply(
        lambda row: round(
            (row["mean_abs_Δcash_12"] / _denom_cash) / _eps_rel(row),
            2,
        ),
        axis=1,
    )

    return df, rank


# ─────────────────────────────────────────────────────────────────────────────
# 3. RULE OF 40
# ─────────────────────────────────────────────────────────────────────────────

def compute_rule_of_40(result: SimulationResult) -> dict:
    """
    Rule of 40 = MRR Growth (%) + EBITDA Margin (%)

    MRR Growth: percentage change in MRR from month 1 to month 12.
    EBITDA Margin: cumulative EBITDA / cumulative revenue * 100.
    Passes the threshold if Rule of 40 >= 40 (VC benchmark).
    """
    df = result.monthly_df
    mrr_inicio = df["mrr"].iloc[0]
    mrr_fin    = df["mrr"].iloc[-1]

    if mrr_inicio > 0:
        mrr_growth = ((mrr_fin - mrr_inicio) / mrr_inicio) * 100.0
    else:
        mrr_growth = 0.0

    total_rev   = result.summary["total_revenue"]
    total_ebit  = result.summary["total_ebitda"]

    if total_rev > 0:
        ebitda_margin = (total_ebit / total_rev) * 100.0
    else:
        ebitda_margin = 0.0

    rule_of_40 = mrr_growth + ebitda_margin

    return {
        "mrr_growth_pct":   round(mrr_growth, 1),
        "ebitda_margin_pct": round(ebitda_margin, 1),
        "rule_of_40":       round(rule_of_40, 1),
        "passes":           rule_of_40 >= 40,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(
    baseline_scenario: Scenario,
    n_simulations: int = 500,
    seed: int = 42,
    churn_beta_a: float = 3.0,
    churn_beta_b: float = 97.0,
    alpha_lognorm_mu: float = None,
    alpha_lognorm_sigma: float = 0.20,
) -> dict:
    """
    Monte Carlo simulation over uncertain parameters:
      - churn ~ Beta(a, b)
      - alpha ~ LogNormal(mu, sigma)
    Returns the Cash_12 distribution, P(insolvency), and percentiles.
    """
    rng = np.random.default_rng(seed)
    cfg = baseline_scenario.config

    if alpha_lognorm_mu is None:
        alpha_lognorm_mu = np.log(max(cfg.marketing_efficiency_alpha, 1e-9))

    cash_12_list   = []
    insolvent_list = []
    runway_list    = []

    for _ in range(n_simulations):
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.monthly_churn_rate         = float(rng.beta(churn_beta_a, churn_beta_b))
        cfg_copy.marketing_efficiency_alpha = float(rng.lognormal(alpha_lognorm_mu, alpha_lognorm_sigma))

        sc = Scenario(
            name="mc_run",
            config=cfg_copy,
            decisions=copy.deepcopy(baseline_scenario.decisions),
        )
        res = run_simulation(sc)
        cash_12_list.append(res.summary["cash_12"])
        insolvent_list.append(res.summary["insolvent"])
        runway_list.append(res.summary["cash_runway_months"])

    arr = np.array(cash_12_list)

    return {
        "cash_12_samples": arr,
        "prob_insolvent":  float(np.mean(insolvent_list)),
        "percentiles":     {5:  float(np.percentile(arr, 5)),
                            50: float(np.percentile(arr, 50)),
                            95: float(np.percentile(arr, 95))},
        "results_df":      pd.DataFrame({"cash_12":   cash_12_list,
                                          "insolvent": insolvent_list,
                                          "runway":    runway_list}),
        "n_simulations":   n_simulations,
    }
