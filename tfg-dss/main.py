"""
main.py — DSS for SaaS B2B
Runs simulations, sensitivity analysis, Monte Carlo, and explanations.
"""

import copy
import pandas as pd

from dss.config import CompanyConfig
from dss.decisions import Decision, Scenario
from dss.engine import run_simulation
from dss.analysis import compare_scenarios, sensitivity_analysis, run_monte_carlo
from dss.explain import explain_scenario, print_explanation
from dss import plots

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:,.2f}".format)


# ─────────────────────────────────────────────────────────────────────────────
# (a) BASELINE
# Regime: marketing-constrained (alpha*spend = 0.004*5000 = 20 < capacity=30)
# → alpha and marketing spend are the real acquisition drivers.
# ─────────────────────────────────────────────────────────────────────────────
baseline_cfg = CompanyConfig(
    initial_customers=100,
    initial_arpu=500.0,
    monthly_churn_rate=0.03,
    gross_margin=0.75,
    fixed_costs_monthly=20_000.0,
    marketing_spend_monthly=5_000.0,
    marketing_efficiency_alpha=0.004,
    acquisition_beta=0.8,
    max_new_customers_capacity=30,
    initial_cash=150_000.0,
    horizon_months=12,
)
baseline_scenario = Scenario(name="Baseline", config=baseline_cfg, decisions=[])


# ─────────────────────────────────────────────────────────────────────────────
# (b) SCENARIO A — Price increase +5% ARPU from month 3
# ─────────────────────────────────────────────────────────────────────────────
price_up_scenario = Scenario(
    name="Scenario A — Price +5%",
    config=copy.deepcopy(baseline_cfg),
    decisions=[
        Decision(
            month=3,
            arpu_change_pct=+5.0,
            label="Price increase: ARPU +5% from month 3",
        )
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# (c) SCENARIO B — Growth Push (marketing +20% + hiring)
# ─────────────────────────────────────────────────────────────────────────────
growth_scenario = Scenario(
    name="Scenario B — Growth Push",
    config=copy.deepcopy(baseline_cfg),
    decisions=[
        Decision(
            month=2,
            marketing_spend_change_pct=+20.0,
            capacity_change_abs=+15,
            fixed_cost_change_abs=+3_000.0,
            label="Marketing +20%, capacity +15 slots, fixed costs +€3k (month 2)",
        )
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# (d) STRESS SCENARIO — High churn (10%)
# ─────────────────────────────────────────────────────────────────────────────
stress_cfg = copy.deepcopy(baseline_cfg)
stress_cfg.monthly_churn_rate = 0.10
stress_scenario = Scenario(
    name="Scenario C — Stress (churn 10%)",
    config=stress_cfg,
    decisions=[],
)


# ─────────────────────────────────────────────────────────────────────────────
# RUN SIMULATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  RUNNING SIMULATIONS")
print("=" * 65)

res_baseline = run_simulation(baseline_scenario)
res_price    = run_simulation(price_up_scenario)
res_growth   = run_simulation(growth_scenario)
res_stress   = run_simulation(stress_scenario)

all_results = [res_baseline, res_price, res_growth, res_stress]


# ─────────────────────────────────────────────────────────────────────────────
# MONTHLY KPIs — BASELINE
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Monthly KPIs — Baseline ──")
cols = ["month", "customers", "new_customers", "arpu", "mrr", "ebitda", "cash", "cac", "ltv_cac"]
print(res_baseline.monthly_df[cols].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Scenario Comparison ──")
comparison = compare_scenarios(all_results)
key_cols   = ["cash_12", "customers_12", "mrr_12", "total_ebitda",
              "cash_runway_months", "insolvent",
              "Δcash_12", "Δcustomers_12", "Δmrr_12"]
available  = [c for c in key_cols if c in comparison.columns]
print(comparison[available].to_string())


# ─────────────────────────────────────────────────────────────────────────────
# OAT SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── OAT Sensitivity Analysis ──")
sensitivity_df, sensitivity_rank = sensitivity_analysis(baseline_scenario)
print(sensitivity_df.to_string(index=False))
print("\n── OAT Ranking ──")
print(sensitivity_rank.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Monte Carlo (N=500, Baseline) ──")
mc_result = run_monte_carlo(baseline_scenario, n_simulations=500, seed=42)
print(f"  Simulations         : {mc_result['n_simulations']}")
print(f"  P5   Cash12         : EUR {mc_result['percentiles'][5]:,.0f}")
print(f"  P50  Cash12 (med.)  : EUR {mc_result['percentiles'][50]:,.0f}")
print(f"  P95  Cash12         : EUR {mc_result['percentiles'][95]:,.0f}")
print(f"  P(insolvency)       : {mc_result['prob_insolvent']*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# EXPLANATIONS
# ─────────────────────────────────────────────────────────────────────────────
_dec = lambda sc: sc.decisions[0] if len(sc.decisions) == 1 else None
bullets_A = explain_scenario(res_baseline, res_price,  sensitivity_rank, decision=_dec(price_up_scenario))
bullets_B = explain_scenario(res_baseline, res_growth, sensitivity_rank, decision=_dec(growth_scenario))
bullets_C = explain_scenario(res_baseline, res_stress, sensitivity_rank, decision=_dec(stress_scenario))

print_explanation(bullets_A, res_price.scenario_name,  res_baseline.scenario_name)
print_explanation(bullets_B, res_growth.scenario_name, res_baseline.scenario_name)
print_explanation(bullets_C, res_stress.scenario_name, res_baseline.scenario_name)


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Generating charts ──\n")
plots.plot_cash(all_results)
plots.plot_customers(all_results)
plots.plot_mrr(all_results)
plots.plot_monte_carlo(mc_result)
plots.plot_sensitivity(sensitivity_rank)

print("\n✅ Done! PNG files saved to the project folder.\n")
