"""
nlp_interface.py — Interactive natural language interface (Groq API)
Run this file separately to use the NLP mode.

Usage:
    python nlp_interface.py
"""

import copy
import os
from dss.config import CompanyConfig
from dss.decisions import Scenario
from dss.engine import run_simulation
from dss.analysis import sensitivity_analysis
from dss.explain import (
    explain_scenario, print_explanation,
    build_executive_summary, print_executive_summary,
)
from dss.nlp_parser import nlp_to_decision
from dss import plots

# ── Groq API key from environment variable ────────────────────────────────────
# Configure before running:
#   Mac/Linux:  export GROQ_API_KEY="gsk_..."
#   Windows:    set GROQ_API_KEY="gsk_..."
# Get a free key at: https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Baseline configuration ────────────────────────────────────────────────────
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
res_baseline      = run_simulation(baseline_scenario)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def _show_decision(decision):
    """Prints the parameters of the interpreted decision."""
    print("  ✅ Decision interpreted:")
    print(f"     📅 Application month     : {decision.month}")
    print(f"     💰 ARPU change           : {decision.arpu_change_pct:+.1f}%")
    print(f"     📣 Marketing change      : {decision.marketing_spend_change_pct:+.1f}%")
    print(f"     👥 Capacity change       : {decision.capacity_change_abs:+d} slots")
    print(f"     🏢 Fixed cost change     : €{decision.fixed_cost_change_abs:+,.0f}")
    print(f"     📉 Churn change          : {decision.churn_change_abs:+.3f}")
    print(f"     🏷️  Label                 : {decision.label}")


def _short_verdict(bs: dict, sc: dict) -> str:
    """Returns a short verdict label for the comparison table."""
    delta_cash = sc["cash_12"] - bs["cash_12"]
    delta_mrr  = sc["mrr_12"]  - bs["mrr_12"]
    cash_pct   = (delta_cash / abs(bs["cash_12"])) * 100 if bs["cash_12"] != 0 else 0
    if sc["insolvent"]:
        return "❌ NOT REC."
    elif cash_pct < -20.0:
        return "❌ NOT REC."
    elif delta_cash < 0 and delta_mrr > 0:
        return "⚠️  CAUTION"
    else:
        return "✅ REC."


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — SIMULATE A SINGLE DECISION
# ─────────────────────────────────────────────────────────────────────────────

def single_decision_mode():
    """Interprets a natural language decision, simulates it, and shows the executive summary."""
    user_input = input("\n🤖 Enter your decision: ").strip()
    if not user_input:
        return

    print("\n⏳ Interpreting your decision...\n")

    try:
        # 1. NLP: text → Decision
        decision, _, ambiguity_warning = nlp_to_decision(user_input, GROQ_API_KEY)
        if decision is None:
            print("\n  ❌ The NLP assistant could not interpret the decision.")
            print("     Try being more specific. Example: 'Raise prices by 5% in month 3'")
            return

        # 2. Show interpretation
        _show_decision(decision)

        # 3. NLP ambiguity warning (if the LLM detected ambiguity)
        if ambiguity_warning:
            print(f"\n  ⚠️  {ambiguity_warning}")

        # 4. Confirmation
        confirm = input("\nConfirm this decision? (y/n): ").strip().lower()
        if confirm != "y":
            print("\n  ❌ Decision cancelled.")
            return

        print("\n🚀 Running simulation...\n")

        # 5. Build scenario and simulate
        user_scenario = Scenario(
            name=decision.label[:50] if decision.label else "User Scenario",
            config=copy.deepcopy(baseline_cfg),
            decisions=[decision],
        )
        res_user = run_simulation(user_scenario)

        # 6. Executive summary for the manager
        summary_lines = build_executive_summary(res_baseline, res_user, decision)
        print_executive_summary(summary_lines)

        # 7. Detailed analysis (contrastive bullets)
        _, sensitivity_rank = sensitivity_analysis(baseline_scenario)
        bullets = explain_scenario(res_baseline, res_user, sensitivity_rank, decision=decision)
        print_explanation(bullets, decision.label, "Baseline")

        # 8. Charts
        plots.plot_cash([res_baseline, res_user])
        plots.plot_customers([res_baseline, res_user])
        plots.plot_mrr([res_baseline, res_user])

    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        print("     Try rephrasing the decision more concretely.")
        print("     Example: 'Raise prices by 5% in month 3'")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — COMPARE TWO DECISIONS
# ─────────────────────────────────────────────────────────────────────────────

def comparison_mode():
    """Interprets two decisions, simulates them, and shows a comparison table."""
    print("\n📊 COMPARISON MODE — Compare two decisions against each other\n")
    print("  Examples:")
    print("  🅰️  'Raise prices by 10% in month 3'")
    print("  🅱️  'Double the marketing budget from June'\n")

    input_a = input("🅰️  First decision: ").strip()
    if not input_a:
        return
    input_b = input("🅱️  Second decision: ").strip()
    if not input_b:
        return

    print("\n⏳ Interpreting and simulating both decisions...\n")

    try:
        # 1. NLP for both decisions
        dec_a, _, warn_a = nlp_to_decision(input_a, GROQ_API_KEY)
        dec_b, _, warn_b = nlp_to_decision(input_b, GROQ_API_KEY)
        if dec_a is None or dec_b is None:
            print("\n  ❌ The NLP assistant could not interpret one or both decisions.")
            print("     Try rephrasing them more concretely.")
            return

        # 2. Show both interpretations
        print("  ─── DECISION A ───────────────────────────────────────")
        _show_decision(dec_a)
        if warn_a:
            print(f"\n  ⚠️  {warn_a}")

        print("\n  ─── DECISION B ───────────────────────────────────────")
        _show_decision(dec_b)
        if warn_b:
            print(f"\n  ⚠️  {warn_b}")

        # 3. Joint confirmation
        confirm = input("\nConfirm both decisions and compare them? (y/n): ").strip().lower()
        if confirm != "y":
            print("\n  ❌ Comparison cancelled.")
            return

        print("\n🚀 Simulating...\n")

        # 4. Simulate both scenarios
        sc_a = Scenario(
            name="Decision A",
            config=copy.deepcopy(baseline_cfg),
            decisions=[dec_a],
        )
        sc_b = Scenario(
            name="Decision B",
            config=copy.deepcopy(baseline_cfg),
            decisions=[dec_b],
        )
        res_a = run_simulation(sc_a)
        res_b = run_simulation(sc_b)

        # 5. Individual verdicts
        v_a = _short_verdict(res_baseline.summary, res_a.summary)
        v_b = _short_verdict(res_baseline.summary, res_b.summary)

        # 6. Comparison table
        W = 65
        print(f"\n{'═' * W}")
        print(f"  {'DECISION COMPARISON':^{W-4}}")
        print(f"{'═' * W}")

        print(f"\n  🅰️  {dec_a.label}")
        print(f"      Cash month 12  : €{res_a.summary['cash_12']:>12,.0f}")
        print(f"      MRR month 12   : €{res_a.summary['mrr_12']:>12,.0f}")
        print(f"      Cumul. EBITDA  : €{res_a.summary['total_ebitda']:>12,.0f}")
        print(f"      Verdict        :  {v_a}")

        print(f"\n  🅱️  {dec_b.label}")
        print(f"      Cash month 12  : €{res_b.summary['cash_12']:>12,.0f}")
        print(f"      MRR month 12   : €{res_b.summary['mrr_12']:>12,.0f}")
        print(f"      Cumul. EBITDA  : €{res_b.summary['total_ebitda']:>12,.0f}")
        print(f"      Verdict        :  {v_b}")

        print(f"\n{'─' * W}")

        # 7. Winner
        cash_a   = res_a.summary["cash_12"]
        cash_b   = res_b.summary["cash_12"]
        ebitda_a = res_a.summary["total_ebitda"]
        ebitda_b = res_b.summary["total_ebitda"]

        gana_cash  = "A" if cash_a   >= cash_b   else "B"
        gana_ebit  = "A" if ebitda_a >= ebitda_b else "B"

        if gana_cash == gana_ebit:
            ganadora = gana_cash
            label_g  = dec_a.label if ganadora == "A" else dec_b.label
            diff_cash = abs(cash_a - cash_b)
            print(
                f"\n  🏆 WINNER: Decision {ganadora} — {label_g}\n"
                f"     Better in cash (+€{diff_cash:,.0f}) and cumulative EBITDA."
            )
        else:
            # Tie: one wins on cash, the other on EBITDA
            label_cash = dec_a.label if gana_cash == "A" else dec_b.label
            label_ebit = dec_a.label if gana_ebit == "A" else dec_b.label
            print(
                f"\n  ⚖️  MIXED RESULT:\n"
                f"     💰 Higher cash    → Decision {gana_cash}: {label_cash}\n"
                f"     📊 Higher EBITDA  → Decision {gana_ebit}: {label_ebit}\n"
                f"     → Prioritise decision {gana_cash} if you need liquidity,\n"
                f"       or decision {gana_ebit} if you seek cumulative profitability."
            )

        print(f"{'═' * W}\n")

        # 8. Comparative charts
        plots.plot_cash([res_baseline, res_a, res_b])
        plots.plot_mrr([res_baseline, res_a, res_b])

    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        print("     Try rephrasing the decisions more concretely.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────────────────────────────────────

if not GROQ_API_KEY:
    print("\n  ⚠️  GROQ_API_KEY is not set.")
    print("     Configure the environment variable before running:")
    print("     Mac/Linux:  export GROQ_API_KEY=\"gsk_...\"")
    print("     Windows:    set GROQ_API_KEY=\"gsk_...\"")
    print("     Get a free key at: https://console.groq.com/keys\n")
    raise SystemExit(1)

print("=" * 65)
print("  DSS — NATURAL LANGUAGE DECISION SIMULATOR")
print("=" * 65)
print("  Examples of what you can type:")
print("  → 'Raise prices by 10% in month 3'")
print("  → 'Double marketing from June and hire 5 sales reps'")
print("  → 'Reduce churn by 1% with support improvements from month 2'")
print("=" * 65)

while True:
    print("\n  What would you like to do?")
    print("  [1] Simulate a decision")
    print("  [2] Compare two decisions")
    print("  [3] Exit")

    opcion = input("\n  Choose an option (1/2/3): ").strip()

    if opcion == "1":
        single_decision_mode()
    elif opcion == "2":
        comparison_mode()
    elif opcion == "3":
        print("\n  👋 Goodbye.\n")
        break
    else:
        print("  Invalid option. Enter 1, 2 or 3.")
