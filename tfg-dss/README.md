# Decision Support System for SaaS B2B

**Final Year Project (TFG) — Data Analytics**
Supervisor: Alfredo Álvarez Pickman
University: IE University

A deterministic decision support system (DSS) for B2B SaaS companies that translates natural language managerial decisions into quantitative financial projections.

---

## Overview

This system allows non-technical managers to type decisions in plain language (e.g. *"raise prices by 15% in March"*) and instantly receive a 12-month financial simulation with key metrics: cash flow, MRR, EBITDA, NRR, and Rule of 40. The NLP layer uses a large language model purely as a structured translator, while all financial logic runs on a deterministic engine with no black-box components.

---

## Architecture

```
tfg-dss/
├── dss/
│   ├── __init__.py        # Package init
│   ├── config.py          # CompanyConfig, SimulationResult, seasonality & price-churn params
│   ├── decisions.py       # Decision, DecisionChain, Scenario dataclasses
│   ├── engine.py          # Deterministic simulation engine (monthly loop)
│   ├── analysis.py        # Scenario comparison, OAT sensitivity, Monte Carlo, Rule of 40
│   ├── explain.py         # Contrastive explanation engine (no LLM)
│   ├── plots.py           # Matplotlib visualisations
│   └── nlp_parser.py      # NLP layer — Groq API (llama-3.3-70b) as JSON translator
├── app.py                 # Streamlit web interface (primary interface)
├── main.py                # Headless simulation — 4 predefined scenarios
├── nlp_interface.py       # Terminal NLP interface
└── README.md
```

### Module descriptions

| Module | Responsibility |
|--------|---------------|
| `config.py` | Defines `CompanyConfig` (all baseline parameters including seasonality factors and price-churn sensitivity) and `SimulationResult` |
| `decisions.py` | `Decision` dataclass encodes a single managerial intervention; `DecisionChain` enables causally linked multi-step decisions |
| `engine.py` | Runs the deterministic monthly loop: customer acquisition with diminishing returns, B2B seasonality, price-dependent churn, and NRR computation |
| `analysis.py` | One-at-a-time (OAT) sensitivity analysis, Monte Carlo simulation, scenario comparison, and `compute_rule_of_40` |
| `explain.py` | Generates contrastive bullet-point explanations and the executive summary card without calling any LLM |
| `nlp_parser.py` | Single LLM call to Groq: converts free-text input into a validated `Decision` JSON; raises `ambiguity_warning` for vague inputs |
| `app.py` | Streamlit interface: session history, verdict card (Recommended / Caution / Not Recommended), KPI metrics with deltas (NRR, Rule of 40), Plotly charts, OAT sensitivity in sidebar, contextual risk warnings, and multi-scenario comparison with grouped bar charts and CSV export |

### Key features implemented

- **Deterministic engine with diminishing returns** — customer acquisition follows `NewCustomers = min(α · spend^β, capacity)` with β = 0.8, preventing unrealistic linear scaling of marketing spend.
- **B2B seasonality** — 12 configurable monthly multipliers applied to acquisition (default profile: strong Q1, slow summer, Q4 peak).
- **Price-dependent churn** — ARPU increases above a configurable threshold trigger an automatic churn penalty proportional to the price rise.
- **NRR and Rule of 40** — computed automatically for every simulation and displayed with industry benchmarks (Snowflake, Datadog, BVP, McKinsey).
- **DecisionChain** — allows causally linked decisions across different months (e.g. server purchase in month 4 → capacity increase in month 5).
- **Streamlit interface** — session history, financial verdict card (Recommended / Cautious / Not Recommended), OAT sensitivity sidebar summary and contextual risk warnings, Monte Carlo expander showing Cash₁₂ distribution with percentiles and insolvency probability, OAT sensitivity ranking with interactive Plotly chart and local elasticity table, and multi-decision comparison with grouped bar charts.

---

## Decision flow

```
User input (natural language)
        ↓
NLP layer (Groq API — single call)
        ↓
Validated Decision JSON
        ↓
Deterministic engine (monthly loop × 12)
        ↓
SimulationResult (KPIs, monthly DataFrame)
        ↓
Analysis layer (OAT, Monte Carlo, Rule of 40, NRR)
        ↓
Explain layer (contrastive bullets, verdict)
        ↓
Streamlit UI (charts, verdict card, session history)
```

---

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure the Groq API key

The system uses the Groq API to parse natural language decisions. The key must **never** be hardcoded. Set it as an environment variable before running:

**Mac / Linux:**
```bash
export GROQ_API_KEY="gsk_..."
```

**Windows (cmd):**
```cmd
set GROQ_API_KEY="gsk_..."
```

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="gsk_..."
```

Get a free key at: https://console.groq.com/keys

> If the variable is not set, the Streamlit interface will show a warning and a sidebar text field to enter the key temporarily (in-memory only).

---

## Running the system

**Streamlit web interface (primary — recommended):**
```bash
streamlit run app.py
```

**Headless simulation — 4 predefined scenarios:**
```bash
python main.py
```

**Terminal NLP interface:**
```bash
export GROQ_API_KEY="gsk_..."
python nlp_interface.py
```

---

## Baseline parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `initial_customers` | 100 | Starting customer count |
| `initial_arpu` | €500 | Average monthly revenue per customer |
| `monthly_churn_rate` | 3% | Monthly cancellation rate |
| `marketing_efficiency_alpha` | 0.004 | Marketing efficiency coefficient |
| `acquisition_beta` | 0.8 | Diminishing returns exponent (1.0 = linear) |
| `monthly_capacity` | 30 | Maximum new customers acquirable per month |
| `initial_cash` | €150,000 | Starting cash balance |
| `gross_margin` | 75% | Gross margin |
| `fixed_costs` | €20,000/mo | Monthly fixed operating costs |
| `marketing_spend` | €5,000/mo | Monthly marketing budget |

---

## Design decisions

### Why a deterministic engine instead of ML?

The goal of a DSS is not prediction accuracy but *causal transparency*. Managers need to understand *why* a metric changes, not just *what* it will be. A deterministic engine makes every output fully traceable to its inputs, which is essential for decision support and managerial trust. ML models would introduce opaqueness without adding explanatory value for this use case.

### Why a Beta distribution for Monte Carlo?

The Beta distribution is bounded on [0, 1], making it naturally suited to model uncertainty in rates (churn, margin) without producing physically impossible values (negative rates or rates above 100%). For spend-based parameters, a log-normal distribution is applied instead, reflecting the positive-only and right-skewed nature of costs.

### Why β = 0.8 for acquisition?

Empirical marketing literature consistently documents sub-linear returns on advertising spend. A β of 1.0 would mean doubling the budget always doubles new customers — unrealistic at scale. β = 0.8 models the gradual saturation effect: early spend is highly efficient, but marginal returns decline as the addressable audience shrinks or channels become crowded. The value 0.8 is a conservative, widely cited estimate for digital B2B channels.

---

## Known limitations

- **Assumes manager-supplied parameters are correct** — the model does not validate whether α, β, or churn rates reflect the company's actual data. Garbage in, garbage out.
- **No competition or demand elasticity** — the engine does not model competitor responses to price changes or market-level demand curves. Price increases are assumed to affect only churn, not acquisition volume.
- **Complex causal chains require manual decomposition** — decisions with non-linear interdependencies (e.g. "hire a sales team, train them for two months, then double capacity") must be broken into sequential `DecisionChain` steps; the NLP parser cannot always infer this structure automatically.
- **Parameters need calibration against real data** — the baseline values are illustrative defaults. Meaningful projections require fitting α and β to historical acquisition data and verifying churn rates against CRM records.

---

## Demo scenarios (recommended for presentation)

The following inputs demonstrate the system's key capabilities:

| Input | What it demonstrates |
|-------|---------------------|
| `"Raise prices by 15% in month 3"` | Price-churn trade-off |
| `"Raise prices by 60% in month 2"` | Double verdict — financially positive, strategically risky |
| `"Double marketing budget from June"` | Diminishing returns on acquisition spend |
| `"Churn increases 5% from month 4"` | OAT identifies churn as most critical driver |
| `"Buy servers for €30,000 in month 4 which allows us to increase capacity 30% from month 5"` | DecisionChain — investment with delayed return |
| `"Increase marketing 20% in month 5 and buy equipment for €10,000 in month 3"` | Independent multi-decision simulation |

---

## Academic contribution

> The key contribution of this work is the integration of an NLP layer over a deterministic simulation engine, enabling non-technical managers to access quantitative financial analysis through natural language input.