"""
app.py — Interfaz Streamlit del DSS SaaS B2B

Ejecuta con:
    streamlit run app.py
"""

import copy
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from dss.config    import CompanyConfig
from dss.decisions import Scenario, DecisionChain
from dss.engine    import run_simulation
from dss.analysis  import sensitivity_analysis, compute_rule_of_40, run_monte_carlo
from dss.explain   import explain_scenario, build_executive_summary
from dss.nlp_parser import nlp_to_decision, build_decisions_from_sub

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA (debe ser el primer comando Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "DSS · SaaS B2B Decision Simulator",
    page_icon  = "📊",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

_DEFAULT_CFG = CompanyConfig(
    initial_customers          = 100,
    initial_arpu               = 500.0,
    monthly_churn_rate         = 0.03,
    gross_margin               = 0.75,
    fixed_costs_monthly        = 20_000.0,
    marketing_spend_monthly    = 5_000.0,
    marketing_efficiency_alpha = 0.004,
    acquisition_beta           = 0.8,
    max_new_customers_capacity = 30,
    initial_cash               = 150_000.0,
    horizon_months             = 12,
)

# Paleta de colores para las trazas (baseline siempre azul index=0)
_COLORES = ["#5B9BD5", "#FF6B6B", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4", "#E91E63"]



# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — VEREDICTO
# Misma lógica que build_executive_summary en explain.py.
# ─────────────────────────────────────────────────────────────────────────────

def _veredicto(bs: dict, sc: dict, decision=None, bs_result=None, sc_result=None):
    """
    Devuelve (codigo, etiqueta, bg, border, texto_color, avisos_extra).
    ✅ RECOMENDADO   → Δcash ≥ 0, MRR no cae >5%, no insolvent, sin riesgos significativos
    ⚠️ CON CAUTELA   → Δcash < 0 pero MRR sube;
                       O gasto puntual sin retorno en MRR/clientes;
                       O MRR o clientes caen >5% respecto al baseline;
                       O churn_penalty > 0.5pp (Regla 1);
                       O 2+ avisos de riesgo activos (Regla 3)
    ❌ NO RECOMENDADO → insolvent O Δcash < −20%
    """
    delta_cash = sc["cash_12"] - bs["cash_12"]
    delta_mrr  = sc["mrr_12"] - bs["mrr_12"]
    delta_cust = sc["customers_12"] - bs["customers_12"]
    cash_pct   = (delta_cash / abs(bs["cash_12"])) * 100 if bs["cash_12"] != 0 else 0
    mrr_pct    = (delta_mrr  / bs["mrr_12"])       * 100 if bs["mrr_12"]       > 0 else 0
    cust_pct   = (delta_cust / bs["customers_12"]) * 100 if bs["customers_12"] > 0 else 0

    # ── Churn penalty computado (requiere monthly_df) ─────────────────────────
    churn_penalty_pp = 0.0
    if bs_result is not None and sc_result is not None:
        churn_penalty_pp = (
            sc_result.monthly_df["churn_rate"].mean()
            - bs_result.monthly_df["churn_rate"].mean()
        ) * 100

    # ── Avisos de riesgo activos ──────────────────────────────────────────────
    _arpu    = getattr(decision, "arpu_change_pct",           0) if decision is not None else 0
    _mkt_up  = getattr(decision, "marketing_spend_change_pct", 0) if decision is not None else 0
    _cap     = getattr(decision, "capacity_change_abs",        0) if decision is not None else 0
    _churn_r = getattr(decision, "churn_change_abs",           0) if decision is not None else 0

    aviso_arpu_precio   = _arpu > 0                   # subida de ARPU → riesgo de churn
    aviso_churn_penalty = churn_penalty_pp > 0.5      # penalización computada > 0.5pp
    aviso_capacidad     = _mkt_up > 0 and _cap == 0   # marketing sube sin ampliar capacidad
    aviso_churn_red     = _churn_r < 0                # reducción de churn sin coste modelado
    n_avisos = sum([aviso_arpu_precio, aviso_churn_penalty, aviso_capacidad, aviso_churn_red])

    # ── Veredicto base ────────────────────────────────────────────────────────
    if sc["insolvent"]:
        codigo = "NO_REC"
    elif cash_pct < -20.0:
        codigo = "NO_REC"
    # Gasto puntual sin retorno directo en MRR ni clientes → CON CAUTELA
    elif (decision is not None
            and getattr(decision, "one_time_cost", 0) > 0
            and abs(delta_mrr) < 1.0
            and abs(delta_cust) < 0.5):
        codigo = "CAUTELA"
    elif delta_cash < 0 and delta_mrr > 0:
        codigo = "CAUTELA"
    # MRR o clientes caen >5% aunque la caja se mantenga → CON CAUTELA
    # (ej. eliminar marketing ahorra costes pero destruye base de clientes)
    elif mrr_pct < -5.0 or cust_pct < -5.0:
        codigo = "CAUTELA"
    else:
        codigo = "REC"

    # ── Reglas de degradación (solo aplican si el veredicto base es REC) ─────
    # Regla 1: churn_penalty > 0.5pp → REC degrada a CON CAUTELA
    if codigo == "REC" and aviso_churn_penalty:
        codigo = "CAUTELA"
    # Regla 3: 2+ avisos activos → mínimo CON CAUTELA, nunca RECOMENDADO
    if codigo == "REC" and n_avisos >= 2:
        codigo = "CAUTELA"

    # ── Avisos extra para mostrar bajo la tarjeta (Regla 2) ──────────────────
    avisos_extra = []
    if aviso_capacidad:
        avisos_extra.append(
            "⚠️ The marketing increase may be limited by acquisition capacity"
        )

    # ── Colores según código final ────────────────────────────────────────────
    if codigo == "NO_REC":
        return "NO_REC", "❌ NOT RECOMMENDED", "#f8d7da", "#dc3545", "#721c24", avisos_extra
    if codigo == "CAUTELA":
        return "CAUTELA", "⚠️ PROCEED WITH CAUTION",   "#fff3cd", "#ffc107", "#6d4c00", avisos_extra
    return     "REC",     "✅ RECOMMENDED",   "#d4edda", "#28a745", "#155724", avisos_extra


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — PLOTLY
# ─────────────────────────────────────────────────────────────────────────────

def _fig_lineas(resultados: list, columna: str, titulo: str, ylabel: str):
    """Gráfico de líneas temporal. Baseline: trazo punteado. Resto: sólido."""
    fig = go.Figure()
    for i, res in enumerate(resultados):
        df = res.monthly_df
        fig.add_trace(go.Scatter(
            x    = df["month"].tolist(),
            y    = df[columna].tolist(),
            name = res.scenario_name,
            mode = "lines+markers",
            marker = dict(size=5),
            line   = dict(
                color = _COLORES[i % len(_COLORES)],
                width = 2,
                dash  = "dot" if i == 0 else "solid",
            ),
        ))

    if columna == "cash":
        fig.add_hline(
            y=0, line_dash="dash", line_color="red",
            annotation_text="Insolvency (€0)",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title    = titulo,
        height   = 370,
        margin   = dict(t=50, b=30, l=10, r=10),
        hovermode = "x unified",
        xaxis = dict(title="Month", tickvals=list(range(1, 13))),
        yaxis = dict(title=ylabel, tickprefix="€" if columna != "customers" else "",
                     tickformat=",.0f"),
        legend = dict(orientation="h", yanchor="bottom", y=1.02,
                      xanchor="right", x=1),
    )
    return fig


def _fig_escenario(res_b, res_u, columna: str, titulo: str, ylabel: str):
    """Gráfico de dos trazas: baseline vs escenario usuario.
    Construye cada traza de forma explícita a partir de listas Python puras
    para evitar que Plotly use el índice del pandas Series como eje X.
    """
    df_b = res_b.monthly_df
    df_u = res_u.monthly_df

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x    = df_b["month"].tolist(),
        y    = df_b[columna].tolist(),
        name = res_b.scenario_name,
        mode = "lines+markers",
        marker = dict(size=5),
        line   = dict(color=_COLORES[0], width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x    = df_u["month"].tolist(),
        y    = df_u[columna].tolist(),
        name = res_u.scenario_name,
        mode = "lines+markers",
        marker = dict(size=5),
        line   = dict(color=_COLORES[1], width=2, dash="solid"),
    ))

    if columna == "cash":
        fig.add_hline(
            y=0, line_dash="dash", line_color="red",
            annotation_text="Insolvency (€0)",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title    = titulo,
        height   = 370,
        margin   = dict(t=50, b=30, l=10, r=10),
        hovermode = "x unified",
        xaxis = dict(title="Month", tickvals=list(range(1, 13))),
        yaxis = dict(title=ylabel, tickprefix="€" if columna != "customers" else "",
                     tickformat=",.0f"),
        legend = dict(orientation="h", yanchor="bottom", y=1.02,
                      xanchor="right", x=1),
    )
    return fig


def _fig_barras(items: list, res_baseline):
    """Barras agrupadas: Caja / MRR / EBITDA para baseline + cada decisión."""
    etiquetas = [res_baseline.scenario_name] + [it["resultado"].scenario_name for it in items]
    cash      = [res_baseline.summary["cash_12"]]       + [it["resultado"].summary["cash_12"]       for it in items]
    mrr       = [res_baseline.summary["mrr_12"]]        + [it["resultado"].summary["mrr_12"]        for it in items]
    ebitda    = [res_baseline.summary["total_ebitda"]]  + [it["resultado"].summary["total_ebitda"]  for it in items]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Cash month 12",   x=etiquetas, y=cash,   marker_color="#5B9BD5"))
    fig.add_trace(go.Bar(name="MRR month 12",    x=etiquetas, y=mrr,    marker_color="#4CAF50"))
    fig.add_trace(go.Bar(name="EBITDA accum.",  x=etiquetas, y=ebitda, marker_color="#FF9800"))
    fig.update_layout(
        barmode   = "group",
        title     = "Scenario comparison — Key KPIs (€)",
        height    = 430,
        margin    = dict(t=55, b=80, l=10, r=10),
        hovermode = "x unified",
        yaxis     = dict(tickprefix="€", tickformat=",.0f"),
        xaxis     = dict(tickangle=-15),
        legend    = dict(orientation="h", yanchor="bottom", y=1.02,
                         xanchor="right", x=1),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — FORMATO MONEDA
# ─────────────────────────────────────────────────────────────────────────────

def _eur(v):         return f"€{v:,.0f}"
def _eur_delta(v):   return ("+" if v >= 0 else "-") + f"€{abs(v):,.0f}"
def _pct_delta(v):   return f"{v:+.1f}%"


def _metric_delta(value: float, threshold: float, formatter):
    """
    Devuelve (delta_str, delta_color, delta_arrow) para st.metric.
    Si abs(value) < threshold → "— sin cambio" en gris sin flecha.
    """
    if abs(value) < threshold:
        return "— no change", "off", "off"
    return formatter(value), "normal", "auto"


def _nrr_label(nrr: float) -> str:
    """Etiqueta de interpretación del NRR."""
    if nrr > 100:
        return "✅ Expansion"
    elif nrr >= 90:
        return "⚠️ Mild contraction"
    return "❌ Severe contraction"


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — inicializar UNA SOLA VEZ
# ─────────────────────────────────────────────────────────────────────────────

if "historial" not in st.session_state:
    st.session_state.historial        = []
if "idx_activo" not in st.session_state:
    st.session_state.idx_activo       = None
if "modo_comparacion" not in st.session_state:
    st.session_state.modo_comparacion = False
if "_multi_pending" not in st.session_state:
    st.session_state["_multi_pending"] = None
if "company_config" not in st.session_state:
    st.session_state.company_config   = copy.deepcopy(_DEFAULT_CFG)

# ── Baseline: calcular en primer arranque o cuando el usuario actualiza config ──
if "res_baseline" not in st.session_state or st.session_state.get("_baseline_needs_update"):
    _sc_bs = Scenario(name="Baseline",
                      config=copy.deepcopy(st.session_state.company_config),
                      decisions=[])
    st.session_state.res_baseline      = run_simulation(_sc_bs)
    _, st.session_state.sensitivity_rank = sensitivity_analysis(_sc_bs)
    st.session_state["_baseline_needs_update"] = False

# Alias locales — apuntan siempre a los valores de sesión actuales
BASELINE_CFG     = st.session_state.company_config
res_baseline     = st.session_state.res_baseline
sensitivity_rank = st.session_state.sensitivity_rank


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — historial de decisiones
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── API Key ───────────────────────────────────────────────────────────────
    _api_key_env = GROQ_API_KEY
    if not _api_key_env:
        _api_key_input = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Only stored in memory for the duration of the session.",
        )
        if _api_key_input:
            st.session_state["groq_api_key"] = _api_key_input
        if st.session_state.get("groq_api_key"):
            st.success("✅ API key configured")
        else:
            st.warning("⚠️ Set your Groq API key as an environment variable.")
    else:
        st.session_state["groq_api_key"] = _api_key_env

    st.divider()

    # ── Configurar empresa ────────────────────────────────────────────────────
    with st.expander("⚙️ Configure my company"):
        st.caption(
            "Enter your company's real data to get personalised simulations. "
            "Default values correspond to a reference SaaS B2B company."
        )

        if st.session_state.get("_show_baseline_updated"):
            st.success("✅ Baseline updated with your company data")
            st.session_state["_show_baseline_updated"] = False

        _cfg = st.session_state.company_config

        _n_clientes = st.number_input(
            "Current customers", min_value=1, max_value=100_000,
            value=int(_cfg.initial_customers), step=10,
        )
        _arpu = st.number_input(
            "Monthly ARPU (€)", min_value=1.0,
            value=float(_cfg.initial_arpu), step=10.0, format="%.0f",
        )
        _churn_pct = st.number_input(
            "Monthly churn (%)", min_value=0.1, max_value=50.0,
            value=float(_cfg.monthly_churn_rate * 100), step=0.1, format="%.1f",
        )
        _margen_pct = st.number_input(
            "Gross margin (%)", min_value=0.0, max_value=100.0,
            value=float(_cfg.gross_margin * 100), step=1.0, format="%.0f",
        )
        _fixed = st.number_input(
            "Fixed costs/month (€)", min_value=0.0,
            value=float(_cfg.fixed_costs_monthly), step=1_000.0, format="%.0f",
        )
        _mkt = st.number_input(
            "Marketing/month (€)", min_value=0.0,
            value=float(_cfg.marketing_spend_monthly), step=500.0, format="%.0f",
        )
        _caja = st.number_input(
            "Current cash (€)", min_value=0.0,
            value=float(_cfg.initial_cash), step=10_000.0, format="%.0f",
        )

        # ── Validaciones inline ───────────────────────────────────────────────
        if _fixed > 0 and _caja < _fixed * 3:
            st.warning("⚠️ Current runway below 3 months — critical situation.")
        if _churn_pct > 10:
            st.warning("⚠️ Churn > 10%/month: average customer lifetime < 10 months.")
        if _fixed > 0 and _mkt > _fixed * 2:
            st.warning("⚠️ Marketing budget very high relative to fixed costs.")
        if _n_clientes * _arpu * (_margen_pct / 100) < _fixed + _mkt:
            st.warning("⚠️ With these parameters the company runs a deficit at baseline.")

        if st.button("🔄 Update baseline", use_container_width=True, type="primary"):
            st.session_state.company_config = CompanyConfig(
                initial_customers          = int(_n_clientes),
                initial_arpu               = float(_arpu),
                monthly_churn_rate         = float(_churn_pct / 100),
                gross_margin               = float(_margen_pct / 100),
                fixed_costs_monthly        = float(_fixed),
                marketing_spend_monthly    = float(_mkt),
                marketing_efficiency_alpha = _cfg.marketing_efficiency_alpha,
                acquisition_beta           = _cfg.acquisition_beta,
                max_new_customers_capacity = _cfg.max_new_customers_capacity,
                initial_cash               = float(_caja),
                horizon_months             = 12,
            )
            st.session_state.historial        = []
            st.session_state.idx_activo       = None
            st.session_state.modo_comparacion = False
            st.session_state["_baseline_needs_update"]  = True
            st.session_state["_show_baseline_updated"]  = True
            st.rerun()

    st.divider()

    st.markdown("## 📋 Decision history")

    if not st.session_state.historial:
        st.caption("You haven't simulated any decisions yet in this session.")
    else:
        # Mostrar en orden inverso (más reciente primero)
        for i in range(len(st.session_state.historial) - 1, -1, -1):
            item = st.session_state.historial[i]
            _, etiq, bg, brd, tc, _ = _veredicto(
                res_baseline.summary, item["resultado"].summary,
                bs_result=res_baseline, sc_result=item["resultado"],
            )

            st.markdown(
                f'<div style="background:{bg}; border-left:4px solid {brd}; '
                f'padding:7px 10px; border-radius:5px; margin-bottom:5px;">'
                f'<span style="color:{tc}; font-weight:700; font-size:0.78em;">{etiq}</span><br>'
                f'<span style="color:#333; font-size:0.82em;">{item["decision"].label[:48]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("View this scenario", key=f"ver_{i}", use_container_width=True):
                st.session_state.idx_activo       = i
                st.session_state.modo_comparacion = False

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📊 Compare all", use_container_width=True, type="primary"):
                st.session_state.modo_comparacion = True
                st.session_state.idx_activo       = None
        with c2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.historial        = []
                st.session_state.idx_activo       = None
                st.session_state.modo_comparacion = False

    st.divider()
    st.caption(
        "**Most critical driver (OAT):**  \n"
        f"`{sensitivity_rank.iloc[0]['parameter']}`  \n"
        f"Mean impact: {_eur(sensitivity_rank.iloc[0]['mean_abs_Δcash_12'])} on Cash₁₂"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — cabecera
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 📊 DSS — SaaS B2B Decision Simulator")
st.caption("Decision Support System · Final Thesis · Deterministic engine · 12-month horizon")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — KPIs baseline
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("🏢 Current company state (Baseline)", expanded=True):
    b = res_baseline.summary
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Customers (month 12)",  f"{b['customers_12']:.0f}")
    k2.metric("MRR (month 12)",       _eur(b["mrr_12"]))
    k3.metric("Cash (month 12)",      _eur(b["cash_12"]))
    k4.metric("Cumulative EBITDA",   _eur(b["total_ebitda"]))
    k5.metric("Solvency 12m.",    "✅ Yes" if not b["insolvent"] else "❌ No")

    with st.container():
        st.caption(
            f"Base parameters: **{int(BASELINE_CFG.initial_customers)} customers** · "
            f"ARPU **{_eur(BASELINE_CFG.initial_arpu)}** · "
            f"Churn **{BASELINE_CFG.monthly_churn_rate*100:.0f}%/month** · "
            f"Marketing **{_eur(BASELINE_CFG.marketing_spend_monthly)}/month** · "
            f"Fixed costs **{_eur(BASELINE_CFG.fixed_costs_monthly)}/month**"
        )

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — formulario de entrada NLP
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 🤖 Simulate a decision in natural language")
st.caption(
    "Examples: *'Raise prices 10% in March'* · "
    "*'Double marketing from June and hire 5 sales reps'* · "
    "*'Reduce churn 1% from month 2'*"
)

with st.form("form_decision", clear_on_submit=False):
    user_input = st.text_input(
        "decision_input",
        placeholder="Type your management decision here…",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button(
        "▶  Simulate decision",
        use_container_width=True,
        type="primary",
    )

if submitted and user_input.strip():
    st.session_state["_multi_pending"] = None   # limpiar selección anterior
    _key_activa = st.session_state.get("groq_api_key", "")
    if not _key_activa:
        st.error("⚠️ Enter your Groq API key in the sidebar to run simulations.")
    else:
        with st.spinner("⏳ Interpreting with AI and running simulation…"):
            decision, parsed_nlp, ambiguity_warning = nlp_to_decision(user_input.strip(), _key_activa)

        if decision is None and parsed_nlp is not None:
            # ── Múltiples acciones independientes: ofrecer opciones de simulación ──
            _decs_sub = build_decisions_from_sub(parsed_nlp)
            if _decs_sub:
                st.session_state["_multi_pending"] = {
                    "texto":      user_input.strip(),
                    "decisiones": _decs_sub,
                }
            else:
                # sub_decisions vacíos o no parseables → fallback manual
                st.warning(
                    "⚠️ Could not interpret the decision automatically. "
                    "Fill in the parameters manually and click **Confirm**."
                )
                st.session_state["_fallback_activo"] = True
                st.session_state["_fallback_texto"]  = user_input.strip()

        elif decision is None:
            # NLP falló → activar formulario manual
            st.warning(
                "⚠️ No se pudo interpretar la decisión automáticamente. "
                "Completa los parámetros manualmente y pulsa **Confirmar**."
            )
            st.session_state["_fallback_activo"]  = True
            st.session_state["_fallback_texto"]   = user_input.strip()
        else:
            # ── Validaciones UI (advertencias, no bloqueantes) ─────────────
            # Para DecisionChain se verifican cada sub-decisión individualmente
            _single_decisions = (
                decision.decisions if isinstance(decision, DecisionChain) else [decision]
            )
            for _d in _single_decisions:
                _churn_final = BASELINE_CFG.monthly_churn_rate + _d.churn_change_abs
                if _churn_final > 0.5:
                    st.warning(
                        f"⚠️ The resulting churn would be **{_churn_final*100:.1f}%/month** — "
                        "an extremely high value that typically leads to rapid insolvency."
                    )
                if abs(_d.arpu_change_pct) > 50:
                    st.warning(
                        f"⚠️ ARPU change is **{_d.arpu_change_pct:+.1f}%** — "
                        "above ±50%. Please verify the value is correct."
                    )
                if _d.marketing_spend_change_pct > 200:
                    st.warning(
                        f"⚠️ Marketing increase is **+{_d.marketing_spend_change_pct:.0f}%** — "
                        "more than triple the current budget."
                    )

            # Aviso si month=1 por defecto (el usuario no especificó mes)
            _all_month1 = all(d.month == 1 for d in _single_decisions)
            _input_lower = user_input.lower()
            _mes1_mencionado = any(
                kw in _input_lower
                for kw in ("mes 1", "enero", "primer mes", "month 1", "january")
            )
            if _all_month1 and not _mes1_mencionado:
                st.info(
                    "ℹ️ No month specified — the decision applies from month 1. "
                    "You can add **'in month X'** to change when it takes effect."
                )

            # NLP OK → simular directamente
            nombre_sc = decision.label[:55] if decision.label else "Escenario Usuario"
            sc_user   = Scenario(
                name      = nombre_sc,
                config    = copy.deepcopy(BASELINE_CFG),
                decisions = [decision],
            )
            res_user = run_simulation(sc_user)
            st.session_state.historial.append({
                "texto":             user_input.strip(),
                "decision":          decision,
                "resultado":         res_user,
                "ambiguity_warning": ambiguity_warning,
            })
            st.session_state.idx_activo       = len(st.session_state.historial) - 1
            st.session_state.modo_comparacion = False


# ── Multi-action pending: UI de elección ────────────────────────────────────
if st.session_state.get("_multi_pending"):
    _mp   = st.session_state["_multi_pending"]
    _decs = _mp["decisiones"]
    _emojis_mp = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]

    # Info box con las decisiones detectadas
    _info_lines = [
        f"**You have described {len(_decs)} independent decisions.** "
        "How would you like to simulate them?", ""
    ]
    for _i, _d in enumerate(_decs):
        _pref = _emojis_mp[_i] if _i < len(_emojis_mp) else f"{_i+1}."
        _info_lines.append(f"{_pref} *{_d.label}*")
    st.info("\n".join(_info_lines))

    # Botones: "Simular juntas" + uno por decisión individual
    _mp_cols = st.columns(min(len(_decs) + 1, 4))
    _btn_juntas_mp = _mp_cols[0].button(
        "▶ Simulate together", use_container_width=True, type="primary",
        key="_btn_mp_juntas",
    )
    _btns_solo_mp = []
    for _i, _d in enumerate(_decs[:3]):
        _pref = _emojis_mp[_i] if _i < len(_emojis_mp) else f"{_i+1}."
        _b = _mp_cols[_i + 1].button(
            f"{_pref} This one only",
            help=_d.label,
            use_container_width=True,
            key=f"_btn_mp_solo_{_i}",
        )
        _btns_solo_mp.append((_i, _b))

    def _mp_simular(decisions_list, label_sc):
        _sc_mp = Scenario(
            name      = label_sc[:55],
            config    = copy.deepcopy(BASELINE_CFG),
            decisions = decisions_list,
        )
        _res_mp = run_simulation(_sc_mp)
        _dec_mp = (
            DecisionChain(decisions=decisions_list, label=f"Combined: {label_sc}")
            if len(decisions_list) > 1
            else decisions_list[0]
        )
        st.session_state.historial.append({
            "texto":             _mp["texto"],
            "decision":          _dec_mp,
            "resultado":         _res_mp,
            "ambiguity_warning": "",
        })
        st.session_state.idx_activo       = len(st.session_state.historial) - 1
        st.session_state.modo_comparacion = False
        st.session_state["_multi_pending"] = None

    if _btn_juntas_mp:
        _meses_str = ", ".join(map(str, sorted(set(_d.month for _d in _decs))))
        _mp_simular(_decs, f"Combined decision — {len(_decs)} actions in months {_meses_str}")

    for _i, _clicked in _btns_solo_mp:
        if _clicked:
            _d_solo = _decs[_i]
            _mp_simular([_d_solo], _d_solo.label)


# ── Formulario manual (fallback cuando el NLP falla) ─────────────────────────
if "historial" not in st.session_state:
    st.session_state.historial = []
if "_fallback_activo" not in st.session_state:
    st.session_state["_fallback_activo"] = False

if st.session_state.get("_fallback_activo"):
    from dss.decisions import Decision as _Decision

    st.markdown("#### 🔧 Manual parameter entry")
    with st.form("form_manual"):
        c1, c2, c3 = st.columns(3)
        fb_mes    = c1.number_input("Application month (1–12)", min_value=1, max_value=12, value=1, step=1)
        fb_arpu   = c2.number_input("ARPU change (%)", value=0.0, step=1.0)
        fb_mkt    = c3.number_input("Marketing change (%)", value=0.0, step=5.0)

        c4, c5, c6 = st.columns(3)
        fb_cap    = c4.number_input("Capacity change (slots)", value=0, step=1)
        fb_fixed  = c5.number_input("Fixed cost change (€)", value=0.0, step=500.0)
        fb_churn  = c6.number_input("Churn change (decimal, e.g. -0.01)", value=0.0, step=0.005, format="%.3f")

        fb_margin = st.number_input(
            "Gross margin change (decimal, e.g. -0.05 if production costs +5%)",
            value=0.0, step=0.01, format="%.3f",
            help="Negative if production costs rise. Positive if efficiency improves.",
        )
        fb_one_time = st.number_input(
            "One-time cost that month (€, e.g. 4000 for an ad campaign)",
            value=0.0, step=500.0, format="%.0f",
            help="Only deducted from cash that month. Non-recurring. Different from 'fixed costs'.",
        )

        fb_label  = st.text_input("Scenario label", value=st.session_state.get("_fallback_texto", "Manual decision")[:60])
        fb_ok     = st.form_submit_button("✅ Confirm and simulate", type="primary", use_container_width=True)
        fb_cancel = st.form_submit_button("Cancel")

    if fb_cancel:
        st.session_state["_fallback_activo"] = False
        st.rerun()

    if fb_ok:
        # Validaciones UI para el formulario manual
        _churn_fin_manual = BASELINE_CFG.monthly_churn_rate + float(fb_churn)
        if _churn_fin_manual > 0.5:
            st.warning(f"⚠️ Resulting churn {_churn_fin_manual*100:.1f}%/month — very high value.")
        if abs(float(fb_arpu)) > 50:
            st.warning(f"⚠️ ARPU change {float(fb_arpu):+.1f}% — above ±50%.")
        if float(fb_mkt) > 200:
            st.warning(f"⚠️ Marketing increase +{float(fb_mkt):.0f}% — more than triple the current.")

        _dec_manual = _Decision(
            month                      = int(fb_mes),
            arpu_change_pct            = float(fb_arpu),
            marketing_spend_change_pct = float(fb_mkt),
            capacity_change_abs        = int(fb_cap),
            fixed_cost_change_abs      = float(fb_fixed),
            churn_change_abs           = float(fb_churn),
            gross_margin_change_abs    = float(fb_margin),
            one_time_cost              = float(fb_one_time),
            label                      = fb_label or "Manual decision",
        )
        _sc_manual = Scenario(
            name      = _dec_manual.label[:55],
            config    = copy.deepcopy(BASELINE_CFG),
            decisions = [_dec_manual],
        )
        _res_manual = run_simulation(_sc_manual)
        st.session_state.historial.append({
            "texto":             st.session_state.get("_fallback_texto", ""),
            "decision":          _dec_manual,
            "resultado":         _res_manual,
            "ambiguity_warning": "Parameters entered manually (NLP unavailable).",
        })
        st.session_state.idx_activo        = len(st.session_state.historial) - 1
        st.session_state.modo_comparacion  = False
        st.session_state["_fallback_activo"] = False
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — resultado activo (una decisión simulada)
# ─────────────────────────────────────────────────────────────────────────────

idx = st.session_state.idx_activo
hay_resultado_activo = (
    not st.session_state.modo_comparacion
    and idx is not None
    and idx < len(st.session_state.historial)
)

if hay_resultado_activo:
    item      = st.session_state.historial[idx]
    decision  = item["decision"]
    res_user  = item["resultado"]
    amb_warn  = item["ambiguity_warning"]

    bs = res_baseline.summary
    sc = res_user.summary

    delta_cash   = sc["cash_12"]      - bs["cash_12"]
    delta_cust   = sc["customers_12"] - bs["customers_12"]
    delta_mrr    = sc["mrr_12"]       - bs["mrr_12"]
    delta_ebitda = sc["total_ebitda"] - bs["total_ebitda"]
    cash_pct     = (delta_cash / abs(bs["cash_12"])) * 100 if bs["cash_12"] != 0 else 0

    bs_df = res_baseline.monthly_df
    sc_df = res_user.monthly_df
    coste_extra = (
        (sc_df["marketing_spend"] + sc_df["fixed_costs"])
        - (bs_df["marketing_spend"] + bs_df["fixed_costs"])
    ).mean()

    codigo_v, etiq_v, bg_v, brd_v, tc_v, avisos_v = _veredicto(
        bs, sc, decision, bs_result=res_baseline, sc_result=res_user,
    )

    # ── Advertencia de ambigüedad NLP ─────────────────────────────────────────
    if amb_warn:
        st.warning(f"**Uncertain interpretation:** {amb_warn}")

    # ── Tarjeta de veredicto ──────────────────────────────────────────────────
    # Normalizar campos según tipo de decisión (Decision vs DecisionChain)
    from dss.decisions import Decision as _Decision
    if isinstance(decision, DecisionChain):
        _one_time  = sum(d.one_time_cost for d in decision.decisions)
        _fixed     = sum(d.fixed_cost_change_abs for d in decision.decisions)
        _mkt_dec   = next((d for d in decision.decisions if d.marketing_spend_change_pct > 0), None)
        _mkt       = _mkt_dec is not None
        _fixed_dec = next((d for d in decision.decisions if d.fixed_cost_change_abs > 0), None)
        _month     = decision.decisions[0].month
        _month_mkt = _mkt_dec.month if _mkt_dec else _month
        _month_fix = _fixed_dec.month if _fixed_dec else _month
    else:
        _one_time  = decision.one_time_cost
        _fixed     = decision.fixed_cost_change_abs
        _mkt       = decision.marketing_spend_change_pct > 0
        _month     = decision.month
        _month_mkt = decision.month
        _month_fix = decision.month

    if _one_time > 0 and abs(delta_mrr) < 1.0 and abs(delta_cust) < 0.5:
        # Gasto puntual sin retorno en MRR ni clientes
        sub_veredicto = (
            f"One-time investment of <b>{_eur(_one_time)}</b> in month {_month} · "
            f"no MRR impact. "
            f"Consider whether it generates indirect value the model doesn't capture."
        )
    elif _one_time > 0:
        # Gasto puntual con retorno
        sub_veredicto = (
            f"One-time investment of <b>{_eur(_one_time)}</b> in month {_month}"
        )
    elif coste_extra < -1.0:
        # Ahorro neto en costes (ej. reducir marketing) — puede ir acompañado de pérdida de MRR
        if delta_mrr < -500:
            sub_veredicto = (
                f"saves <b>{_eur(abs(coste_extra))}/month</b> · "
                f"but MRR drops {_eur(abs(delta_mrr))}/month · negative impact on future revenue"
            )
        else:
            sub_veredicto = f"saves <b>{_eur(abs(coste_extra))}/month</b> with no significant MRR impact"
    elif _fixed > 0:
        sub_veredicto = (
            f"Additional recurring cost of <b>{_eur(_fixed)}/month</b> "
            f"from month {_month_fix}"
        )
    elif _mkt:
        sub_veredicto = f"Marketing budget increase from month {_month_mkt}"
    elif coste_extra < 1.0:
        sub_veredicto = "No additional investment required"
    elif delta_mrr > 0:
        meses_rec = coste_extra / delta_mrr
        sub_veredicto = (
            f"{_eur(coste_extra)}/month extra cost · "
            f"estimated payback in <b>{meses_rec:.1f} months</b>"
        )
    else:
        sub_veredicto = f"{_eur(coste_extra)}/month extra cost with no MRR increase"

    st.markdown(
        f"""
        <div style="
            background:{bg_v};
            border:2px solid {brd_v};
            border-radius:10px;
            padding:18px 22px;
            margin:16px 0 10px 0;
        ">
            <h3 style="color:{tc_v}; margin:0 0 4px 0;">{etiq_v}</h3>
            <p style="color:{tc_v}; margin:0; font-size:0.93em;">
                <b>{decision.label}</b> · {sub_veredicto}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Avisos de riesgo adicionales (Regla 2: capacidad + marketing) ─────────
    for aviso in avisos_v:
        st.warning(aviso)

    # ── KPI metrics con deltas ─────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)

    _d_cash_str, _d_cash_col, _d_cash_arr = _metric_delta(
        delta_cash, 1.0,
        lambda v: f"{_eur_delta(v)} ({_pct_delta(cash_pct)})",
    )
    k1.metric("Cash month 12", _eur(sc["cash_12"]),
              delta=_d_cash_str, delta_color=_d_cash_col, delta_arrow=_d_cash_arr)

    _d_cust_str, _d_cust_col, _d_cust_arr = _metric_delta(
        delta_cust, 0.5,
        lambda v: f"{v:+.1f}",
    )
    k2.metric("Customers month 12", f"{sc['customers_12']:.0f}",
              delta=_d_cust_str, delta_color=_d_cust_col, delta_arrow=_d_cust_arr)

    _d_mrr_str, _d_mrr_col, _d_mrr_arr = _metric_delta(delta_mrr, 1.0, _eur_delta)
    k3.metric("MRR month 12", _eur(sc["mrr_12"]),
              delta=_d_mrr_str, delta_color=_d_mrr_col, delta_arrow=_d_mrr_arr)

    _d_ebit_str, _d_ebit_col, _d_ebit_arr = _metric_delta(delta_ebitda, 1.0, _eur_delta)
    k4.metric("Cumulative EBITDA", _eur(sc["total_ebitda"]),
              delta=_d_ebit_str, delta_color=_d_ebit_col, delta_arrow=_d_ebit_arr)

    _nrr_sc    = sc.get("nrr", 100.0)
    _nrr_delta = _nrr_sc - bs.get("nrr", 100.0)
    _d_nrr_str, _d_nrr_col, _d_nrr_arr = _metric_delta(
        _nrr_delta, 0.1,
        lambda v: f"{v:+.1f} pp",
    )
    k5.metric("NRR", f"{_nrr_sc:.1f}%",
              delta=_d_nrr_str, delta_color=_d_nrr_col, delta_arrow=_d_nrr_arr,
              help=_nrr_label(_nrr_sc))

    with st.expander("📖 What is NRR and how to interpret it?"):
        st.markdown(
            f"**Net Revenue Retention (NRR)** measures the revenue generated by customers "
            f"existing at the start of the period, after applying churn. "
            f"An NRR > 100% means the customer base grows by itself even without "
            f"acquiring new customers (expansion).\n\n"
            f"| Range | Interpretation |\n"
            f"|---|---|\n"
            f"| > 100% | ✅ Expansion: you grow even without acquiring new customers |\n"
            f"| 90–100% | ⚠️ Mild contraction: churn erodes revenue |\n"
            f"| < 90% | ❌ Severe contraction: serious retention problem |\n\n"
            f"**SaaS B2B Benchmarks:**  \n"
            f"- Enterprise SaaS (Snowflake, Datadog): NRR 120–150%  \n"
            f"- Mid-market SaaS: NRR 100–120%  \n"
            f"- Typical SMB SaaS: NRR 70–90% (no expansion revenue)  \n\n"
            f"*Source: Bessemer Venture Partners State of the Cloud, 2023.*"
        )
        if _nrr_sc < 100:
            st.info(
                "💡 **NRR < 100%** indicates this model does not include expansion revenue "
                "(upsell/cross-sell). The 3% monthly churn erodes the initial cohort's MRR "
                "without expansion offset."
            )

    # ── Rule of 40 ────────────────────────────────────────────────────────────
    _r40_bs = compute_rule_of_40(res_baseline)
    _r40_sc = compute_rule_of_40(res_user)
    _r40_icon  = "✅" if _r40_sc["passes"] else "❌"
    _r40_delta = _r40_sc["rule_of_40"] - _r40_bs["rule_of_40"]
    _r40_mrr_color  = "#28a745" if _r40_sc["mrr_growth_pct"]  >= 0 else "#dc3545"
    _r40_ebit_color = "#28a745" if _r40_sc["ebitda_margin_pct"] >= 0 else "#dc3545"
    st.markdown(
        f"**Rule of 40:** `{_r40_sc['rule_of_40']:.1f}` {_r40_icon} "
        f"({'above' if _r40_sc['passes'] else 'below'} threshold ≥40)&nbsp;&nbsp;"
        f"— MRR Growth "
        f"<span style='color:{_r40_mrr_color};font-weight:600'>{_r40_sc['mrr_growth_pct']:+.1f}%</span>"
        f" + EBITDA Margin "
        f"<span style='color:{_r40_ebit_color};font-weight:600'>{_r40_sc['ebitda_margin_pct']:+.1f}%</span>"
        f"&nbsp;&nbsp;*(Δ vs Baseline: {_r40_delta:+.1f})*",
        unsafe_allow_html=True,
    )
    with st.expander("📖 What is the Rule of 40?"):
        st.markdown(
            "The **Rule of 40** is a benchmark for SaaS companies that balances growth "
            "and profitability: a company is healthy if the sum of its MRR growth rate "
            "and its EBITDA margin exceeds 40%.\n\n"
            "| Value | Interpretation |\n"
            "|---|---|\n"
            "| ≥ 40 | ✅ Healthy according to VC benchmark |\n"
            "| < 40 | ❌ Below the threshold |\n\n"
            "*'The best SaaS B2B companies maintain Rule of 40 > 40 — Bessemer Venture Partners, "
            "McKinsey (2023).'*"
        )

    # ── Advertencias contextuales (para Decision simple o cada sub-decisión de Chain) ──
    import math as _math
    _all_decs = decision.decisions if isinstance(decision, DecisionChain) else [decision]
    for _d in _all_decs:
        if _d.arpu_change_pct / 100.0 > BASELINE_CFG.price_churn_threshold:
            _arpu_inc    = _d.arpu_change_pct / 100.0
            _penalty_pp  = BASELINE_CFG.price_churn_sensitivity * (
                _math.exp(_arpu_inc * 2) - 1
            ) * 0.05 * 100
            _churn_base  = BASELINE_CFG.monthly_churn_rate * 100
            st.info(
                f"💡 **Price-churn (month {_d.month}):** a **{_d.arpu_change_pct:.0f}% ARPU** increase "
                f"raises estimated churn by **+{_penalty_pp:.2f} pp** "
                f"(from {_churn_base:.1f}% to {_churn_base + _penalty_pp:.2f}%)."
            )
        if _d.marketing_spend_change_pct > 0 and _d.capacity_change_abs == 0:
            st.warning(
                f"⚠️ (Month {_d.month}) You increased the marketing budget but **not the maximum "
                "acquisition capacity**. Consider adding capacity slots."
            )
        if _d.churn_change_abs < 0:
            st.info(
                "💡 **Assumption:** churn reduction already includes its associated cost modelled "
                "in fixed costs. If not, add it as `fixed_cost_change_abs`."
            )

    # ── Gráficos interactivos ──────────────────────────────────────────────────
    st.markdown("#### 📈 Time evolution vs Baseline")
    tab_cash, tab_cust, tab_mrr = st.tabs(["💰 Cash", "👥 Customers", "📈 MRR"])

    with tab_cash:
        st.plotly_chart(
            _fig_escenario(res_baseline, res_user, "cash", "Cash (€) over time", "€"),
            use_container_width=True,
        )
    with tab_cust:
        st.plotly_chart(
            _fig_escenario(res_baseline, res_user, "customers", "Customers over time", "Customers"),
            use_container_width=True,
        )
    with tab_mrr:
        st.plotly_chart(
            _fig_escenario(res_baseline, res_user, "mrr", "MRR (€) over time", "€"),
            use_container_width=True,
        )

    # ── Exportar CSV ───────────────────────────────────────────────────────────
    _etiq_csv  = decision.label[:40].replace(" ", "_").replace("/", "-")
    _csv_bytes = res_user.monthly_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label             = "⬇️ Download monthly results (CSV)",
        data              = _csv_bytes,
        file_name         = f"simulation_{_etiq_csv}.csv",
        mime              = "text/csv",
        use_container_width = True,
    )

    # ── Decisión interpretada (parámetros técnicos) ────────────────────────────
    col_det, col_analisis = st.columns([1, 1])

    with col_det:
        with st.expander("🔧 AI-interpreted parameters"):
            if isinstance(decision, DecisionChain):
                _es_combinada = decision.label.startswith("Combined:")
                _tipo_chain   = "Combined decisions" if _es_combinada else "Causal chain"
                _desc_chain   = "independent actions" if _es_combinada else "chained effects"
                st.markdown(f"**{_tipo_chain}** — {len(decision.decisions)} {_desc_chain}")
                for _i, _d in enumerate(decision.decisions, 1):
                    st.markdown(f"**Effect {_i}: {_d.label}**")
                    _rows = [
                        f"| Month | {_d.month} |",
                        f"| ARPU change | {_d.arpu_change_pct:+.1f}% |",
                        f"| Marketing change | {_d.marketing_spend_change_pct:+.1f}% |",
                        f"| Capacity change | {_d.capacity_change_abs:+d} slots |",
                        f"| Recurring fixed costs | {_eur_delta(_d.fixed_cost_change_abs)}/month |",
                        f"| Churn change | {_d.churn_change_abs:+.3f} |",
                        f"| Gross margin change | {_d.gross_margin_change_abs:+.3f} |",
                    ]
                    if _d.one_time_cost:
                        _rows.append(f"| One-time cost | {_eur(_d.one_time_cost)} (month {_d.month} only) |")
                    st.markdown("| Parameter | Value |\n|---|---|\n" + "\n".join(_rows))
            else:
                _rows_params = [
                    f"| Application month | {decision.month} |",
                    f"| ARPU change | {decision.arpu_change_pct:+.1f}% |",
                    f"| Marketing change | {decision.marketing_spend_change_pct:+.1f}% |",
                    f"| Capacity change | {decision.capacity_change_abs:+d} slots |",
                    f"| Recurring fixed costs | {_eur_delta(decision.fixed_cost_change_abs)}/month |",
                    f"| Churn change | {decision.churn_change_abs:+.3f} |",
                    f"| Gross margin change | {decision.gross_margin_change_abs:+.3f} |",
                ]
                if decision.one_time_cost:
                    _rows_params.append(
                        f"| One-time cost month {decision.month} | {_eur(decision.one_time_cost)} (that month only) |"
                    )
                st.markdown(
                    "| Parameter | Value |\n|---|---|\n" + "\n".join(_rows_params)
                )

    with col_analisis:
        with st.expander("🔍 Detailed analysis"):
            bullets = explain_scenario(res_baseline, res_user, sensitivity_rank, decision=decision)
            for bullet in bullets:
                st.markdown(bullet)

    # ── Advertencia OAT-decisión: conecta el driver crítico con el impacto real ──
    _top_driver = sensitivity_rank.iloc[0]["parameter"] if not sensitivity_rank.empty else ""
    _top_impact = sensitivity_rank.iloc[0]["mean_abs_Δcash_12"] if not sensitivity_rank.empty else 0
    _all_decs_oat = decision.decisions if isinstance(decision, DecisionChain) else [decision]

    # Obtener el delta de churn real simulado (pp)
    _churn_pp_real = (
        res_user.monthly_df["churn_rate"].mean()
        - res_baseline.monthly_df["churn_rate"].mean()
    ) * 100

    # Driver crítico = churn Y la decisión incrementa el churn
    if _top_driver == "monthly_churn_rate" and _churn_pp_real > 0.02:
        st.warning(
            f"⚠️ **Warning — critical driver at risk:** sensitivity analysis "
            f"identifies **churn** as the highest-impact factor for this company "
            f"(±{_eur(_top_impact)} on Cash₁₂ per ±1 pp). "
            f"This decision increases churn by **+{_churn_pp_real:.2f} pp** — "
            f"the real impact could be higher if churn exceeds estimates."
        )

    # Driver crítico = marketing Y la decisión reduce el marketing
    elif _top_driver == "marketing_spend_monthly":
        _mkt_reducido = any(
            getattr(_d, "marketing_spend_change_pct", 0) < 0
            for _d in _all_decs_oat
        )
        if _mkt_reducido:
            _mkt_pct = min(
                getattr(_d, "marketing_spend_change_pct", 0)
                for _d in _all_decs_oat
            )
            st.warning(
                f"⚠️ **Warning — critical driver at risk:** sensitivity analysis "
                f"identifies **marketing** as the highest-impact factor for this company "
                f"(±{_eur(_top_impact)} on Cash₁₂ per ±10%). "
                f"This decision reduces the marketing budget by **{_mkt_pct:.0f}%** — "
                f"the impact on acquisition could be higher than estimated."
            )

    # ── Advanced analysis: Monte Carlo + OAT Sensitivity ──────────────────────
    col_mc, col_oat = st.columns(2)

    with col_mc:
        with st.expander("🎲 Monte Carlo simulation (N=500)", expanded=False):
            _mc_scenario = Scenario(
                name="mc",
                config=copy.deepcopy(BASELINE_CFG),
                decisions=decision.decisions if isinstance(decision, DecisionChain) else [decision],
            )
            mc_result = run_monte_carlo(_mc_scenario, n_simulations=500, seed=42)

            _p5  = mc_result["percentiles"][5]
            _p50 = mc_result["percentiles"][50]
            _p95 = mc_result["percentiles"][95]
            _prob_insol = mc_result["prob_insolvent"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("P5 (worst case)", _eur(_p5))
            m2.metric("P50 (median)", _eur(_p50))
            m3.metric("P95 (best case)", _eur(_p95))
            m4.metric("P(insolvency)", f"{_prob_insol*100:.1f}%")

            _mc_df = mc_result["results_df"]
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(
                x=_mc_df["cash_12"],
                nbinsx=40,
                marker_color="#5B9BD5",
                opacity=0.8,
                name="Cash₁₂ distribution",
            ))
            for _pval, _pname, _dash in [
                (_p5,  "P5",          "dash"),
                (_p50, "P50 (median)", "solid"),
                (_p95, "P95",         "dash"),
            ]:
                fig_mc.add_vline(
                    x=_pval, line_dash=_dash, line_color="#FF5722", line_width=1.5,
                    annotation_text=f"{_pname}: {_eur(_pval)}",
                    annotation_position="top",
                )
            fig_mc.add_vline(x=0, line_dash="dot", line_color="red", line_width=1.5)
            fig_mc.update_layout(
                title=f"Monte Carlo — Cash₁₂ Distribution (N={mc_result['n_simulations']})",
                xaxis_title="Cash at month 12 (€)",
                yaxis_title="Frequency",
                height=350,
                margin=dict(t=50, b=30, l=10, r=10),
                showlegend=False,
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            st.caption(
                "Uncertainty modelled via **Beta(3, 97)** for churn and "
                "**LogNormal** for marketing efficiency (α). "
                "500 simulations with random parameter draws."
            )

    with col_oat:
        with st.expander("📊 OAT Sensitivity ranking", expanded=False):
            fig_oat = go.Figure()
            fig_oat.add_trace(go.Bar(
                y=sensitivity_rank["parameter"],
                x=sensitivity_rank["mean_abs_Δcash_12"],
                orientation="h",
                marker_color="#4CAF50",
                text=[f"€{v:,.0f}" for v in sensitivity_rank["mean_abs_Δcash_12"]],
                textposition="auto",
            ))
            fig_oat.update_layout(
                title="OAT Sensitivity Ranking — Impact on Cash₁₂",
                xaxis_title="Mean |ΔCash₁₂| (€)",
                yaxis=dict(autorange="reversed"),
                height=350,
                margin=dict(t=50, b=30, l=10, r=10),
                xaxis=dict(tickprefix="€", tickformat=",.0f"),
            )
            st.plotly_chart(fig_oat, use_container_width=True)

            st.markdown("**Local elasticity (approximate):**")
            _oat_display = sensitivity_rank[["parameter", "mean_abs_Δcash_12", "local_elasticity"]].copy()
            _oat_display.columns = ["Parameter", "Mean |ΔCash₁₂| (€)", "Local elasticity"]
            st.dataframe(
                _oat_display.style.format({
                    "Mean |ΔCash₁₂| (€)": "€{:,.0f}",
                    "Local elasticity": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "One-at-a-time (OAT) analysis: each parameter is perturbed ±ε "
                "while holding others constant. Local elasticity approximates "
                "the relative sensitivity of Cash₁₂ to each driver."
            )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — modo comparación (todas las decisiones de la sesión)
# ─────────────────────────────────────────────────────────────────────────────

elif st.session_state.modo_comparacion and st.session_state.historial:

    st.markdown("### 📊 Comparison of all simulated decisions")
    st.caption(
        f"Comparing **{len(st.session_state.historial)} decision(s)** from this session "
        f"against the Baseline."
    )

    # Gráfico de barras agrupadas
    st.plotly_chart(
        _fig_barras(st.session_state.historial, res_baseline),
        use_container_width=True,
    )

    # Líneas de caja superpuestas (todos los escenarios)
    todos = [res_baseline] + [it["resultado"] for it in st.session_state.historial]
    tab_c, tab_m = st.tabs(["💰 Cash — all scenarios", "📈 MRR — all scenarios"])
    with tab_c:
        st.plotly_chart(
            _fig_lineas(todos, "cash", "Cash (€) — all scenarios", "€"),
            use_container_width=True,
        )
    with tab_m:
        st.plotly_chart(
            _fig_lineas(todos, "mrr", "MRR (€) — all scenarios", "€"),
            use_container_width=True,
        )

    # Tabla resumen comparativa
    st.markdown("#### Summary table")

    def _fmt_eur(x):       return f"€{x:,.0f}"
    def _fmt_delta(x):     return ("+" if x >= 0 else "−") + f"€{abs(x):,.0f}"
    def _fmt_pct(x):       return f"{x:+.1f}%"

    filas = []
    for it in st.session_state.historial:
        sc_s = it["resultado"].summary
        bs_s = res_baseline.summary
        _, etiq, _, _, _, _ = _veredicto(
            bs_s, sc_s,
            bs_result=res_baseline, sc_result=it["resultado"],
        )
        d_cash  = sc_s["cash_12"]      - bs_s["cash_12"]
        d_mrr   = sc_s["mrr_12"]       - bs_s["mrr_12"]
        d_ebit  = sc_s["total_ebitda"] - bs_s["total_ebitda"]
        c_pct   = (d_cash / abs(bs_s["cash_12"])) * 100 if bs_s["cash_12"] != 0 else 0
        _r40    = compute_rule_of_40(it["resultado"])
        filas.append({
            "Decision":          it["decision"].label,
            "Verdict":           etiq,
            "Cash month 12":     sc_s["cash_12"],
            "Δ Cash":            d_cash,
            "Δ Cash %":          c_pct,
            "MRR month 12":      sc_s["mrr_12"],
            "Δ MRR":             d_mrr,
            "EBITDA accum.":     sc_s["total_ebitda"],
            "Δ EBITDA":          d_ebit,
            "NRR":               f"{sc_s.get('nrr', 100.0):.1f}%",
            "Rule of 40":        f"{_r40['rule_of_40']:.1f} {'✅' if _r40['passes'] else '❌'}",
            "Solvent":           "✅" if not sc_s["insolvent"] else "❌",
        })

    df_comp = pd.DataFrame(filas)
    st.dataframe(
        df_comp.style.format({
            "Cash month 12":  _fmt_eur,
            "Δ Cash":         _fmt_delta,
            "Δ Cash %":       _fmt_pct,
            "MRR month 12":   _fmt_eur,
            "Δ MRR":          _fmt_delta,
            "EBITDA accum.":  _fmt_eur,
            "Δ EBITDA":       _fmt_delta,
        }),
        use_container_width=True,
        hide_index=True,
        height=min(80 + 40 * len(filas), 420),
    )

    # ── Exportar historial CSV ─────────────────────────────────────────────────
    st.download_button(
        label             = "⬇️ Download full history (CSV)",
        data              = df_comp.to_csv(index=False).encode("utf-8"),
        file_name         = "scenario_history.csv",
        mime              = "text/csv",
        use_container_width = True,
    )

    # Ganadora global (por Caja mes 12)
    mejor_idx  = max(range(len(st.session_state.historial)),
                     key=lambda i: st.session_state.historial[i]["resultado"].summary["cash_12"])
    mejor_item = st.session_state.historial[mejor_idx]
    mejor_cash = mejor_item["resultado"].summary["cash_12"]
    base_cash  = res_baseline.summary["cash_12"]
    _, etiq_m, bg_m, brd_m, tc_m, _ = _veredicto(
        res_baseline.summary, mejor_item["resultado"].summary,
        bs_result=res_baseline, sc_result=mejor_item["resultado"],
    )

    st.markdown(
        f"""
        <div style="
            background:{bg_m}; border:2px solid {brd_m};
            border-radius:10px; padding:14px 18px; margin-top:12px;
        ">
            <h4 style="color:{tc_m}; margin:0 0 4px 0;">
                🏆 Best decision of the session ({etiq_m})
            </h4>
            <p style="color:{tc_m}; margin:0; font-size:0.92em;">
                <b>{mejor_item['decision'].label}</b><br>
                Cash month 12: <b>{_eur(mejor_cash)}</b>
                ({_eur_delta(mejor_cash - base_cash)} vs Baseline)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — estado vacío (sin decisiones todavía)
# ─────────────────────────────────────────────────────────────────────────────

elif not st.session_state.historial:
    st.info(
        "👆 Type a management decision in the field above and click **▶ Simulate decision** "
        "to see its 12-month financial impact.\n\n"
        "The system will interpret your text in natural language, run the simulation, and show "
        "you the impact on cash, customers, MRR and EBITDA with an automatic verdict."
    )
