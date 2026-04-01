"""
Rule-based explanation layer (no LLM).
Generates causal bullets and an executive summary for managers.
"""

import unicodedata
from typing import Dict, List
from dss.config import SimulationResult


# ─────────────────────────────────────────────────────────────────────────────
# DRIVER DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def _descomponer_cash(
    baseline: SimulationResult,
    scenario: SimulationResult,
) -> Dict[str, float]:
    """
    Cumulative contribution of each driver to the cash delta (12 months).

    First-order substitution method:
      contrib_arpu    = Σ (arpu_sc − arpu_bs) · customers_sc · gm_sc
      contrib_volumen = Σ (customers_sc − customers_bs) · arpu_bs · gm_bs
      contrib_mkt     = −Σ (mkt_sc − mkt_bs)
      contrib_fijos   = −Σ (fixed_sc − fixed_bs)
      contrib_one_time= −Σ (one_time_sc − one_time_bs)

    The sum ≈ actual delta_cash (error = ARPU × volume cross-term, usually < 2%).
    """
    bs_df = baseline.monthly_df
    sc_df = scenario.monthly_df

    # Gross margin month by month
    rev_bs = bs_df["revenue"].replace(0, float("nan"))
    rev_sc = sc_df["revenue"].replace(0, float("nan"))
    gm_bs  = ((rev_bs - bs_df["cogs"]) / rev_bs).fillna(0)
    gm_sc  = ((rev_sc - sc_df["cogs"]) / rev_sc).fillna(0)

    delta_arpu = sc_df["arpu"]      - bs_df["arpu"]
    delta_cust = sc_df["customers"] - bs_df["customers"]

    return {
        "arpu":     float((delta_arpu * sc_df["customers"] * gm_sc).sum()),
        "volumen":  float((delta_cust * bs_df["arpu"]      * gm_bs).sum()),
        "mkt":      float(-(sc_df["marketing_spend"] - bs_df["marketing_spend"]).sum()),
        "fijos":    float(-(sc_df["fixed_costs"]     - bs_df["fixed_costs"]).sum()),
        "one_time": float(-(sc_df["one_time_cost"]   - bs_df["one_time_cost"]).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _display_len(s: str) -> int:
    """
    Computes the display width of a string in terminal.
    Emojis and wide characters (W/F/A) count as 2.
    Variation selectors (U+FE0E, U+FE0F) count as 0.
    """
    total = 0
    for c in s:
        cp = ord(c)
        if cp in (0xFE0E, 0xFE0F):   # variation selectors: add no width
            continue
        if unicodedata.east_asian_width(c) in ('W', 'F', 'A'):
            total += 2
        else:
            total += 1
    return total


def _pad(s: str, width: int) -> str:
    """Pads a string with spaces on the right until the display width reaches exactly `width`."""
    diff = width - _display_len(s)
    return s + " " * max(0, diff)


def _wrap(text: str, width: int) -> List[str]:
    """
    Wraps text to lines of at most `width` display characters.
    If a word exceeds the width, it is truncated.
    """
    words   = text.split()
    lines   = []
    current = ""
    for word in words:
        w_len = _display_len(word)
        c_len = _display_len(current)
        sep   = 1 if current else 0
        if c_len + sep + w_len <= width:
            current = (current + " " + word).strip() if current else word
        else:
            if current:
                lines.append(current)
            # If a single word exceeds the width, truncate it (edge case)
            current = word[:width] if w_len > width else word
    if current:
        lines.append(current)
    return lines or [""]


# ─────────────────────────────────────────────────────────────────────────────
# CONTRASTIVE BULLETS — for main.py and detailed analysis
# ─────────────────────────────────────────────────────────────────────────────

def _de(frag: str) -> str:
    """Legacy helper for Spanish grammar contractions (de + el → del).
    Kept for compatibility — not used in English bullet generation."""
    if frag.startswith("el "):
        return "del " + frag[3:]
    return "de " + frag


def explain_scenario(
    baseline: SimulationResult,
    scenario: SimulationResult,
    sensitivity_rank=None,
    decision=None,      # Decision object — activates causal bullets based on the driver
) -> List[str]:
    """
    Generates causal bullets comparing `scenario` with `baseline`.

    When `decision` is passed, bullets 1–3 name the driver that
    drove the decision and explain the mechanism:
      "the ARPU increase (+20%), generating €X of monthly MRR,
       partially offset by the churn increase (+0.74 pp) causing
       the loss of 7 customers."

    Without `decision`, numeric decomposition is used as a fallback.
    """
    bs    = baseline.summary
    sc    = scenario.summary
    bs_df = baseline.monthly_df
    sc_df = scenario.monthly_df
    bullets = []

    delta_cash   = sc["cash_12"]      - bs["cash_12"]
    delta_cust   = sc["customers_12"] - bs["customers_12"]
    delta_mrr    = sc["mrr_12"]       - bs["mrr_12"]
    delta_ebitda = sc["total_ebitda"] - bs["total_ebitda"]

    # ── Valores derivados compartidos por varios bullets ──────────────────────
    delta_new     = float(sc_df["new_customers"].sum() - bs_df["new_customers"].sum())
    delta_churned = float(sc_df["churned"].sum()       - bs_df["churned"].sum())
    churn_pp      = (sc_df["churn_rate"].mean() - bs_df["churn_rate"].mean()) * 100
    mkt_extra     = float((sc_df["marketing_spend"] - bs_df["marketing_spend"]).mean())

    arpu_bs_12 = bs_df["arpu"].iloc[-1]
    arpu_sc_12 = sc_df["arpu"].iloc[-1]
    arpu_pct_12 = (arpu_sc_12 / arpu_bs_12 - 1) * 100 if arpu_bs_12 > 0 else 0

    # ── 1. Cash final — causal ────────────────────────────────────────────────
    if abs(delta_cash) >= 1:
        contrib  = _descomponer_cash(baseline, scenario)
        resumen  = (
            f"€{abs(delta_cash):,.0f} "
            f"({'+'  if delta_cash >= 0 else '−'}"
            f"{abs(delta_cash / bs['cash_12'] * 100):.1f}% vs baseline)"
            if bs["cash_12"] != 0 else f"€{abs(delta_cash):,.0f}"
        )

        if decision is not None:
            # ── Causal narrative based on the Decision fields ────────────────
            arpu_dec  = getattr(decision, "arpu_change_pct",            0)
            mkt_dec   = getattr(decision, "marketing_spend_change_pct",  0)
            churn_dec = getattr(decision, "churn_change_abs",            0)
            fixed_dec = getattr(decision, "fixed_cost_change_abs",       0)
            ot_dec    = getattr(decision, "one_time_cost",               0)

            frags_pos: List[tuple] = []   # (magnitud, texto)
            frags_neg: List[tuple] = []

            # Driver ARPU
            if abs(arpu_dec) >= 0.1:
                c_arpu = contrib.get("arpu", 0)
                dir_a  = "increase" if arpu_dec > 0 else "decrease"
                frag   = (
                    f"the ARPU {dir_a} ({arpu_dec:+.0f}%), "
                    f"generating €{abs(delta_mrr):,.0f} extra monthly MRR"
                )
                # Matiz: churn penalty inducido por la subida de precio
                if churn_pp > 0.05 and arpu_dec > 0:
                    frag += (
                        f", partially offset by the churn increase "
                        f"(+{churn_pp:.2f} pp) causing the loss of "
                        f"{abs(int(round(delta_cust)))} customers"
                    )
                (frags_pos if c_arpu >= 0 else frags_neg).append((abs(c_arpu), frag))

            # Driver marketing
            if abs(mkt_dec) >= 0.1:
                c_vol = contrib.get("volumen", 0)
                c_mkt = contrib.get("mkt",     0)
                net   = c_vol + c_mkt
                dir_m = "increase" if mkt_dec > 0 else "decrease"
                accion = "acquired" if delta_new > 0 else "acquired fewer"
                frag  = (
                    f"the marketing budget {dir_m} ({mkt_dec:+.0f}%), "
                    f"which {accion} {abs(delta_new):.0f} additional customers"
                )
                if delta_mrr > 0 and delta_new > 0:
                    frag += f", generating €{abs(delta_mrr):,.0f}/month of extra MRR"
                # Note: 'despite the cost' is NOT embedded here to avoid double negation
                # when the fragment forms part of a contrastive sentence ('Despite X, Y...')
                (frags_pos if net >= 0 else frags_neg).append((abs(net), frag))

            # Direct churn driver (not price-induced)
            if abs(churn_dec) > 0 and abs(mkt_dec) < 0.1:
                c_vol  = contrib.get("volumen", 0)
                dir_ch = "reduction" if churn_dec < 0 else "increase"
                accion = "retained" if churn_dec < 0 else "lost"
                frag   = (
                    f"the churn {dir_ch} ({churn_dec * 100:+.2f} pp), "
                    f"which {accion} {abs(int(round(delta_cust)))} additional customers"
                )
                if abs(delta_mrr) >= 1 and churn_dec < 0:
                    frag += f", generating €{abs(delta_mrr):,.0f}/month of extra MRR"
                (frags_pos if c_vol >= 0 else frags_neg).append((abs(c_vol), frag))

            # Fixed cost driver
            if abs(fixed_dec) > 0:
                c_fij  = contrib.get("fijos", 0)
                if fixed_dec > 0:
                    frag = f"the fixed cost increase (+€{fixed_dec:,.0f}/month)"
                else:
                    frag = f"the fixed cost reduction (€{fixed_dec:,.0f}/month)"
                (frags_pos if c_fij >= 0 else frags_neg).append((abs(c_fij), frag))

            # One-time expense
            if ot_dec > 0:
                frags_neg.append((abs(contrib.get("one_time", 0)),
                                  f"the one-time investment of €{ot_dec:,.0f}"))

            frags_pos.sort(reverse=True)
            frags_neg.sort(reverse=True)

            if frags_pos and not frags_neg:
                dir_cash_en = "increased" if delta_cash >= 0 else "decreased"
                txt = f"• Cash {dir_cash_en} {resumen} mainly due to {frags_pos[0][1]}"
                if len(frags_pos) > 1:
                    txt += f", also supported by {frags_pos[1][1]}"
                bullets.append(txt + ".")

            elif frags_neg and not frags_pos:
                dir_cash_en = "increased" if delta_cash >= 0 else "decreased"
                txt = f"• Cash {dir_cash_en} {resumen} mainly due to {frags_neg[0][1]}"
                if len(frags_neg) > 1:
                    txt += f", worsened by {frags_neg[1][1]}"
                bullets.append(txt + ".")

            elif frags_pos and frags_neg:
                if delta_cash >= 0:
                    dir_cash_en = "increased" if delta_cash >= 0 else "decreased"
                    bullets.append(
                        f"• Cash {dir_cash_en} {resumen} mainly due to "
                        f"{frags_pos[0][1]}, despite {frags_neg[0][1]}."
                    )
                else:
                    bullets.append(
                        f"• Despite {frags_pos[0][1]}, "
                        f"{frags_neg[0][1]} resulted in a net negative balance of {resumen}."
                    )
            else:
                # No recognised drivers in Decision → numeric fallback
                dir_cash_en = "increased" if delta_cash >= 0 else "decreased"
                bullets.append(
                    f"• Cash by month 12 {dir_cash_en} by {resumen} "
                    f"(€{bs['cash_12']:,.0f} → €{sc['cash_12']:,.0f})."
                )

        else:
            # ── No Decision: numeric decomposition ──────────────────────────
            umbral    = max(abs(delta_cash) * 0.05, 200)
            rel       = {k: v for k, v in contrib.items() if abs(v) >= umbral}
            positivos = {k: v for k, v in rel.items() if v > 0}
            negativos = {k: v for k, v in rel.items() if v < 0}
            ordenados = sorted(rel.items(), key=lambda x: abs(x[1]), reverse=True)

            _labels = {
                "arpu":     lambda v: (f"{'the ARPU increase' if v > 0 else 'the ARPU decrease'} "
                                       f"({arpu_pct_12:+.0f}%)"),
                "volumen":  lambda v: (f"{'the higher number' if v > 0 else 'the loss'} of customers "
                                       f"({delta_cust:+.0f})"),
                "mkt":      lambda v: ("the marketing savings" if v > 0 else "the higher marketing spend"),
                "fijos":    lambda v: ("the fixed cost reduction" if v > 0 else "the fixed cost increase"),
                "one_time": lambda v: f"the one-time investment of €{abs(v):,.0f}",
            }
            nombre = lambda k, v: _labels[k](v)

            dir_cash_en = "increased" if delta_cash >= 0 else "decreased"
            if not ordenados:
                bullets.append(
                    f"• Cash by month 12 {dir_cash_en} by {resumen} "
                    f"(€{bs['cash_12']:,.0f} → €{sc['cash_12']:,.0f})."
                )
            elif not negativos or not positivos:
                pk, pv = ordenados[0]
                txt = f"• Cash {dir_cash_en} {resumen} mainly due to {nombre(pk, pv)}"
                if len(ordenados) > 1:
                    sk, sv = ordenados[1]
                    if abs(sv) / max(abs(delta_cash), 1) > 0.15:
                        txt += f", with support from {nombre(sk, sv)}"
                bullets.append(txt + ".")
            else:
                top_pos_k, top_pos_v = max(positivos.items(), key=lambda x: abs(x[1]))
                top_neg_k, top_neg_v = min(negativos.items(), key=lambda x: x[1])
                balance = "positive" if delta_cash >= 0 else "negative"
                bullets.append(
                    f"• Despite {nombre(top_neg_k, top_neg_v)} "
                    f"(−€{abs(top_neg_v):,.0f}), "
                    f"{nombre(top_pos_k, top_pos_v)} "
                    f"(+€{abs(top_pos_v):,.0f}) resulted in a net {balance} balance of {resumen}."
                )

    # ── 2. Customers — causal ─────────────────────────────────────────────────
    if abs(delta_cust) >= 0.5:
        c_dir = "gained" if delta_cust >= 0 else "lost"

        if decision is not None:
            arpu_dec  = getattr(decision, "arpu_change_pct",            0)
            mkt_dec   = getattr(decision, "marketing_spend_change_pct",  0)
            churn_dec = getattr(decision, "churn_change_abs",            0)

            partes = []

            # Marketing acquisition
            if abs(mkt_dec) >= 0.1 and abs(delta_new) >= 0.5:
                dir_m  = "higher" if mkt_dec > 0 else "lower"
                accion = "acquired" if delta_new > 0 else "acquired fewer"
                partes.append(
                    f"the {dir_m} marketing budget ({mkt_dec:+.0f}%) "
                    f"{accion} {abs(delta_new):.0f} customers"
                )
            elif abs(delta_new) >= 0.5:
                partes.append(
                    f"marketing {'acquired' if delta_new > 0 else 'acquired fewer'} "
                    f"{abs(delta_new):.0f} customers"
                )

            # Churn loss/gain (direct or price-induced)
            if abs(churn_dec) > 0 and abs(delta_churned) >= 0.5:
                dir_ch = "reduction" if churn_dec < 0 else "increase"
                accion = "retained" if churn_dec < 0 else "lost"
                partes.append(
                    f"the churn {dir_ch} ({churn_dec * 100:+.2f} pp) "
                    f"{accion} {abs(delta_churned):.0f} additional"
                )
            elif arpu_dec > 0 and churn_pp > 0.05 and abs(delta_churned) >= 0.5:
                partes.append(
                    f"the price-driven churn increase (+{churn_pp:.2f} pp) "
                    f"cost {abs(delta_churned):.0f} customers"
                )
            elif abs(delta_churned) >= 0.5:
                partes.append(
                    f"{'lost' if delta_churned > 0 else 'retained'} "
                    f"{abs(delta_churned):.0f} {'more' if delta_churned > 0 else 'fewer'} through churn"
                )

            if partes:
                sep = ", although " if len(partes) > 1 else ": "
                bullets.append(
                    f"• The customer base {c_dir} {abs(delta_cust):.0f} "
                    f"by month 12{sep if len(partes) == 1 else ': '}"
                    f"{(', although '.join(partes) if len(partes) > 1 else partes[0])}."
                )
            else:
                bullets.append(
                    f"• The customer base {c_dir} {abs(delta_cust):.1f} customers "
                    f"by month 12 ({bs['customers_12']:.0f} → {sc['customers_12']:.0f})."
                )
        else:
            partes = []
            if abs(delta_new) >= 0.5:
                partes.append(
                    f"{'acquired' if delta_new > 0 else 'acquired fewer'} "
                    f"{abs(delta_new):.0f} customers via marketing"
                )
            if abs(delta_churned) >= 0.5:
                partes.append(
                    f"{'lost' if delta_churned > 0 else 'retained'} "
                    f"{abs(delta_churned):.0f} {'more' if delta_churned > 0 else 'fewer'} through churn"
                )
            if partes:
                bullets.append(
                    f"• The customer base {c_dir} {abs(delta_cust):.0f} by month 12 "
                    f"({'; '.join(partes)})."
                )
            else:
                bullets.append(
                    f"• The customer base {c_dir} {abs(delta_cust):.1f} customers "
                    f"by month 12 ({bs['customers_12']:.0f} → {sc['customers_12']:.0f})."
                )

    # ── 3. MRR — causal (efecto precio vs. efecto volumen, lenguaje verbal) ───
    if abs(delta_mrr) >= 1:
        cust_bs_12 = bs_df["customers"].iloc[-1]
        cust_sc_12 = sc_df["customers"].iloc[-1]

        contrib_precio = (arpu_sc_12 - arpu_bs_12) * cust_sc_12
        contrib_vol    = (cust_sc_12 - cust_bs_12) * arpu_bs_12

        mrr_dir = "rose" if delta_mrr >= 0 else "fell"
        partes  = []
        if abs(contrib_precio) >= 1:
            dir_a  = "increase" if arpu_pct_12 > 0 else "decrease"
            signo  = "contributed" if contrib_precio > 0 else "deducted"
            partes.append(
                f"the ARPU {dir_a} ({arpu_pct_12:+.0f}%) "
                f"{signo} €{abs(contrib_precio):,.0f}"
            )
        if abs(contrib_vol) >= 1:
            dir_v  = "acquisition of additional customers" if delta_cust > 0 else "loss of customers"
            signo  = "contributed" if contrib_vol > 0 else "deducted"
            partes.append(
                f"the {dir_v} ({delta_cust:+.0f}) "
                f"{signo} €{abs(contrib_vol):,.0f}"
            )

        if partes:
            # Price↑ / customers↓ tension (typical case: ARPU rises but churn increases)
            if len(partes) == 2 and contrib_precio > 0 and contrib_vol < 0:
                bullets.append(
                    f"• MRR at month 12 {mrr_dir} €{abs(delta_mrr):,.0f} "
                    f"(€{bs['mrr_12']:,.0f} → €{sc['mrr_12']:,.0f}): "
                    f"{partes[0]}, partially offset by {partes[1]}."
                )
            elif len(partes) == 2 and contrib_precio < 0 and contrib_vol > 0:
                bullets.append(
                    f"• MRR at month 12 {mrr_dir} €{abs(delta_mrr):,.0f}: "
                    f"{partes[1]}, despite {partes[0]}."
                )
            else:
                bullets.append(
                    f"• MRR at month 12 {mrr_dir} €{abs(delta_mrr):,.0f} "
                    f"(€{bs['mrr_12']:,.0f} → €{sc['mrr_12']:,.0f}): "
                    f"{'; '.join(partes)}."
                )
        else:
            bullets.append(
                f"• MRR at month 12 {mrr_dir} by €{abs(delta_mrr):,.0f} "
                f"(€{bs['mrr_12']:,.0f} → €{sc['mrr_12']:,.0f})."
            )

    # ── 4. EBITDA acumulado — causal (ingresos vs. costes) ───────────────────
    if abs(delta_ebitda) >= 1:
        delta_rev  = float((sc_df["revenue"]     - bs_df["revenue"]).sum())
        delta_cost = float((sc_df["total_costs"]  - bs_df["total_costs"]).sum())
        e_dir = "improved" if delta_ebitda >= 0 else "worsened"

        if abs(delta_rev) >= 1 and abs(delta_cost) >= 1:
            if delta_rev >= 0 and delta_cost >= 0:
                if delta_rev > delta_cost:
                    bullets.append(
                        f"• Cumulative EBITDA {e_dir} €{abs(delta_ebitda):,.0f}: "
                        f"revenue +€{abs(delta_rev):,.0f} exceeded "
                        f"the cost increase +€{abs(delta_cost):,.0f}."
                    )
                else:
                    bullets.append(
                        f"• Cumulative EBITDA {e_dir} €{abs(delta_ebitda):,.0f}: "
                        f"the cost increase +€{abs(delta_cost):,.0f} "
                        f"exceeded revenue growth +€{abs(delta_rev):,.0f}."
                    )
            elif delta_rev >= 0 and delta_cost < 0:
                bullets.append(
                    f"• Cumulative EBITDA {e_dir} €{abs(delta_ebitda):,.0f}: "
                    f"revenue +€{abs(delta_rev):,.0f} and cost savings "
                    f"€{abs(delta_cost):,.0f} both moved in a positive direction."
                )
            elif delta_rev < 0 and delta_cost >= 0:
                bullets.append(
                    f"• Cumulative EBITDA {e_dir} €{abs(delta_ebitda):,.0f}: "
                    f"revenue decline −€{abs(delta_rev):,.0f} and cost increase "
                    f"+€{abs(delta_cost):,.0f} both moved in a negative direction."
                )
            else:
                # delta_rev < 0 and delta_cost < 0 (cost savings with revenue decline)
                if abs(delta_cost) > abs(delta_rev):
                    bullets.append(
                        f"• Cumulative EBITDA {e_dir} €{abs(delta_ebitda):,.0f}: "
                        f"the cost savings €{abs(delta_cost):,.0f} offset "
                        f"the revenue decline −€{abs(delta_rev):,.0f}."
                    )
                else:
                    bullets.append(
                        f"• Cumulative EBITDA {e_dir} €{abs(delta_ebitda):,.0f}: "
                        f"revenue decline −€{abs(delta_rev):,.0f} partially "
                        f"offset by cost savings €{abs(delta_cost):,.0f}."
                    )
        elif abs(delta_rev) >= 1:
            rev_dir = "rise" if delta_rev >= 0 else "decline"
            bullets.append(
                f"• Cumulative EBITDA {e_dir} €{abs(delta_ebitda):,.0f} "
                f"due to the revenue {rev_dir} "
                f"({'+'  if delta_rev >= 0 else '−'}€{abs(delta_rev):,.0f})."
            )
        elif abs(delta_cost) >= 1:
            cost_dir = "reduction" if delta_cost < 0 else "increase"
            bullets.append(
                f"• Cumulative EBITDA {e_dir} €{abs(delta_ebitda):,.0f} "
                f"due to the cost {cost_dir} "
                f"({'−' if delta_cost < 0 else '+'}€{abs(delta_cost):,.0f})."
            )
        else:
            bullets.append(
                f"• Cumulative EBITDA over 12 months {e_dir} by €{abs(delta_ebitda):,.0f} "
                f"(€{bs['total_ebitda']:,.0f} → €{sc['total_ebitda']:,.0f})."
            )

    # ── No impact on financial KPIs ──────────────────────────────────────────
    _kpi_bullets_added = (
        abs(delta_cash) >= 1
        or abs(delta_cust) >= 0.5
        or abs(delta_mrr) >= 1
        or abs(delta_ebitda) >= 1
    )
    if not _kpi_bullets_added:
        bullets.append(
            "• This decision has no direct impact on the financial KPIs "
            "within the simulated horizon."
        )

    # ── 5. Incremental cost-benefit ───────────────────────────────────────────
    # Extra spend on controllable drivers (marketing + fixed costs).
    # COGS is excluded because it is a consequence of revenue, not a direct decision.
    coste_extra_mensual = (
        (sc_df["marketing_spend"] + sc_df["fixed_costs"])
        - (bs_df["marketing_spend"] + bs_df["fixed_costs"])
    ).mean()

    if coste_extra_mensual > 1.0:
        if delta_mrr > 0:
            meses_recuperacion = coste_extra_mensual / delta_mrr
            bullets.append(
                f"• 💡 Cost-benefit: €{coste_extra_mensual:,.0f}/month of extra spend generates "
                f"€{delta_mrr:,.0f}/month of additional MRR. "
                f"Estimated payback in {meses_recuperacion:.1f} months."
            )
        else:
            bullets.append(
                f"• 💡 Cost-benefit: €{coste_extra_mensual:,.0f}/month of extra spend "
                f"with no MRR increase — review whether the additional spend is justified."
            )
    elif coste_extra_mensual < -1.0:
        ahorros = abs(coste_extra_mensual)
        dir_mrr = "generates" if delta_mrr > 0 else "reduces"
        bullets.append(
            f"• 💡 Cost-benefit: saves €{ahorros:,.0f}/month in operating costs "
            f"and {dir_mrr} €{abs(delta_mrr):,.0f} of MRR vs baseline."
        )
    else:
        # No extra cost (e.g. pure price increase)
        if delta_mrr > 0:
            bullets.append(
                f"• 💡 Cost-benefit: this decision has no additional direct cost. "
                f"The extra MRR of €{delta_mrr:,.0f}/month is generated without new investment."
            )

    # ── 6. Runway / solvencia ─────────────────────────────────────────────────
    if sc["insolvent"] and not bs["insolvent"]:
        bullets.append(
            f"• ⚠️  This scenario leads to insolvency (cash < 0) in month "
            f"{sc['cash_runway_months']} — the baseline remains solvent."
        )
    elif bs["insolvent"] and not sc["insolvent"]:
        bullets.append(
            f"• ✅ This scenario avoids insolvency "
            f"(the baseline went negative in month {bs['cash_runway_months']})."
        )
    elif bs["insolvent"] and sc["insolvent"]:
        delta_runway = sc["cash_runway_months"] - bs["cash_runway_months"]
        r_dir = "delays" if delta_runway >= 0 else "brings forward"
        bullets.append(
            f"• ⚠️  Both scenarios are insolvent. This scenario {r_dir} "
            f"insolvency by {abs(delta_runway)} months "
            f"(month {bs['cash_runway_months']} → month {sc['cash_runway_months']})."
        )
    else:
        delta_cash_pct = (delta_cash / abs(bs["cash_12"])) * 100 if bs["cash_12"] != 0 else 0
        if abs(delta_cash_pct) < 0.1:
            bullets.append(
                "• ✅ Both scenarios maintain positive cash throughout the horizon "
                "with a final cash identical to baseline."
            )
        else:
            signo = "+" if delta_cash_pct >= 0 else ""
            bullets.append(
                f"• ✅ Both scenarios maintain positive cash throughout the horizon. "
                f"This scenario closes month 12 with {signo}{abs(delta_cash_pct):.1f}% "
                f"{'more' if delta_cash_pct >= 0 else 'less'} cash than baseline."
            )

    # ── 7. Churn penalised by price increase ─────────────────────────────────
    arpu_ratio = (sc_df["arpu"].iloc[-1] / bs_df["arpu"].iloc[-1]
                  if bs_df["arpu"].iloc[-1] > 0 else 1.0)
    if arpu_ratio > 1.05:   # ARPU increase above 5%
        arpu_increase_pct = (arpu_ratio - 1) * 100
        churn_delta_pp    = (sc_df["churn_rate"].mean() - bs_df["churn_rate"].mean()) * 100
        if churn_delta_pp > 0:
            bullets.append(
                f"• ⚠️ Assumption: a {arpu_increase_pct:.0f}% ARPU increase raises "
                f"estimated churn by +{churn_delta_pp:.2f} pp (sensitivity=0.3). "
                f"Validate with real data before executing."
            )

    # ── 8. Most sensitive driver ─────────────────────────────────────────────
    if sensitivity_rank is not None and len(sensitivity_rank) > 0:
        top = sensitivity_rank.iloc[0]["parameter"]
        imp = sensitivity_rank.iloc[0]["mean_abs_Δcash_12"]
        bullets.append(
            f"• Sensitivity analysis identifies '{top}' as the most critical driver "
            f"(mean |ΔCash₁₂| = €{imp:,.0f}); "
            f"small variations here have the greatest financial impact."
        )

    return bullets


def print_explanation(bullets: List[str], scenario_name: str, baseline_name: str):
    """Prints the detailed analysis block with formatting."""
    print(f"\n{'═'*65}")
    print(f"  DETAILED ANALYSIS: '{scenario_name}' vs '{baseline_name}'")
    print(f"{'═'*65}")
    for b in bullets:
        print(f"  {b}")
    print(f"{'═'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTIVE SUMMARY — for managers
# ─────────────────────────────────────────────────────────────────────────────

def build_executive_summary(
    baseline: SimulationResult,
    scenario: SimulationResult,
    decision=None,      # optional Decision object, for contextual warnings
) -> List[str]:
    """
    Generates the executive summary for managers.
    Returns a list of lines ready to print with print_executive_summary().

    Automatic verdict:
      ✅ RECOMMENDED         → delta_cash >= 0 and not insolvent
      ⚠️  PROCEED WITH CAUTION → delta_cash < 0 but MRR rises and not insolvent
      ❌ NOT RECOMMENDED     → insolvent OR delta_cash < -20% of baseline
    """
    bs = baseline.summary
    sc = scenario.summary
    W  = 62   # inner box width (number of ═ chars and display chars per row)

    delta_cash   = sc["cash_12"]      - bs["cash_12"]
    delta_cust   = sc["customers_12"] - bs["customers_12"]
    delta_mrr    = sc["mrr_12"]       - bs["mrr_12"]
    delta_ebitda = sc["total_ebitda"] - bs["total_ebitda"]
    cash_pct     = (delta_cash / abs(bs["cash_12"])) * 100 if bs["cash_12"] != 0 else 0

    bs_df = baseline.monthly_df
    sc_df = scenario.monthly_df
    coste_extra_mensual = (
        (sc_df["marketing_spend"] + sc_df["fixed_costs"])
        - (bs_df["marketing_spend"] + bs_df["fixed_costs"])
    ).mean()

    # ── Verdict ────────────────────────────────────────────────────────────────
    if sc["insolvent"]:
        veredicto   = "❌ NOT RECOMMENDED"
        texto_vered = (
            f"This scenario leads the company to insolvency "
            f"in month {sc['cash_runway_months']}."
        )
    elif cash_pct < -20.0:
        veredicto   = "❌ NOT RECOMMENDED"
        texto_vered = (
            f"Cash drops {abs(cash_pct):.1f}% compared to baseline. "
            f"Review assumptions before executing."
        )
    elif delta_cash < 0 and delta_mrr > 0:
        veredicto   = "⚠️  PROCEED WITH CAUTION"
        texto_vered = (
            "Reduces cash in the short term but increases MRR. "
            "Evaluate whether the growth justifies the cost."
        )
    else:
        veredicto = "✅ RECOMMENDED"
        if coste_extra_mensual < 1.0:
            texto_vered = "This decision improves profitability with no additional cost."
        else:
            texto_vered = (
                f"This decision improves profitability with an extra cost "
                f"of €{coste_extra_mensual:,.0f}/month."
            )

    # ── Contextual warnings (based on the decision, not the result) ──────────
    advertencias = []
    if decision is not None:
        if decision.arpu_change_pct > 0:
            advertencias.append(
                "Price increase may raise churn. "
                "Model assumption: constant churn. Validate with historical data."
            )
        if decision.marketing_spend_change_pct > 0 and decision.capacity_change_abs == 0:
            advertencias.append(
                "Without additional acquisition capacity, extra marketing budget "
                "may not convert to more customers."
            )
        if decision.churn_change_abs < 0:
            advertencias.append(
                "Reducing churn requires investment in product or support "
                "— ensure associated costs are modelled."
            )

    # ── Solvency ───────────────────────────────────────────────────────────────
    if sc["insolvent"]:
        solvencia_txt = f"⚠️  Insolvency in month {sc['cash_runway_months']}"
    else:
        solvencia_txt = "✅ Positive cash throughout the year"

    # ── Box formatting helpers ─────────────────────────────────────────────────
    def fila(contenido: str) -> str:
        """Inner box line padded to display width W."""
        inner = "  " + contenido
        return "║" + _pad(inner, W) + "║"

    def sep() -> str:
        return "╠" + "═" * W + "╣"

    # ── Signs and arrows ──────────────────────────────────────────────────────
    signo_cash  = "+" if delta_cash   >= 0 else "-"
    signo_cust  = "+" if delta_cust   >= 0 else ""   # handled via text ("no change" / abs)
    signo_mrr   = "+" if delta_mrr    >= 0 else "-"
    signo_ebit  = "+" if delta_ebitda >= 0 else "-"
    flecha_cash = "↑" if delta_cash   >= 0 else "↓"

    # ── Build the box ─────────────────────────────────────────────────────────
    titulo = f"EXECUTIVE SUMMARY — {scenario.scenario_name}"

    lines = []
    lines.append("╔" + "═" * W + "╗")
    lines.append(fila(titulo))
    lines.append(sep())

    # Cash
    lines.append(fila(
        f"💰 Cash impact            "
        f"{signo_cash}€{abs(delta_cash):,.0f}  "
        f"({flecha_cash} {abs(cash_pct):.1f}%)"
    ))

    # Customers
    if abs(delta_cust) < 0.5:
        cust_detalle = "no change"
    else:
        cust_detalle = f"{signo_cust}{delta_cust:.1f}"
    lines.append(fila(
        f"👥 Customers at month 12   "
        f"{sc['customers_12']:.0f}  ({cust_detalle})"
    ))

    # MRR
    lines.append(fila(
        f"📈 MRR at month 12        "
        f"€{sc['mrr_12']:,.0f}  ({signo_mrr}€{abs(delta_mrr):,.0f})"
    ))

    # Cumulative EBITDA
    lines.append(fila(
        f"📊 Cumulative EBITDA      "
        f"€{sc['total_ebitda']:,.0f}  ({signo_ebit}€{abs(delta_ebitda):,.0f})"
    ))

    # Solvency
    lines.append(fila(f"🏦 Solvency               {solvencia_txt}"))

    lines.append(sep())

    # Verdict
    lines.append(fila(f"VERDICT:  {veredicto}"))

    # Verdict text (wrapped to content width = W-2)
    for linea in _wrap(texto_vered, W - 2):
        lines.append(fila(linea))

    # Contextual warnings
    for adv in advertencias:
        for linea in _wrap(f"⚠️  {adv}", W - 2):
            lines.append(fila(linea))

    lines.append("╚" + "═" * W + "╝")
    return lines


def print_executive_summary(lines: List[str]):
    """Prints the executive summary generated by build_executive_summary()."""
    print()
    for line in lines:
        print(f"  {line}")
    print()
