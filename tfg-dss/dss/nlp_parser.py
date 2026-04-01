"""
NLP layer for translating natural language decisions into Decision objects.
The LLM acts ONLY as a JSON translator — it does not make business decisions.
"""

import json
import re
from groq import Groq
from dss.decisions import Decision, DecisionChain


SYSTEM_PROMPT = """
You are a business decision parser for a SaaS B2B simulation system.

You can return TWO types of JSON depending on what the user describes:

═══ TYPE A — Simple decision (type: "single") ═══
A single action, or several drivers applied at the SAME causal moment:
{
  "type": "single",
  "month": <int, month when the decision takes effect, 1-12>,
  "arpu_change_pct": <float>,
  "marketing_spend_change_pct": <float>,
  "capacity_change_abs": <int>,
  "fixed_cost_change_abs": <float>,
  "churn_change_abs": <float>,
  "gross_margin_change_abs": <float>,
  "one_time_cost": <float>,
  "sub_decisions": [<list of independent decision objects; see MULTI-ACTION rule. Empty [] if not applicable.>],
  "label": <string>,
  "ambiguity_warning": <string or "">
}

═══ TYPE B — Causal chain (type: "chain") ═══
When an action in one month CAUSES OR ENABLES another action in a LATER month.
Keywords: "which allows us to", "which enables", "which leads to", "as a result",
"from then on", "which increases", "which reduces", "so that from month X".
{
  "type": "chain",
  "chain_label": <string, overall description of the chain>,
  "decisions": [
    {
      "month": <int>,
      "arpu_change_pct": <float>,
      "marketing_spend_change_pct": <float>,
      "capacity_change_abs": <int>,
      "fixed_cost_change_abs": <float>,
      "churn_change_abs": <float>,
      "gross_margin_change_abs": <float>,
      "one_time_cost": <float>,
      "label": <string>
    },
    { ... second effect, in the later month ... }
  ]
}

STRICT RULES:
- Return ONLY the JSON, with no additional explanation or text.
- If the user does not mention a field, set it to 0 (or [] for sub_decisions).
- If the user does not mention a month, use month=1.
- If the user mentions "summer" use month=6, "spring" month=3, "autumn"/"fall" month=9.
- churn_change_abs must be negative if the user wants to reduce churn.

CRITICAL RULE — Choosing between "single", "chain", and sub_decisions:

1. CHAIN (type: "chain"):
   - Use chain when an action in one month CAUSES the effect in the next month.
   - Signals: "which allows us to", "which enables", "which leads to", "as a result",
     "from then on", "which increases/reduces".
   - Both effects must occur in DIFFERENT MONTHS.
   - Example: "we buy servers in May WHICH ALLOWS US TO increase sales from June"
     → chain: {month 5: one_time_cost=X} + {month 6: capacity/marketing++}

2. MULTI-ACTION (type: "single" with non-empty sub_decisions):
   - Use sub_decisions when there are SEVERAL INDEPENDENT actions with no causal link.
   - Signals: "and also", "and additionally", "we also want to", "but also".
   - Set the top-level JSON fields to 0 and describe EACH action as a complete object in sub_decisions.
   - Each sub_decisions object has exactly: month, arpu_change_pct, marketing_spend_change_pct,
     capacity_change_abs, fixed_cost_change_abs, churn_change_abs, gross_margin_change_abs, one_time_cost, label.
   - Example: "we buy servers in May AND ALSO hire sales reps in June"
   - IMPORTANT: if one of the independent actions involves "increase sales X%" without specifying how,
     include it as a sub_decision with all driver fields = 0 and note the ambiguity in the top-level ambiguity_warning.

3. SINGLE with multiple drivers (type: "single", sub_decisions=[]):
   - A decision can affect several drivers AT ONCE if they apply in the SAME month.
   - Example: "raise price 5% and reduce churn 1% in January" → single, month=1.

CRITICAL RULE — marketing_spend vs fixed_cost_change_abs:
- marketing_spend_change_pct: EXCLUSIVE lever for marketing budget, advertising, ads, lead generation.
  → "reduce marketing by 50%"   → marketing_spend_change_pct = -50, fixed_cost_change_abs = 0
  → "double the marketing"      → marketing_spend_change_pct = 100, fixed_cost_change_abs = 0
  → "cut advertising by 100%"   → marketing_spend_change_pct = -100, fixed_cost_change_abs = 0
- A marketing change NEVER touches fixed_cost_change_abs. They are INDEPENDENT levers.
- fixed_cost_change_abs: ONLY for structural fixed costs (salaries, rent, licences, etc.).
  NEVER for marketing. Example that IS fixed: "hire 2 engineers" → fixed_cost_change_abs > 0.

CRITICAL RULE — One-time vs recurring costs:
- fixed_cost_change_abs: cost that repeats EVERY MONTH from the indicated month (e.g. new employee salary, new subscription).
- one_time_cost: cost that occurs ONLY THAT MONTH, non-recurring (e.g. ad campaign, event, consulting, equipment purchase).
- ONE-TIME indicators: "spend X on...", "campaign", "event", "consulting", "one-off", "that month".
- RECURRING indicators: "hire", "from month X", "new employee", "salary", "monthly subscription".
- If the user says something ambiguous like "spend €X in month Y" without clarifying recurring vs one-time,
  put the amount in one_time_cost (safer default) and add an ambiguity_warning explaining the distinction.

CRITICAL RULE — "Sales" is NEVER ARPU:
- arpu_change_pct ONLY changes when the manager EXPLICITLY says: "price", "ARPU",
  "charge more", "rates", "pricing", "monthly fee", "subscription price".
  → "raise the price"      → arpu_change_pct ≠ 0
  → "increase ARPU"        → arpu_change_pct ≠ 0
  → "charge more"          → arpu_change_pct ≠ 0
  → "raise rates"          → arpu_change_pct ≠ 0
- "sales", "sell more", "more sales", "increase sales", "grow sales", "sales increment",
  "boost sales", "ventas", "vender más", "más ventas", "crecer en ventas", "aumentar ventas"
  → arpu_change_pct = 0 ALWAYS.
  "Sales" means GETTING MORE CUSTOMERS, NOT raising the price.
  → "sell more" without specifying how      → ambiguity_warning with 3 options, all fields = 0
  → "increase sales X%" without specifying  → ambiguity_warning with 3 options, all fields = 0
  → "more sales via more marketing"         → marketing_spend_change_pct
  → "more sales by hiring reps"             → capacity_change_abs
- SPECIAL CASE — "increase sales X%" as part of a multi-action decision:
  If the user combines "increase sales X% in month M" with other clear actions in different months,
  treat the ambiguous "increase sales" part as a sub_decision with all driver fields = 0,
  the specified month, and include the ambiguity in the top-level ambiguity_warning.

CRITICAL RULE — ARPU vs gross_margin:
- arpu_change_pct ONLY changes when the CUSTOMER pays MORE or LESS for the product.
  Examples that ARE ARPU: "raise price by 10%", "discount rates by 5%", "increase monthly fee".
- arpu_change_pct NEVER changes due to internal production or manufacturing costs.
  Examples that are NOT ARPU: "production costs rise", "materials get more expensive",
  "server costs increase", "supplier price rises", "efficiency improvement reduces costs".
- When internal production costs rise → use gross_margin_change_abs negative.
  Example: "production costs rise 5%" → gross_margin_change_abs = -0.05, arpu_change_pct = 0.
- When internal costs fall (efficiency, automation) → use gross_margin_change_abs positive.
  Example: "automation reduces costs by 3%" → gross_margin_change_abs = +0.03, arpu_change_pct = 0.

CRITICAL RULE — Language of label and ambiguity_warning fields:
- Always generate the label and ambiguity_warning fields in English.
- Examples: "Price +10% in month 3", "Marketing x2 and 5 sales reps from month 6", "Churn -1pp from month 2".

CRITICAL RULE — Sales rep cost:
- For sales reps, assume €2,500/month per rep as default fixed_cost_change_abs unless a specific salary is mentioned.
- Examples: "hire 5 sales reps"              → fixed_cost_change_abs = 12500 (5 × 2500)
            "hire 3 sales reps"              → fixed_cost_change_abs = 7500  (3 × 2500)
            "hire 10 reps at €4,000/month"   → fixed_cost_change_abs = 40000 (explicit salary).

RULES FOR ambiguity_warning:
- Set "" (empty string) if the decision is clear and specific.
- Set an English warning if the decision is ambiguous. Ambiguous cases:
  * "increase sales", "boost sales", "sell more", "more sales", "grow sales", "sales increment X%"
    without specifying the MECHANISM → all driver fields = 0 and ambiguity_warning EXACTLY:
    "How do you want to achieve more sales?\n [1] More marketing (more leads)\n [2] More sales reps (more closing capacity)\n [3] Both"
  * "grow faster" or "grow more" without clarifying → same 3-option warning.
  * "improve results" or "improve the business" → too generic, ask for specifics.
  * "acquire more customers" without specifying how → same 3-option warning.
  * "invest more" without clarifying where → ambiguous, ask for specifics.
- The warning must offer concrete, actionable alternatives.

Examples:
User: "raise prices 10% in March"
JSON: {"type": "single", "month": 3, "arpu_change_pct": 10, "marketing_spend_change_pct": 0, "capacity_change_abs": 0, "fixed_cost_change_abs": 0, "churn_change_abs": 0, "gross_margin_change_abs": 0, "one_time_cost": 0, "sub_decisions": [], "label": "Price +10% from month 3", "ambiguity_warning": ""}

User: "double the marketing budget from June and hire 10 sales reps"
JSON: {"type": "single", "month": 6, "arpu_change_pct": 0, "marketing_spend_change_pct": 100, "capacity_change_abs": 10, "fixed_cost_change_abs": 25000, "churn_change_abs": 0, "gross_margin_change_abs": 0, "one_time_cost": 0, "sub_decisions": [], "label": "Marketing x2 and 10 sales reps from month 6", "ambiguity_warning": ""}

User: "spend €4000 on advertising in month 4"
JSON: {"type": "single", "month": 4, "arpu_change_pct": 0, "marketing_spend_change_pct": 0, "capacity_change_abs": 0, "fixed_cost_change_abs": 0, "churn_change_abs": 0, "gross_margin_change_abs": 0, "one_time_cost": 4000, "sub_decisions": [], "label": "One-time ad spend €4,000 in month 4", "ambiguity_warning": ""}

User: "we buy servers for €20,000 in May which allows us to increase sales by 20% from June"
JSON: {"type": "chain", "chain_label": "Servers €20,000 in month 5 → marketing +20% from month 6", "decisions": [{"month": 5, "arpu_change_pct": 0, "marketing_spend_change_pct": 0, "capacity_change_abs": 0, "fixed_cost_change_abs": 0, "churn_change_abs": 0, "gross_margin_change_abs": 0, "one_time_cost": 20000, "label": "Server purchase €20,000 month 5"}, {"month": 6, "arpu_change_pct": 0, "marketing_spend_change_pct": 20, "capacity_change_abs": 0, "fixed_cost_change_abs": 0, "churn_change_abs": 0, "gross_margin_change_abs": 0, "one_time_cost": 0, "label": "Marketing +20% from month 6 enabled by servers"}]}

User: "we invest €15,000 in a new CRM in March which reduces churn by 0.5% from April"
JSON: {"type": "chain", "chain_label": "CRM €15,000 in month 3 → churn -0.5pp from month 4", "decisions": [{"month": 3, "arpu_change_pct": 0, "marketing_spend_change_pct": 0, "capacity_change_abs": 0, "fixed_cost_change_abs": 0, "churn_change_abs": 0, "gross_margin_change_abs": 0, "one_time_cost": 15000, "label": "CRM purchase €15,000 month 3"}, {"month": 4, "arpu_change_pct": 0, "marketing_spend_change_pct": 0, "capacity_change_abs": 0, "fixed_cost_change_abs": 0, "churn_change_abs": -0.005, "gross_margin_change_abs": 0, "one_time_cost": 0, "label": "Churn -0.5pp from month 4"}]}

User: "we buy computers for €40,000 in month 5 and also increase capacity by 20% from month 7"
JSON: {"type": "single", "month": 1, "arpu_change_pct": 0, "marketing_spend_change_pct": 0, "capacity_change_abs": 0, "fixed_cost_change_abs": 0, "churn_change_abs": 0, "gross_margin_change_abs": 0, "one_time_cost": 0, "sub_decisions": [{"month": 5, "arpu_change_pct": 0, "marketing_spend_change_pct": 0, "capacity_change_abs": 0, "fixed_cost_change_abs": 0, "churn_change_abs": 0, "gross_margin_change_abs": 0, "one_time_cost": 40000, "label": "Computer purchase €40,000 month 5"}, {"month": 7, "arpu_change_pct": 0, "marketing_spend_change_pct": 0, "capacity_change_abs": 6, "fixed_cost_change_abs": 0, "churn_change_abs": 0, "gross_margin_change_abs": 0, "one_time_cost": 0, "label": "Capacity increase from month 7"}], "label": "Multiple independent decisions", "ambiguity_warning": ""}

User: "raise price 5% in January and also reduce churn by 1pp"
JSON: {"type": "single", "month": 1, "arpu_change_pct": 5, "marketing_spend_change_pct": 0, "capacity_change_abs": 0, "fixed_cost_change_abs": 0, "churn_change_abs": -0.01, "gross_margin_change_abs": 0, "one_time_cost": 0, "sub_decisions": [], "label": "ARPU +5% and churn -1pp from month 1", "ambiguity_warning": ""}

User: "increase sales by 20% in June"
JSON: {"type": "single", "month": 6, "arpu_change_pct": 0, "marketing_spend_change_pct": 0, "capacity_change_abs": 0, "fixed_cost_change_abs": 0, "churn_change_abs": 0, "gross_margin_change_abs": 0, "one_time_cost": 0, "sub_decisions": [], "label": "Sales increase 20% from month 6 — mechanism unspecified", "ambiguity_warning": "How do you want to achieve more sales?\n [1] More marketing (more leads)\n [2] More sales reps (more closing capacity)\n [3] Both"}

User: "raise price 20% in June"
JSON: {"type": "single", "month": 6, "arpu_change_pct": 20, "marketing_spend_change_pct": 0, "capacity_change_abs": 0, "fixed_cost_change_abs": 0, "churn_change_abs": 0, "gross_margin_change_abs": 0, "one_time_cost": 0, "sub_decisions": [], "label": "Price +20% from month 6", "ambiguity_warning": ""}

User: "Double marketing from June and hire 5 sales reps but increase sales 30% in September"
JSON: {"type": "single", "month": 1, "arpu_change_pct": 0, "marketing_spend_change_pct": 0, "capacity_change_abs": 0, "fixed_cost_change_abs": 0, "churn_change_abs": 0, "gross_margin_change_abs": 0, "one_time_cost": 0, "sub_decisions": [{"month": 6, "arpu_change_pct": 0, "marketing_spend_change_pct": 100, "capacity_change_abs": 5, "fixed_cost_change_abs": 12500, "churn_change_abs": 0, "gross_margin_change_abs": 0, "one_time_cost": 0, "label": "Marketing x2 and 5 sales reps from month 6"}, {"month": 9, "arpu_change_pct": 0, "marketing_spend_change_pct": 0, "capacity_change_abs": 0, "fixed_cost_change_abs": 0, "churn_change_abs": 0, "gross_margin_change_abs": 0, "one_time_cost": 0, "label": "Sales increase 30% from month 9 — mechanism unspecified"}], "label": "Multiple independent decisions", "ambiguity_warning": "The 'increase sales 30%' part is ambiguous. How do you want to achieve it?\n [1] More marketing (more leads)\n [2] More sales reps (more closing capacity)\n [3] Both"}
"""


# Sentinel object returned when NLP fails (no exception raised).
_NLP_FALLBACK = object()


def parse_decision_from_text(user_text: str, api_key: str):
    """
    Calls Groq and returns the parsed JSON.

    If any error occurs (connection, invalid JSON, empty API key),
    returns _NLP_FALLBACK instead of raising an exception, so the app
    can show the manual form without crashing.

    Returns dict | _NLP_FALLBACK.
    """
    if not api_key:
        return _NLP_FALLBACK

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_text},
            ],
            temperature=0.0,
            max_tokens=800,   # chain JSON with 2 decision blocks
            timeout=15,
        )

        raw = response.choices[0].message.content.strip()

    except Exception as e:
        tipo = type(e).__name__
        print(
            f"  ⚠️  Groq connection error ({tipo}): {str(e)[:120]}"
        )
        return _NLP_FALLBACK

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON block found in the response.")
        parsed = json.loads(match.group())

        # ── Validación según type ──────────────────────────────────────────────
        if parsed.get("type") == "chain":
            chain_decs = parsed.get("decisions")
            if not isinstance(chain_decs, list) or len(chain_decs) < 2:
                raise ValueError("chain JSON must have at least 2 decisions in 'decisions'")
            for i, sub in enumerate(chain_decs):
                if not isinstance(sub, dict) or "month" not in sub:
                    raise ValueError(f"chain decisions[{i}] must be a dict with 'month'")
            meses = [d.get("month") for d in chain_decs]
            print(f"  🔗 Chain detected: {len(chain_decs)} decisions in months {meses}")
        else:
            # Validate required numeric fields for single
            for campo in ("month", "arpu_change_pct", "one_time_cost", "fixed_cost_change_abs"):
                if campo not in parsed:
                    raise ValueError(f"Required field missing in JSON: '{campo}'")
                if not isinstance(parsed[campo], (int, float)):
                    raise ValueError(f"Field '{campo}' is not numeric: {parsed[campo]!r}")
            print(
                f"  ✅ NLP OK — month={parsed['month']}  "
                f"arpu={parsed['arpu_change_pct']:+}%  "
                f"one_time={parsed['one_time_cost']}€  "
                f"fixed={parsed['fixed_cost_change_abs']}€/month"
            )
        return parsed

    except (ValueError, json.JSONDecodeError) as e:
        print(
            f"  ⚠️  NLP assistant returned invalid JSON: {str(e)[:80]}\n"
            f"     Response received: {raw[:200]}"
        )
        return _NLP_FALLBACK


def validate_and_build_decision(parsed: dict, user_text: str):
    """
    Validates the LLM JSON and builds a Decision object.
    Returns (Decision, ambiguity_warning: str).
    user_text is used as a fallback for the label if the LLM does not return one.
    """
    month = int(parsed.get("month", 1))
    if not (1 <= month <= 12):
        print(f"  ⚠️  Month {month} out of range — corrected to month 1")
        month = max(1, min(12, month))

    churn_change = float(parsed.get("churn_change_abs", 0.0))
    if not (-1.0 <= churn_change <= 1.0):
        print(f"  ⚠️  churn_change_abs out of range — corrected to 0")
        churn_change = 0.0

    label             = str(parsed.get("label", "")) or user_text[:60]
    ambiguity_warning = str(parsed.get("ambiguity_warning", "")).strip()

    gross_margin_change = float(parsed.get("gross_margin_change_abs", 0.0))
    if not (-1.0 <= gross_margin_change <= 1.0):
        print(f"  ⚠️  gross_margin_change_abs out of range — corrected to 0")
        gross_margin_change = 0.0

    one_time_cost = float(parsed.get("one_time_cost", 0.0))
    if one_time_cost < 0:
        print(f"  ⚠️  one_time_cost is negative — corrected to 0")
        one_time_cost = 0.0

    decision = Decision(
        month                      = month,
        arpu_change_pct            = float(parsed.get("arpu_change_pct", 0.0)),
        marketing_spend_change_pct = float(parsed.get("marketing_spend_change_pct", 0.0)),
        capacity_change_abs        = int(parsed.get("capacity_change_abs", 0)),
        fixed_cost_change_abs      = float(parsed.get("fixed_cost_change_abs", 0.0)),
        churn_change_abs           = churn_change,
        gross_margin_change_abs    = gross_margin_change,
        one_time_cost              = one_time_cost,
        label                      = label,
    )
    return decision, ambiguity_warning


def _build_chain(parsed: dict, user_text: str) -> DecisionChain:
    """
    Builds a DecisionChain from the LLM's type:"chain" JSON.
    Each element of parsed["decisions"] is converted into an individual Decision.
    """
    decisions = []
    for d_dict in parsed.get("decisions", []):
        month = int(d_dict.get("month", 1))
        month = max(1, min(12, month))

        churn = float(d_dict.get("churn_change_abs", 0.0))
        if not (-1.0 <= churn <= 1.0):
            churn = 0.0

        gm = float(d_dict.get("gross_margin_change_abs", 0.0))
        if not (-1.0 <= gm <= 1.0):
            gm = 0.0

        ot = max(0.0, float(d_dict.get("one_time_cost", 0.0)))

        decisions.append(Decision(
            month                      = month,
            arpu_change_pct            = float(d_dict.get("arpu_change_pct", 0.0)),
            marketing_spend_change_pct = float(d_dict.get("marketing_spend_change_pct", 0.0)),
            capacity_change_abs        = int(d_dict.get("capacity_change_abs", 0)),
            fixed_cost_change_abs      = float(d_dict.get("fixed_cost_change_abs", 0.0)),
            churn_change_abs           = churn,
            gross_margin_change_abs    = gm,
            one_time_cost              = ot,
            label                      = str(d_dict.get("label", "")) or f"Decision month {month}",
        ))

    label = str(parsed.get("chain_label", "")).strip() or user_text[:60]
    return DecisionChain(decisions=decisions, label=label)


def build_decisions_from_sub(parsed: dict) -> list:
    """
    Builds a list of Decision objects from a type:"single" JSON
    with a non-empty sub_decisions list (multiple independent actions).
    Ignores elements that are not dicts with a 'month' key.
    Used in app.py to offer "simulate together" or individually.
    """
    result = []
    for d_dict in parsed.get("sub_decisions", []):
        if not isinstance(d_dict, dict) or "month" not in d_dict:
            continue
        month = max(1, min(12, int(d_dict.get("month", 1))))

        churn = float(d_dict.get("churn_change_abs", 0.0))
        if not (-1.0 <= churn <= 1.0):
            churn = 0.0

        gm = float(d_dict.get("gross_margin_change_abs", 0.0))
        if not (-1.0 <= gm <= 1.0):
            gm = 0.0

        ot = max(0.0, float(d_dict.get("one_time_cost", 0.0)))

        result.append(Decision(
            month                      = month,
            arpu_change_pct            = float(d_dict.get("arpu_change_pct", 0.0)),
            marketing_spend_change_pct = float(d_dict.get("marketing_spend_change_pct", 0.0)),
            capacity_change_abs        = int(d_dict.get("capacity_change_abs", 0)),
            fixed_cost_change_abs      = float(d_dict.get("fixed_cost_change_abs", 0.0)),
            churn_change_abs           = churn,
            gross_margin_change_abs    = gm,
            one_time_cost              = ot,
            label                      = str(d_dict.get("label", "")) or f"Decision month {month}",
        ))
    return result


def nlp_to_decision(user_text: str, api_key: str):
    """
    Full pipeline: text → JSON → validated Decision or DecisionChain.

    Returns:
      (Decision, parsed_dict, ambiguity_warning)   → simple decision
      (DecisionChain, parsed_dict, None)           → causal chain (type:"chain")
      (None, parsed_dict, None)                    → multiple independent actions;
                                                      parsed_dict["sub_decisions"] contains the list
      (None, None, None)                           → NLP failed (→ manual form)
    """
    parsed = parse_decision_from_text(user_text, api_key)
    if parsed is _NLP_FALLBACK:
        return None, None, None

    # Cadena causal: construir DecisionChain
    if parsed.get("type") == "chain":
        chain = _build_chain(parsed, user_text)
        return chain, parsed, None

    # Múltiples acciones independientes: informar al usuario sin simular
    sub = parsed.get("sub_decisions", [])
    if isinstance(sub, list) and len(sub) >= 2:
        print(f"  🔀 Multi-action detected: {len(sub)} sub-decisions")
        return None, parsed, None

    decision, warning = validate_and_build_decision(parsed, user_text)
    return decision, parsed, warning
