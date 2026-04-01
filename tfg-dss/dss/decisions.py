"""
Decision and Scenario objects.
A Decision modifies a driver starting from a specific month.
"""

from dataclasses import dataclass


@dataclass
class Decision:
    """
    Managerial intervention applied in the indicated month.

    Parameters
    ----------
    month                     : month when the decision takes effect (1-indexed)
    arpu_change_pct           : % change in ARPU  (e.g. +5 → +5%)
    marketing_spend_change_pct: % change in the marketing budget
    capacity_change_abs       : absolute change in maximum acquisition capacity
    fixed_cost_change_abs     : absolute change in fixed costs (€)
    churn_change_abs          : absolute change in churn rate
    gross_margin_change_abs   : absolute change in gross margin (e.g. -0.05 = −5 pp)
    label                     : human-readable description
    """
    month: int
    arpu_change_pct: float = 0.0
    marketing_spend_change_pct: float = 0.0
    capacity_change_abs: int = 0
    fixed_cost_change_abs: float = 0.0
    churn_change_abs: float = 0.0
    gross_margin_change_abs: float = 0.0
    one_time_cost: float = 0.0
    label: str = ""


@dataclass
class DecisionChain:
    """
    Group of decisions with chained effects across different months.
    Example: server purchase in month 5 → capacity increase in month 6.

    The engine flattens the chain into individual decisions before simulating,
    so each Decision retains its own application month.
    """
    decisions: list   # lista de Decision, ordenadas por mes
    label: str = ""


@dataclass
class Scenario:
    """
    Scenario = baseline configuration + ordered list of decisions.
    """
    name: str
    config: object       # CompanyConfig
    decisions: list = None

    def __post_init__(self):
        if self.decisions is None:
            self.decisions = []
