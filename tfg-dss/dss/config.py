"""
Data structures for the DSS simulation.
"""

import warnings
from dataclasses import dataclass, field


@dataclass
class CompanyConfig:
    """
    Baseline assumptions for the simulated company.
    Monetary unit: euros. Time unit: month.
    """
    # Customers
    initial_customers: int = 100
    initial_arpu: float = 500.0
    monthly_churn_rate: float = 0.03

    # Unit economics
    gross_margin: float = 0.75

    # Costs
    fixed_costs_monthly: float = 20_000.0
    marketing_spend_monthly: float = 5_000.0

    # Acquisition model with diminishing returns
    # new_customers = min(alpha * spend^beta, capacity)
    #
    # beta = 1.0  → linear returns (special case)
    # beta < 1.0  → diminishing returns: doubling the budget does NOT double customers
    #               justification: addressable market gradually saturates,
    #               the first euros of marketing are the most efficient.
    # beta = 0.8  → conservative choice, well documented in marketing literature
    marketing_efficiency_alpha: float = 0.004
    acquisition_beta: float = 0.8

    max_new_customers_capacity: int = 30

    # Cash
    initial_cash: float = 150_000.0

    # Horizon
    horizon_months: int = 12

    # Price-churn sensitivity
    # For every 10% ARPU increase, churn rises by price_churn_sensitivity pp.
    # Only triggered if the cumulative increase exceeds price_churn_threshold.
    # Example with defaults: ARPU +20% → churn +0.6 pp (from 3% to 3.6%)
    price_churn_sensitivity: float = 0.3
    price_churn_threshold: float   = 0.05

    # Monthly seasonality (acquisition multiplier)
    # Represents B2B corporate budget cycles:
    #   - Strong Q1 (new budgets), slow summer, Q4 peak (year-end close).
    # With all values at 1.0 the behaviour without seasonality is restored.
    seasonality_factors: list = field(default_factory=lambda: [
        1.10,  # January   — strong Q1 (new budgets)
        1.05,  # February
        1.08,  # March     — quarter-end close
        0.95,  # April
        0.90,  # May
        0.80,  # June      — summer start, slow decisions
        0.75,  # July      — slow summer
        0.78,  # August
        0.95,  # September — back to work
        1.05,  # October   — Q4 starts strong
        1.10,  # November
        1.20,  # December  — year-end close, peak budget
    ])

    def validate(self):
        """Validates ranges. Uses raise ValueError (assert can be disabled with -O)."""
        if not (0.0 <= self.monthly_churn_rate <= 1.0):
            raise ValueError(f"churn must be in [0,1], received: {self.monthly_churn_rate}")
        if self.initial_arpu < 0:
            raise ValueError(f"ARPU must be >= 0, received: {self.initial_arpu}")
        if self.marketing_spend_monthly < 0:
            raise ValueError("Marketing spend must be >= 0")
        if self.fixed_costs_monthly < 0:
            raise ValueError("Fixed costs must be >= 0")
        if self.max_new_customers_capacity < 0:
            raise ValueError("Capacity must be >= 0")
        if self.gross_margin < 0:
            raise ValueError("Gross margin must be >= 0")
        if self.marketing_efficiency_alpha < 0:
            raise ValueError("Alpha must be >= 0")
        if not (0.0 < self.acquisition_beta <= 1.0):
            raise ValueError(f"beta must be in (0,1], received: {self.acquisition_beta}")
        if len(self.seasonality_factors) != self.horizon_months:
            raise ValueError(
                f"seasonality_factors must have {self.horizon_months} elements, "
                f"has {len(self.seasonality_factors)}"
            )
        if any(f < 0 for f in self.seasonality_factors):
            raise ValueError("All seasonality factors must be >= 0")

        if self.monthly_churn_rate > 0.20:
            warnings.warn(
                f"Churn {self.monthly_churn_rate:.0%} exceeds 20% — "
                "unusually high for SaaS B2B.",
                UserWarning, stacklevel=2
            )

        if self.gross_margin < 0.20:
            warnings.warn(
                f"Gross margin {self.gross_margin:.0%} is below 20%.",
                UserWarning, stacklevel=2
            )

        if self.marketing_spend_monthly > self.fixed_costs_monthly * 3:
            warnings.warn(
                f"Marketing (€{self.marketing_spend_monthly:,.0f}) exceeds "
                f"3× fixed costs (€{self.fixed_costs_monthly:,.0f}).",
                UserWarning, stacklevel=2
            )

        return self


@dataclass
class SimulationResult:
    """Complete simulation result."""
    scenario_name: str
    monthly_df: object
    summary: dict = field(default_factory=dict)
