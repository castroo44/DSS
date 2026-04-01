"""
Visualisation functions (matplotlib only).
Each function saves the figure to disk and displays it on screen.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List
from dss.config import SimulationResult

_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9)


def plot_cash(results: List[SimulationResult], save_path="cash_over_time.png"):
    """Cash (€) evolution for each scenario."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, res in enumerate(results):
        df = res.monthly_df
        ax.plot(df["month"], df["cash"], marker="o", markersize=4,
                color=_COLORS[i % len(_COLORS)], label=res.scenario_name)
    ax.axhline(0, color="red", linewidth=1.2, linestyle="--", alpha=0.7, label="Insolvency (€0)")
    _style_ax(ax, "Cash over time", "Month", "Cash (€)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"  [saved] {save_path}")


def plot_customers(results: List[SimulationResult], save_path="customers_over_time.png"):
    """Customer count evolution for each scenario."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, res in enumerate(results):
        df = res.monthly_df
        ax.plot(df["month"], df["customers"], marker="s", markersize=4,
                color=_COLORS[i % len(_COLORS)], label=res.scenario_name)
    _style_ax(ax, "Customers over time", "Month", "Customers (#)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"  [saved] {save_path}")


def plot_mrr(results: List[SimulationResult], save_path="mrr_over_time.png"):
    """MRR (€) evolution for each scenario."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, res in enumerate(results):
        df = res.monthly_df
        ax.plot(df["month"], df["mrr"], marker="^", markersize=4,
                color=_COLORS[i % len(_COLORS)], label=res.scenario_name)
    _style_ax(ax, "MRR over time", "Month", "MRR (€)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"  [saved] {save_path}")


def plot_monte_carlo(mc_result: dict, save_path="monte_carlo_cash12.png"):
    """Histogram of the Monte Carlo Cash₁₂ distribution."""
    samples = mc_result["cash_12_samples"]
    pct     = mc_result["percentiles"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(samples, bins=40, color="#2196F3", edgecolor="white", alpha=0.8)
    for p, ls, lbl in [(5, "--", "P5"), (50, "-", "P50 (median)"), (95, "--", "P95")]:
        ax.axvline(pct[p], color="#FF5722", linestyle=ls, linewidth=1.5,
                   label=f"{lbl}: €{pct[p]:,.0f}")
    ax.axvline(0, color="red", linewidth=1.2, linestyle=":", label="Insolvency (€0)")
    _style_ax(ax,
              f"Monte Carlo — Cash₁₂ Distribution  (N={mc_result['n_simulations']})",
              "Cash at month 12 (€)", "Frequency")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"  [saved] {save_path}")


def plot_sensitivity(rank_df, save_path="sensitivity_ranking.png"):
    """OAT sensitivity ranking (horizontal bars)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(rank_df["parameter"], rank_df["mean_abs_Δcash_12"],
            color="#4CAF50", edgecolor="white")
    ax.set_xlabel("Mean |ΔCash₁₂| (€)", fontsize=10)
    ax.set_title("OAT Sensitivity Ranking — Impact on Cash₁₂",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"  [saved] {save_path}")
