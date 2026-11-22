"""
visualization.py
Interactive Plotly visualizations for the Bayesian Optimization Algorithm Selection Platform.
Provides publication-quality plots for both Benchmark execution and Algorithm suitability comparison.
"""

from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from problem_schema import RecommendationResult, AlgorithmSuitability


# Modern, polished color palette
PALETTE = [
    "#2563eb",  # Blue (Primary)
    "#059669",  # Emerald Green
    "#d97706",  # Amber
    "#7c3aed",  # Purple
    "#dc2626",  # Red / Coral
    "#0891b2",  # Cyan
    "#4b5563"   # Slate
]


def create_suitability_chart(recommendation: RecommendationResult) -> go.Figure:
    """
    Horizontal bar chart comparing platform-defined algorithm suitability scores (0-100).
    """
    algos = [s.algorithm for s in recommendation.suitability_table]
    scores = [s.score for s in recommendation.suitability_table]
    levels = [s.suitability.value for s in recommendation.suitability_table]

    # Color code by suitability level
    color_map = {
        "High": "#10b981",       # Green
        "Medium": "#3b82f6",     # Blue
        "Low": "#f59e0b",        # Amber
        "Baseline": "#64748b",   # Gray
        "Not Recommended": "#ef4444"
    }
    bar_colors = [color_map.get(lvl, "#3b82f6") for lvl in levels]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scores,
        y=algos,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="#1e293b", width=1)
        ),
        text=[f"{s:.1f} ({lvl})" for s, lvl in zip(scores, levels)],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Suitability Score: %{x:.1f}/100<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="Algorithm Suitability Assessment (Platform-Defined Scores)",
            font=dict(size=15, color="#1e293b")
        ),
        xaxis=dict(
            title="Suitability Score (0–100)",
            range=[0, 115],
            gridcolor="#f1f5f9"
        ),
        yaxis=dict(
            autorange="reversed",
            gridcolor="#f1f5f9"
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=100, r=40, t=50, b=40),
        height=320
    )
    return fig


def create_convergence_plot(results: List[Dict[str, Any]], benchmark_name: str) -> go.Figure:
    """
    Convergence curves showing best objective value found over evaluations.
    """
    fig = go.Figure()

    for i, result in enumerate(results):
        evals = list(range(1, len(result["best_values"]) + 1))
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=evals,
            y=result["best_values"],
            mode="lines+markers",
            name=result["algorithm"],
            marker=dict(size=4, color=color),
            line=dict(color=color, width=2.2),
            hovertemplate="<b>%{data.name}</b><br>Evaluation: %{x}<br>Best Value: %{y:.4f}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text=f"Optimization Convergence Trajectory — {benchmark_name}",
            font=dict(size=15, color="#1e293b")
        ),
        xaxis=dict(
            title="Function Evaluations",
            gridcolor="#f1f5f9"
        ),
        yaxis=dict(
            title="Best Observed Value (Lower is Better)",
            gridcolor="#f1f5f9"
        ),
        hovermode="x unified",
        template="plotly_white",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=30, t=60, b=40)
    )
    return fig


def create_regret_plot(
    results: List[Dict[str, Any]],
    global_optimum: Optional[float] = None
) -> go.Figure:
    """
    Regret progression on log-scale.
    If global optimum is None or unknown, computes empirical regret relative to best observed point.
    """
    fig = go.Figure()

    # Determine reference value
    if global_optimum is not None:
        ref_val = global_optimum
        title_suffix = "(True Regret: f(x) - f*)"
    else:
        # Find global minimum across all algorithms as empirical reference
        all_vals = [min(r["best_values"]) for r in results if r["best_values"]]
        ref_val = min(all_vals) if all_vals else 0.0
        title_suffix = "(Empirical Regret: f(x) - min_observed)"

    for i, result in enumerate(results):
        evals = list(range(1, len(result["best_values"]) + 1))
        regret = [max(1e-8, v - ref_val) for v in result["best_values"]]
        color = PALETTE[i % len(PALETTE)]

        fig.add_trace(go.Scatter(
            x=evals,
            y=regret,
            mode="lines",
            name=result["algorithm"],
            line=dict(color=color, width=2.2),
            hovertemplate="<b>%{data.name}</b><br>Eval: %{x}<br>Regret: %{y:.4e}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text=f"Optimization Regret Progression {title_suffix}",
            font=dict(size=15, color="#1e293b")
        ),
        xaxis=dict(
            title="Function Evaluations",
            gridcolor="#f1f5f9"
        ),
        yaxis=dict(
            title="Regret (Log Scale)",
            type="log",
            gridcolor="#f1f5f9"
        ),
        hovermode="x unified",
        template="plotly_white",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=30, t=60, b=40)
    )
    return fig


def create_score_plot(ranking_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing composite multi-metric scores (0-100).
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ranking_df["Algorithm"],
        y=ranking_df["Score"],
        marker=dict(
            color=PALETTE[:len(ranking_df)],
            line=dict(color="#1e293b", width=1)
        ),
        text=ranking_df["Score"].apply(lambda x: f"{x:.1f}"),
        textposition="auto",
        hovertemplate="<b>%{x}</b><br>Composite Score: %{y:.1f}/100<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="Composite Experimental Ranking Scores (Weighted Multi-Metric)",
            font=dict(size=15, color="#1e293b")
        ),
        xaxis=dict(title="Optimization Algorithm", gridcolor="#f1f5f9"),
        yaxis=dict(title="Score (0–100)", range=[0, 105], gridcolor="#f1f5f9"),
        template="plotly_white",
        height=380,
        margin=dict(l=60, r=30, t=60, b=40)
    )
    return fig


def create_runtime_efficiency_plot(ranking_df: pd.DataFrame) -> go.Figure:
    """
    Grouped or dual-axis plot comparing Runtime and Sample Efficiency.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Runtime (seconds)",
        x=ranking_df["Algorithm"],
        y=ranking_df["Runtime (s)"],
        marker_color="#3b82f6",
        hovertemplate="<b>%{x}</b><br>Runtime: %{y:.2f}s<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="Computational Overhead Comparison (Wall-Clock Runtime)",
            font=dict(size=15, color="#1e293b")
        ),
        xaxis=dict(title="Algorithm", gridcolor="#f1f5f9"),
        yaxis=dict(title="Runtime (seconds)", gridcolor="#f1f5f9"),
        template="plotly_white",
        height=380,
        margin=dict(l=60, r=30, t=60, b=40)
    )
    return fig
