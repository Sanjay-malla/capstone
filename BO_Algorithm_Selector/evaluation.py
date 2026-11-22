"""
evaluation.py
Performance Evaluation and Multi-Metric Ranking Engine for Bayesian Optimization Benchmarks.

Metric Weightings (Project-Defined Composite Score, 0-100):
- Best Value: 40% (Min-max inverted: lower objective is superior)
- Regret: 25% (Min-max inverted: lower distance to optimum is superior)
- Runtime: 10% (Min-max inverted: lower wall-clock time is superior)
- Sample Efficiency: 15% (Min-max scaled: higher mean reduction per step is superior)
- Convergence Rate: 10% (Min-max scaled: higher regret reduction ratio is superior)

Notice: The composite score is a platform-defined pedagogical ranking metric,
not a universal scientific constant.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


def calculate_regret(
    best_values: List[float],
    global_optimum: Optional[float]
) -> List[float]:
    """
    Calculate regret at each evaluation step.
    If true global optimum is unknown, computes empirical regret relative to the best observed value.
    """
    if global_optimum is not None:
        return [max(0.0, float(v - global_optimum)) for v in best_values]
    else:
        # Empirical regret: distance from best observed value across sequence
        best_ever = min(best_values) if best_values else 0.0
        return [max(0.0, float(v - best_ever)) for v in best_values]


def calculate_metrics(
    result: Dict[str, Any],
    global_optimum: Optional[float],
    budget: int
) -> Dict[str, Any]:
    """
    Compute experimental performance metrics for a single algorithm run.
    """
    best_values = result.get("best_values", [])
    evaluations = result.get("evaluations", [])
    runtime = result.get("runtime", 0.0)

    if not best_values:
        return {
            "best_value": float("inf"),
            "regret": float("inf"),
            "runtime": runtime,
            "evaluation_count": 0,
            "sample_efficiency": 0.0,
            "convergence_rate": 0.0,
            "regret_curve": [],
            "is_empirical_regret": global_optimum is None
        }

    best_value = float(result.get("best_value", min(best_values)))
    regret_curve = calculate_regret(best_values, global_optimum)
    final_regret = regret_curve[-1] if regret_curve else 0.0

    # Sample efficiency: mean magnitude of negative steps (objective reductions)
    if len(best_values) > 1:
        diffs = np.diff(best_values)
        improvements = -diffs[diffs < 0]
        sample_efficiency = float(np.mean(improvements)) if len(improvements) > 0 else 0.0
    else:
        sample_efficiency = 0.0

    # Convergence rate: ratio of regret eliminated relative to initial sample
    convergence_rate = 0.0
    if len(regret_curve) >= 2:
        initial_regret = regret_curve[0]
        final_regret_val = regret_curve[-1]
        if initial_regret > 1e-12:
            convergence_rate = float(max(0.0, 1.0 - (final_regret_val / initial_regret)))

    return {
        "best_value": best_value,
        "regret": final_regret,
        "runtime": runtime,
        "evaluation_count": len(evaluations),
        "sample_efficiency": sample_efficiency,
        "convergence_rate": convergence_rate,
        "regret_curve": regret_curve,
        "is_empirical_regret": global_optimum is None
    }


def normalize_scores(
    results: List[Dict[str, Any]],
    metrics_list: List[Dict[str, Any]]
) -> List[float]:
    """
    Normalize metrics using min-max scaling and compute weighted composite scores (0-100).
    """
    n = len(results)
    if n == 0:
        return []

    metric_names = ["best_value", "regret", "runtime", "sample_efficiency", "convergence_rate"]
    weights = [0.40, 0.25, 0.10, 0.15, 0.10]

    values = np.zeros((n, len(metric_names)))
    for i, m in enumerate(metrics_list):
        for j, name in enumerate(metric_names):
            val = m.get(name, 0.0)
            values[i, j] = 0.0 if np.isinf(val) or np.isnan(val) else float(val)

    normalized = np.zeros_like(values)
    for j in range(len(metric_names)):
        col = values[:, j]
        min_val, max_val = float(np.min(col)), float(np.max(col))
        if max_val > min_val:
            # For best_value, regret, and runtime: smaller is better -> inverted normalization
            if metric_names[j] in ["best_value", "regret", "runtime"]:
                normalized[:, j] = 1.0 - (col - min_val) / (max_val - min_val)
            else:
                # For sample_efficiency and convergence_rate: higher is better
                normalized[:, j] = (col - min_val) / (max_val - min_val)
        else:
            normalized[:, j] = 1.0

    scores = np.sum(normalized * weights, axis=1) * 100.0
    return [round(float(s), 1) for s in scores]


def rank_algorithms(
    results: List[Dict[str, Any]],
    global_optimum: Optional[float],
    budget: int
) -> pd.DataFrame:
    """
    Generate complete ranking table with all evaluation metrics and composite score.
    """
    metrics_list = [calculate_metrics(r, global_optimum, budget) for r in results]
    scores = normalize_scores(results, metrics_list)

    rows = []
    for i, (result, metrics, score) in enumerate(zip(results, metrics_list, scores)):
        regret_col = "Regret (Empirical)" if metrics.get("is_empirical_regret") else "Regret"
        rows.append({
            "Rank": 0,
            "Algorithm": result["algorithm"],
            "Best Value": round(metrics["best_value"], 4),
            regret_col: round(metrics["regret"], 4),
            "Runtime (s)": round(metrics["runtime"], 2),
            "Sample Efficiency": round(metrics["sample_efficiency"], 4),
            "Convergence Rate": round(metrics["convergence_rate"], 4),
            "Score": score
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    return df


def get_best_algorithm(
    results: List[Dict[str, Any]],
    global_optimum: Optional[float],
    budget: int
) -> Dict[str, Any]:
    """Return the algorithm result with the highest composite score."""
    df = rank_algorithms(results, global_optimum, budget)
    best_algo_name = df.iloc[0]["Algorithm"]
    for r in results:
        if r["algorithm"] == best_algo_name:
            return r
    return results[0]