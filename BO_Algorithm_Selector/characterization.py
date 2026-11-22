"""
characterization.py
Problem Characterization Layer for the Bayesian Optimization Algorithm Selection Platform.

Characterizes optimization problems based on verified extracted facts or benchmark mathematics.
Strictly avoids fabricating unverified properties (smoothness, modality, separability) for real-world text.
"""

from typing import Dict, Any, Optional, List
import numpy as np
from benchmarks import BenchmarkFunction, get_benchmark
from problem_schema import ProblemSchema, VariableType


class ProblemCharacterization:
    """
    Holds structured characterization attributes for display and recommendation.
    """
    def __init__(
        self,
        dimensionality: Optional[int],
        dimension_category: str,
        variable_types_summary: str,
        search_space_type: str,
        objective_direction: str,
        evaluation_cost_profile: str,
        noise_profile: str,
        constraint_status: str,
        black_box_nature: str,
        landscape_modality: str,
        landscape_smoothness: str,
        effective_dimension_info: str,
        budget: Optional[int],
        cost_str: Optional[str],
        is_benchmark: bool = False,
        benchmark_name: Optional[str] = None
    ):
        self.dimensionality = dimensionality
        self.dimension_category = dimension_category
        self.variable_types_summary = variable_types_summary
        self.search_space_type = search_space_type
        self.objective_direction = objective_direction
        self.evaluation_cost_profile = evaluation_cost_profile
        self.noise_profile = noise_profile
        self.constraint_status = constraint_status
        self.black_box_nature = black_box_nature
        self.landscape_modality = landscape_modality
        self.landscape_smoothness = landscape_smoothness
        self.effective_dimension_info = effective_dimension_info
        self.budget = budget
        self.cost_str = cost_str
        self.is_benchmark = is_benchmark
        self.benchmark_name = benchmark_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimensionality": self.dimensionality,
            "dimension_category": self.dimension_category,
            "variable_types_summary": self.variable_types_summary,
            "search_space_type": self.search_space_type,
            "objective_direction": self.objective_direction,
            "evaluation_cost_profile": self.evaluation_cost_profile,
            "noise_profile": self.noise_profile,
            "constraint_status": self.constraint_status,
            "black_box_nature": self.black_box_nature,
            "landscape_modality": self.landscape_modality,
            "landscape_smoothness": self.landscape_smoothness,
            "effective_dimension_info": self.effective_dimension_info,
            "budget": self.budget,
            "cost_str": self.cost_str,
            "is_benchmark": self.is_benchmark,
            "benchmark_name": self.benchmark_name
        }


def characterize_parsed_problem(problem: ProblemSchema) -> ProblemCharacterization:
    """
    Characterize a real-world natural language problem from its parsed schema.
    Adheres strictly to scientific rules:
    - Never fabricates modality, smoothness, or effective dimension when objective is black-box.
    """
    dim = problem.dimension

    # Dimensionality category
    if dim is None or dim == 0:
        dim_category = "Unknown / Not specified"
    elif dim <= 6:
        dim_category = f"Low-dimensional (d = {dim})"
    elif dim <= 20:
        dim_category = f"Moderate-dimensional (d = {dim})"
    elif dim <= 50:
        dim_category = f"High-dimensional (d = {dim})"
    else:
        dim_category = f"Very High-dimensional (d = {dim})"

    # Variable types analysis
    var_types = set(v.var_type for v in problem.variables) if problem.variables else set()
    if not var_types:
        var_summary = "Unknown"
        search_space = "Unspecified search space"
    elif len(var_types) == 1:
        single_type = list(var_types)[0]
        if single_type == VariableType.CONTINUOUS:
            var_summary = "Continuous"
            search_space = "Box-constrained continuous [lb, ub]^d"
        elif single_type == VariableType.INTEGER:
            var_summary = "Discrete / Integer"
            search_space = "Integer grid search space"
        else:
            var_summary = "Categorical"
            search_space = "Combinatorial / Categorical space"
    else:
        var_summary = "Mixed (Continuous & Integer/Categorical)"
        search_space = "Mixed-variable search space"

    # Any missing bounds?
    unbounded = [v.name for v in problem.variables if v.lower_bound is None or v.upper_bound is None]
    if unbounded:
        search_space += f" (Note: Missing bounds for {len(unbounded)} variable(s))"

    # Objective direction
    direction = problem.objective.obj_type.value.capitalize()

    # Cost profile
    if problem.expensive_evaluation:
        if problem.evaluation_cost:
            cost_profile = f"Expensive ({problem.evaluation_cost})"
        else:
            cost_profile = "Expensive / Resource-intensive"
    elif problem.evaluation_cost:
        cost_profile = f"Moderate ({problem.evaluation_cost})"
    else:
        cost_profile = "Unknown / Presumed moderate"

    # Noise profile (SCIENTIFIC HONESTY)
    if problem.noise is not None:
        if problem.noise < 0.02:
            noise_profile = f"Low noise (std = {problem.noise:.2f})"
        elif problem.noise <= 0.15:
            noise_profile = f"Moderate noise (std = {problem.noise:.2f})"
        else:
            noise_profile = f"High noise (std = {problem.noise:.2f})"
    else:
        noise_profile = "Unknown / Not specified (Assumed noisy or black-box)"

    # Constraint status
    if problem.constraints:
        constraint_status = f"Constrained ({len(problem.constraints)} operational constraint(s))"
    else:
        constraint_status = "Box-constrained / Unconstrained"

    # Black-box nature
    black_box = "Black-box physical / experimental process"

    # Properties requiring objective evaluation (NEVER FABRICATED)
    modality = "Unknown / Requires objective evaluation"
    smoothness = "Unknown / Requires objective evaluation"
    eff_dim_info = "Unknown / Requires objective evaluation"

    return ProblemCharacterization(
        dimensionality=dim,
        dimension_category=dim_category,
        variable_types_summary=var_summary,
        search_space_type=search_space,
        objective_direction=direction,
        evaluation_cost_profile=cost_profile,
        noise_profile=noise_profile,
        constraint_status=constraint_status,
        black_box_nature=black_box,
        landscape_modality=modality,
        landscape_smoothness=smoothness,
        effective_dimension_info=eff_dim_info,
        budget=problem.evaluation_budget,
        cost_str=problem.evaluation_cost,
        is_benchmark=False
    )


def estimate_effective_dimension(
    benchmark: BenchmarkFunction,
    dimension: int,
    n_samples: int = 25
) -> int:
    """
    Estimate active/effective dimensionality for synthetic benchmark functions
    via numerical finite-difference gradient estimation.
    """
    np.random.seed(42)
    lb, ub = benchmark.bounds
    k = min(n_samples, 20)
    X = np.random.uniform(lb, ub, (k, dimension))

    gradients = np.zeros((k, dimension))
    eps = 1e-4
    for i in range(k):
        for j in range(dimension):
            x_plus = X[i].copy()
            x_minus = X[i].copy()
            x_plus[j] += eps
            x_minus[j] -= eps
            gradients[i, j] = (benchmark.func(x_plus) - benchmark.func(x_minus)) / (2 * eps)

    mean_grads = np.mean(np.abs(gradients), axis=0)
    max_g = np.max(mean_grads)
    if max_g > 1e-9:
        active_dims = np.sum(mean_grads > 1e-3 * max_g)
    else:
        active_dims = dimension

    return max(1, int(active_dims))


def get_problem_characteristics(
    benchmark_name: str,
    dimension: int,
    noise: float
) -> Dict[str, Any]:
    """
    Legacy compatibility function for synthetic benchmark functions (Mode B).
    """
    benchmark = get_benchmark(benchmark_name)

    modality_map = {
        "Unimodal": "Low (Unimodal)",
        "Multimodal": "Medium (Multimodal)",
        "Highly Multimodal": "High (Highly Multimodal)"
    }

    smoothness_map = {
        "High": "High (C2 continuous)",
        "Medium": "Medium (Continuous with high oscillation)",
        "Low": "Low (Non-smooth / Deceptive)"
    }

    interaction_map = {
        "None": "None (Fully separable)",
        "Low": "Low (Partially separable)",
        "Medium": "Medium (Coupled variables)",
        "High": "High (Strong non-linear coupling)"
    }

    effective_dim = estimate_effective_dimension(benchmark, dimension)

    return {
        "dimension": dimension,
        "noise": noise,
        "modality": modality_map.get(benchmark.modality, "Medium"),
        "smoothness": smoothness_map.get(benchmark.smoothness, "Medium"),
        "variable_interaction": interaction_map.get(benchmark.variable_interaction, "Medium"),
        "effective_dimension": effective_dim,
        "global_optimum": benchmark.global_optimum,
        "bounds": benchmark.bounds,
        "description": benchmark.description
    }