import numpy as np
from typing import Dict, Any, Callable, Tuple, List
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
import time
import warnings
warnings.filterwarnings("ignore")


def expected_improvement(
    X: np.ndarray,
    gp: GaussianProcessRegressor,
    y_best: float,
    xi: float = 0.01
) -> np.ndarray:
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    z = (y_best - mu - xi) / sigma
    from scipy.stats import norm
    ei = (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei


def upper_confidence_bound(
    X: np.ndarray,
    gp: GaussianProcessRegressor,
    beta: float = 2.0
) -> np.ndarray:
    mu, sigma = gp.predict(X, return_std=True)
    return -(mu - beta * sigma)


def create_gp_kernel(dimension: int):
    if dimension <= 10:
        return C(1.0, (1e-2, 1e2)) * Matern(
            length_scale=np.ones(dimension),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5
        )
    else:
        return C(1.0, (1e-2, 1e2)) * Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5
        )


def get_gp_restarts(dimension: int) -> int:
    if dimension <= 6:
        return 2
    elif dimension <= 15:
        return 1
    return 0


def gp_ei_optimize(
    objective: Callable,
    bounds: Tuple[float, float],
    dimension: int,
    n_initial: int,
    budget: int,
    noise: float,
    seed: int
) -> Dict[str, Any]:
    np.random.seed(seed)
    start_time = time.time()

    lb, ub = bounds
    X_init = np.random.uniform(lb, ub, (n_initial, dimension))
    y_init = np.array([objective(x) for x in X_init])

    X = X_init.copy()
    y = y_init.copy()

    best_values = [np.min(y)]
    evaluations = list(y)

    kernel = create_gp_kernel(dimension)
    restarts = get_gp_restarts(dimension)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=noise**2 + 1e-6,
        normalize_y=True,
        n_restarts_optimizer=restarts,
        random_state=seed
    )

    n_candidates = min(1500, max(200, 30 * dimension))
    for i in range(budget - n_initial):
        gp.fit(X, y)
        y_best = np.min(y)

        X_candidates = np.random.uniform(lb, ub, (n_candidates, dimension))
        ei = expected_improvement(X_candidates, gp, y_best)
        next_idx = np.argmax(ei)
        x_next = X_candidates[next_idx].reshape(1, -1)

        y_next = objective(x_next[0])

        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

        best_values.append(np.min(y))
        evaluations.append(y_next)

    runtime = time.time() - start_time
    best_idx = np.argmin(y)

    return {
        "algorithm": "GP-EI",
        "evaluations": evaluations,
        "objective_values": y.tolist(),
        "best_values": best_values,
        "best_value": float(y[best_idx]),
        "best_point": X[best_idx].tolist(),
        "runtime": runtime
    }


def gp_ucb_optimize(
    objective: Callable,
    bounds: Tuple[float, float],
    dimension: int,
    n_initial: int,
    budget: int,
    noise: float,
    seed: int
) -> Dict[str, Any]:
    np.random.seed(seed)
    start_time = time.time()

    lb, ub = bounds
    X_init = np.random.uniform(lb, ub, (n_initial, dimension))
    y_init = np.array([objective(x) for x in X_init])

    X = X_init.copy()
    y = y_init.copy()

    best_values = [np.min(y)]
    evaluations = list(y)

    kernel = create_gp_kernel(dimension)
    restarts = get_gp_restarts(dimension)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=noise**2 + 1e-6,
        normalize_y=True,
        n_restarts_optimizer=restarts,
        random_state=seed
    )

    n_candidates = min(1500, max(200, 30 * dimension))
    for i in range(budget - n_initial):
        gp.fit(X, y)
        beta = 2.0 * np.log((i + n_initial + 1)**2 * dimension / 0.1)

        X_candidates = np.random.uniform(lb, ub, (n_candidates, dimension))
        ucb = upper_confidence_bound(X_candidates, gp, beta)
        next_idx = np.argmax(ucb)
        x_next = X_candidates[next_idx].reshape(1, -1)

        y_next = objective(x_next[0])

        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

        best_values.append(np.min(y))
        evaluations.append(y_next)

    runtime = time.time() - start_time
    best_idx = np.argmin(y)

    return {
        "algorithm": "GP-UCB",
        "evaluations": evaluations,
        "objective_values": y.tolist(),
        "best_values": best_values,
        "best_value": float(y[best_idx]),
        "best_point": X[best_idx].tolist(),
        "runtime": runtime
    }


def rembo_optimize(
    objective: Callable,
    bounds: Tuple[float, float],
    dimension: int,
    n_initial: int,
    budget: int,
    noise: float,
    seed: int,
    embedding_dim: int = None
) -> Dict[str, Any]:
    np.random.seed(seed)
    start_time = time.time()

    if embedding_dim is None:
        embedding_dim = min(10, max(2, dimension // 5))

    lb, ub = bounds
    A = np.random.randn(embedding_dim, dimension)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)

    def low_dim_objective(z: np.ndarray) -> float:
        x = np.clip(A.T @ z, lb, ub)
        return objective(x)

    z_bounds = (-np.sqrt(dimension), np.sqrt(dimension))

    z_init = np.random.uniform(z_bounds[0], z_bounds[1], (n_initial, embedding_dim))
    X_init = np.clip(z_init @ A, lb, ub)
    y_init = np.array([objective(x) for x in X_init])

    X = X_init.copy()
    y = y_init.copy()
    Z = z_init.copy()

    best_values = [np.min(y)]
    evaluations = list(y)

    kernel = create_gp_kernel(embedding_dim)
    restarts = get_gp_restarts(embedding_dim)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=noise**2 + 1e-6,
        normalize_y=True,
        n_restarts_optimizer=restarts,
        random_state=seed
    )

    n_candidates = min(1500, max(200, 30 * embedding_dim))
    for i in range(budget - n_initial):
        gp.fit(Z, y)
        y_best = np.min(y)

        Z_candidates = np.random.uniform(z_bounds[0], z_bounds[1], (n_candidates, embedding_dim))
        ei = expected_improvement(Z_candidates, gp, y_best)
        next_idx = np.argmax(ei)
        z_next = Z_candidates[next_idx].reshape(1, -1)
        x_next = np.clip(z_next @ A, lb, ub)

        y_next = objective(x_next[0])

        Z = np.vstack([Z, z_next])
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

        best_values.append(np.min(y))
        evaluations.append(y_next)

    runtime = time.time() - start_time
    best_idx = np.argmin(y)

    return {
        "algorithm": "REMBO (Prototype)",
        "evaluations": evaluations,
        "objective_values": y.tolist(),
        "best_values": best_values,
        "best_value": float(y[best_idx]),
        "best_point": X[best_idx].tolist(),
        "runtime": runtime
    }


def turbo_optimize(
    objective: Callable,
    bounds: Tuple[float, float],
    dimension: int,
    n_initial: int,
    budget: int,
    noise: float,
    seed: int
) -> Dict[str, Any]:
    np.random.seed(seed)
    start_time = time.time()

    lb, ub = bounds
    X_init = np.random.uniform(lb, ub, (n_initial, dimension))
    y_init = np.array([objective(x) for x in X_init])

    X = X_init.copy()
    y = y_init.copy()

    best_values = [np.min(y)]
    evaluations = list(y)

    trust_region_size = 0.8 * (ub - lb)
    trust_region_center = X[np.argmin(y)].copy()
    min_tr_size = 1e-3 * (ub - lb)
    max_tr_size = ub - lb
    success_count = 0
    failure_count = 0
    success_tolerance = 3
    failure_tolerance = 5

    kernel = create_gp_kernel(dimension)
    restarts = get_gp_restarts(dimension)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=noise**2 + 1e-6,
        normalize_y=True,
        n_restarts_optimizer=restarts,
        random_state=seed
    )

    n_candidates = min(1500, max(200, 30 * dimension))
    for i in range(budget - n_initial):
        tr_lb = np.maximum(trust_region_center - trust_region_size / 2, lb)
        tr_ub = np.minimum(trust_region_center + trust_region_size / 2, ub)

        gp.fit(X, y)
        y_best = np.min(y)

        X_candidates = np.random.uniform(tr_lb, tr_ub, (n_candidates, dimension))
        ei = expected_improvement(X_candidates, gp, y_best)
        next_idx = np.argmax(ei)
        x_next = X_candidates[next_idx].reshape(1, -1)

        y_next = objective(x_next[0])

        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

        if y_next < y_best:
            success_count += 1
            failure_count = 0
            trust_region_center = x_next[0].copy()
        else:
            failure_count += 1
            success_count = 0

        if success_count >= success_tolerance:
            trust_region_size = np.minimum(trust_region_size * 2, max_tr_size)
            success_count = 0
        elif failure_count >= failure_tolerance:
            trust_region_size = np.maximum(trust_region_size / 2, min_tr_size)
            failure_count = 0

        best_values.append(np.min(y))
        evaluations.append(y_next)

    runtime = time.time() - start_time
    best_idx = np.argmin(y)

    return {
        "algorithm": "TuRBO (Prototype)",
        "evaluations": evaluations,
        "objective_values": y.tolist(),
        "best_values": best_values,
        "best_value": float(y[best_idx]),
        "best_point": X[best_idx].tolist(),
        "runtime": runtime
    }


def random_search_optimize(
    objective: Callable,
    bounds: Tuple[float, float],
    dimension: int,
    n_initial: int,
    budget: int,
    noise: float,
    seed: int
) -> Dict[str, Any]:
    np.random.seed(seed)
    start_time = time.time()

    lb, ub = bounds
    X = np.random.uniform(lb, ub, (budget, dimension))
    y = np.array([objective(x) for x in X])

    best_values = []
    current_best = np.inf
    for val in y:
        current_best = min(current_best, val)
        best_values.append(current_best)

    evaluations = y.tolist()
    best_idx = np.argmin(y)
    runtime = time.time() - start_time

    return {
        "algorithm": "Random Search",
        "evaluations": evaluations,
        "objective_values": y.tolist(),
        "best_values": best_values,
        "best_value": float(y[best_idx]),
        "best_point": X[best_idx].tolist(),
        "runtime": runtime
    }


ALGORITHMS = {
    "GP-EI": gp_ei_optimize,
    "GP-UCB": gp_ucb_optimize,
    "REMBO": rembo_optimize,
    "TuRBO": turbo_optimize,
    "Random Search": random_search_optimize
}


def run_algorithm(
    name: str,
    objective: Callable,
    bounds: Tuple[float, float],
    dimension: int,
    n_initial: int,
    budget: int,
    noise: float,
    seed: int
) -> Dict[str, Any]:
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {name}")
    return ALGORITHMS[name](
        objective, bounds, dimension, n_initial, budget, noise, seed
    )


def list_algorithms() -> list:
    return list(ALGORITHMS.keys())