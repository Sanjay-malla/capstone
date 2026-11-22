import numpy as np
from typing import Callable, Dict, Tuple, Any


class BenchmarkFunction:
    def __init__(
        self,
        name: str,
        func: Callable,
        bounds: Tuple[float, float],
        global_optimum: float,
        modality: str,
        smoothness: str,
        variable_interaction: str,
        description: str
    ):
        self.name = name
        self.func = func
        self.bounds = bounds
        self.global_optimum = global_optimum
        self.modality = modality
        self.smoothness = smoothness
        self.variable_interaction = variable_interaction
        self.description = description

    def evaluate(self, x: np.ndarray, noise: float = 0.0) -> float:
        y = self.func(x)
        if noise > 0:
            y += np.random.normal(0, noise)
        return y


def sphere(x: np.ndarray) -> float:
    return np.sum(x**2)


def rosenbrock(x: np.ndarray) -> float:
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def ackley(x: np.ndarray) -> float:
    a, b, c = 20.0, 0.2, 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e


def schwefel(x: np.ndarray) -> float:
    d = len(x)
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))


BENCHMARKS: Dict[str, BenchmarkFunction] = {
    "Sphere": BenchmarkFunction(
        name="Sphere",
        func=sphere,
        bounds=(-5.12, 5.12),
        global_optimum=0.0,
        modality="Unimodal",
        smoothness="High",
        variable_interaction="None",
        description="Simple convex function, easy to optimize"
    ),
    "Rosenbrock": BenchmarkFunction(
        name="Rosenbrock",
        func=rosenbrock,
        bounds=(-5.0, 10.0),
        global_optimum=0.0,
        modality="Unimodal",
        smoothness="High",
        variable_interaction="High",
        description="Narrow curved valley, challenging for gradient methods"
    ),
    "Rastrigin": BenchmarkFunction(
        name="Rastrigin",
        func=rastrigin,
        bounds=(-5.12, 5.12),
        global_optimum=0.0,
        modality="Highly Multimodal",
        smoothness="High",
        variable_interaction="Low",
        description="Many local optima, highly multimodal"
    ),
    "Ackley": BenchmarkFunction(
        name="Ackley",
        func=ackley,
        bounds=(-32.768, 32.768),
        global_optimum=0.0,
        modality="Multimodal",
        smoothness="Medium",
        variable_interaction="Medium",
        description="Many local optima with exponential terms"
    ),
    "Schwefel": BenchmarkFunction(
        name="Schwefel",
        func=schwefel,
        bounds=(-500.0, 500.0),
        global_optimum=0.0,
        modality="Multimodal",
        smoothness="Low",
        variable_interaction="Low",
        description="Deceptive global optimum far from local optima"
    )
}


def get_benchmark(name: str) -> BenchmarkFunction:
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}")
    return BENCHMARKS[name]


def list_benchmarks() -> list:
    return list(BENCHMARKS.keys())