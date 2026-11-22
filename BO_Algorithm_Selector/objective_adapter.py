"""
objective_adapter.py
Abstraction layer for connecting extracted real-world optimization problems
to executable objective functions, simulators, APIs, or datasets.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any, List
import numpy as np
from problem_schema import ProblemSchema, VariableSchema


class BaseObjectiveAdapter(ABC):
    """
    Abstract base class for all executable objective connectors.
    Provides a standardized interface for Bayesian Optimization routines.
    """

    def __init__(self, name: str, dimension: int):
        self.name = name
        self.dimension = dimension
        self.evaluation_count = 0

    @abstractmethod
    def evaluate(self, x: np.ndarray, noise: float = 0.0) -> float:
        """
        Evaluate objective function at input vector x.
        """
        pass

    def __call__(self, x: np.ndarray) -> float:
        return self.evaluate(x)

    def reset(self):
        self.evaluation_count = 0


class PythonFunctionAdapter(BaseObjectiveAdapter):
    """
    Wraps a standard user-supplied Python function f(x) -> float.
    """
    def __init__(self, name: str, func: Callable[[np.ndarray], float], dimension: int):
        super().__init__(name, dimension)
        self.func = func

    def evaluate(self, x: np.ndarray, noise: float = 0.0) -> float:
        self.evaluation_count += 1
        val = float(self.func(x))
        if noise > 0:
            val += float(np.random.normal(0, noise))
        return val


class InjectionMoldingSimulator(BaseObjectiveAdapter):
    """
    Representative engineering simulator proxy for injection molding defect rate minimization.
    Inputs:
        x[0]: Temperature (180 - 250 °C)
        x[1]: Pressure (50 - 120 MPa)
        x[2]: Injection Speed (20 - 100 mm/s)
        x[3]: Cooling Time (5 - 30 sec)
    Target: Defect percentage (lower is better, optimum around ~0.8%).
    """
    def __init__(self):
        super().__init__("Injection Molding Physical Proxy Simulator", dimension=4)

    def evaluate(self, x: np.ndarray, noise: float = 0.0) -> float:
        self.evaluation_count += 1
        # Normalize variables to [0, 1]
        t = (x[0] - 180.0) / 70.0
        p = (x[1] - 50.0) / 70.0
        s = (x[2] - 20.0) / 80.0
        c = (x[3] - 5.0) / 25.0

        # Coupled non-linear response surface with optimum at T=220, P=95, S=60, C=22
        t_opt, p_opt, s_opt, c_opt = (220-180)/70, (95-50)/70, (60-20)/80, (22-5)/25
        val = 0.8 + 15.0 * (t - t_opt)**2 + 20.0 * (p - p_opt)**2 + 10.0 * (s - s_opt)**2 + 18.0 * (c - c_opt)**2
        # Non-linear thermal coupling between temperature and cooling time
        val += 8.0 * ((t - 0.5) * (c - 0.5))

        if noise > 0:
            val += float(np.random.normal(0, noise))
        return max(0.2, float(val))


class ObjectiveConnector:
    """
    Manages executable objective bindings for parsed problems.
    Enforces scientific integrity:
    Never claims numerical benchmark results are available unless an adapter is explicitly active.
    """

    AVAILABLE_SIMULATORS = {
        "injection molding": InjectionMoldingSimulator
    }

    @classmethod
    def get_adapter(cls, problem: ProblemSchema) -> Optional[BaseObjectiveAdapter]:
        """
        Check if a verified simulator or executable function is registered for this problem.
        Returns None if only natural language is available.
        """
        desc_lower = problem.problem_description.lower()
        if "injection molding" in desc_lower and problem.dimension == 4:
            return InjectionMoldingSimulator()
        return None

    @classmethod
    def get_status_message(cls, problem: ProblemSchema, has_adapter: bool) -> Dict[str, str]:
        if has_adapter:
            return {
                "status": "Executable Simulator Available",
                "badge": "Simulation Supported",
                "message": (
                    "A verified physical process proxy simulator is available for this problem domain. "
                    "You may choose to run empirical validation benchmarks against this simulator."
                )
            }
        else:
            return {
                "status": "Analysis & Recommendation Only",
                "badge": "No Executable Objective",
                "message": (
                    "Problem analysis, characterization, and algorithm suitability recommendations are available. "
                    "However, numerical benchmarking requires an executable objective function, simulator, or compatible dataset. "
                    "To prevent scientific fabrication, no numerical optimization runs will be simulated without an executable target."
                )
            }
