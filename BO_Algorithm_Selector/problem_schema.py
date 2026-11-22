"""
problem_schema.py
Data structures and schemas for the Bayesian Optimization Algorithm Selection Platform.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ObjectiveType(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    UNKNOWN = "unknown"


class VariableType(str, Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    UNKNOWN = "unknown"


class SuitabilityLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    BASELINE = "Baseline"
    NOT_RECOMMENDED = "Not Recommended"
    UNKNOWN = "Unknown"


@dataclass
class VariableSchema:
    name: str
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    unit: str = ""
    var_type: VariableType = VariableType.CONTINUOUS
    categories: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "unit": self.unit,
            "type": self.var_type.value,
            "categories": self.categories or []
        }


@dataclass
class ObjectiveSchema:
    obj_type: ObjectiveType = ObjectiveType.MINIMIZE
    target: str = "Objective function"
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.obj_type.value,
            "target": self.target,
            "unit": self.unit
        }


@dataclass
class MissingFieldInfo:
    field_name: str
    display_name: str
    impact: str
    severity: str  # "warning", "info", "critical"
    suggested_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "display_name": self.display_name,
            "impact": self.impact,
            "severity": self.severity,
            "suggested_action": self.suggested_action
        }


@dataclass
class ProblemSchema:
    problem_description: str
    objective: ObjectiveSchema
    variables: List[VariableSchema] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    dimension: Optional[int] = None
    noise: Optional[float] = None  # None indicates Unknown - never fabricated
    evaluation_budget: Optional[int] = None
    evaluation_cost: Optional[str] = None
    expensive_evaluation: Optional[bool] = None
    known_optimum: Optional[float] = None
    domain: str = "General Engineering / Science"
    additional_information: str = ""
    missing_fields: List[MissingFieldInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_description": self.problem_description,
            "objective": self.objective.to_dict(),
            "variables": [v.to_dict() for v in self.variables],
            "constraints": self.constraints,
            "dimension": self.dimension,
            "noise": self.noise,
            "evaluation_budget": self.evaluation_budget,
            "evaluation_cost": self.evaluation_cost,
            "expensive_evaluation": self.expensive_evaluation,
            "known_optimum": self.known_optimum,
            "domain": self.domain,
            "additional_information": self.additional_information,
            "missing_fields": [m.to_dict() for m in self.missing_fields]
        }


@dataclass
class AlgorithmSuitability:
    algorithm: str
    suitability: SuitabilityLevel
    score: float  # Platform-defined score (0-100)
    key_reason: str
    detailed_explanation: str
    strengths_for_problem: List[str] = field(default_factory=list)
    drawbacks_for_problem: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "suitability": self.suitability.value,
            "score": self.score,
            "key_reason": self.key_reason,
            "detailed_explanation": self.detailed_explanation,
            "strengths": self.strengths_for_problem,
            "drawbacks": self.drawbacks_for_problem
        }


@dataclass
class RecommendationResult:
    recommended_algorithm: str
    suitability_level: SuitabilityLevel
    suitability_score: float
    confidence: str  # "High", "Medium", "Low"
    reasoning: List[str]
    limitations: List[str]
    suitability_table: List[AlgorithmSuitability]
    problem_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_algorithm": self.recommended_algorithm,
            "suitability_level": self.suitability_level.value,
            "suitability_score": self.suitability_score,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "limitations": self.limitations,
            "suitability_table": [a.to_dict() for a in self.suitability_table],
            "problem_summary": self.problem_summary
        }
