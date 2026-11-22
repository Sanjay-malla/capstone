"""
problem_parser.py
Natural Language Problem Parser for Bayesian Optimization Algorithm Selection Platform.

Extracts structured problem specifications from unstructured natural-language descriptions
without fabricating missing parameters.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from problem_schema import (
    ProblemSchema,
    VariableSchema,
    ObjectiveSchema,
    ObjectiveType,
    VariableType,
    MissingFieldInfo
)


class ProblemParser:
    """
    Rule-based & pattern-based natural language parser designed for engineering
    and scientific optimization problem descriptions.
    """

    # Domain indicators
    DOMAIN_PATTERNS = {
        "Manufacturing / Industrial": [
            r"molding", r"machin", r"manufactur", r"casting", r"weld", r"production line",
            r"injection", r"extru", r"fabricat", r"tooling"
        ],
        "Machine Learning / AI": [
            r"machine learning", r"model", r"hyperparameter", r"learning rate",
            r"batch size", r"accuracy", r"loss", r"dropout", r"neural network",
            r"validation", r"epochs", r"layers", r"transformer"
        ],
        "Robotics & Control": [
            r"robot", r"arm", r"actuator", r"trajectory", r"motor", r"positioning",
            r"kinemat", r"control", r"joint", r"velocity", r"torque"
        ],
        "Energy & HVAC": [
            r"hvac", r"chiller", r"cooling", r"heating", r"energy consumption",
            r"airflow", r"thermostat", r"cfm", r"building", r"power consumption"
        ],
        "Automotive & Aerospace": [
            r"battery", r"electric vehicle", r"ev", r"aerodynamic", r"drag",
            r"range", r"engine", r"propulsion", r"coolant"
        ],
        "Chemical & Materials": [
            r"chemical", r"catalyst", r"reaction", r"yield", r"solvent",
            r"concentration", r"synthesis", r"ph", r"compound"
        ]
    }

    # Integer variable name hints
    INTEGER_HINTS = [
        r"layers", r"batch\s*size", r"epochs", r"units", r"count", r"nodes",
        r"steps", r"number\s+of", r"trees", r"depth", r"iterations"
    ]

    def __init__(self):
        pass

    def parse(self, text: str) -> ProblemSchema:
        """
        Main entrypoint: parses natural language text into a structured ProblemSchema.
        Never invents missing information.
        """
        clean_text = text.strip()

        # 1. Parse Objective
        objective = self._extract_objective(clean_text)

        # 2. Parse Variables & Bounds
        variables = self._extract_variables(clean_text)

        # 3. Parse Constraints
        constraints = self._extract_constraints(clean_text)

        # 4. Parse Evaluation Budget
        budget = self._extract_budget(clean_text)

        # 5. Parse Evaluation Cost
        cost_str, expensive_eval = self._extract_cost_and_expensiveness(clean_text, budget)

        # 6. Parse Noise Level (DO NOT FABRICATE)
        noise = self._extract_noise(clean_text)

        # 7. Known Optimum (DO NOT FABRICATE)
        known_optimum = self._extract_known_optimum(clean_text)

        # 8. Domain
        domain = self._detect_domain(clean_text)

        # 9. Dimension
        dimension = len(variables) if variables else None

        # 10. Identify Missing Information
        missing_fields = self._identify_missing_fields(
            variables=variables,
            noise=noise,
            budget=budget,
            cost_str=cost_str,
            dimension=dimension,
            constraints=constraints
        )

        return ProblemSchema(
            problem_description=clean_text,
            objective=objective,
            variables=variables,
            constraints=constraints,
            dimension=dimension,
            noise=noise,
            evaluation_budget=budget,
            evaluation_cost=cost_str,
            expensive_evaluation=expensive_eval,
            known_optimum=known_optimum,
            domain=domain,
            additional_information="",
            missing_fields=missing_fields
        )

    def _extract_objective(self, text: str) -> ObjectiveSchema:
        """Detect optimization direction (minimize/maximize) and target objective."""
        text_lower = text.lower()

        # Direction detection with word stems matching word boundaries
        min_match = re.search(r"\b(minimiz\w*|reduc\w*|decreas\w*|lower\w*|mitigat\w*|cut down|smallest|minimal)\b", text_lower)
        max_match = re.search(r"\b(maximiz\w*|increas\w*|boost\w*|highest|improv\w*|enhanc\w*|greatest|grow\w*)\b", text_lower)

        is_minimize = bool(min_match)
        is_maximize = bool(max_match)

        obj_type = ObjectiveType.MINIMIZE
        if is_maximize and not is_minimize:
            obj_type = ObjectiveType.MAXIMIZE
        elif is_minimize and not is_maximize:
            obj_type = ObjectiveType.MINIMIZE
        elif is_maximize and is_minimize:
            # Check which occurs first in text
            if min_match and max_match:
                obj_type = ObjectiveType.MINIMIZE if min_match.start() < max_match.start() else ObjectiveType.MAXIMIZE

        # Target detection
        target = "Objective function"
        unit = ""

        # Common target patterns
        target_patterns = [
            # "want to maximize the validation accuracy of a machine learning model"
            r"(?:want to|aim to|goal is to|need to|wish to)\s+(?:minimiz\w*|maximiz\w*|reduc\w*|increas\w*|improv\w*)\s+(?:the\s+)?([a-zA-Z\s]{3,65}?)(?:\.|\,|while|by|where|with|the\s+[a-zA-Z]+\s+can|\bfrom\b|\bfor\b|\bI\b)",
            # "minimize the percentage of defective parts"
            r"(?:minimiz\w*|maximiz\w*|reduc\w*|increas\w*|improv\w*)\s+(?:the\s+)?([a-zA-Z\s]{3,65}?)(?:\.|\,|while|by|where|with|the\s+[a-zA-Z]+\s+can|\bfrom\b|\bfor\b|\bI\b)"
        ]

        for pat in target_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                # filter out noisy tails
                cand = re.sub(r"\s+(range|from|can|between|within)$", "", cand, flags=re.IGNORECASE).strip()
                if len(cand) > 2 and cand.lower() not in ["it", "them", "this", "model", "process"]:
                    target = cand
                    break

        target_lower = target.lower()
        if "percentage" in target_lower or "defect" in target_lower or "rate" in target_lower or "accuracy" in target_lower:
            unit = "%"
        elif "time" in target_lower or "duration" in target_lower:
            unit = "sec / min"
        elif "cost" in target_lower or "expense" in target_lower:
            unit = "currency"
        elif "energy" in target_lower or "power" in target_lower:
            unit = "kWh / W"
        elif "range" in target_lower or "distance" in target_lower:
            unit = "km / miles"

        return ObjectiveSchema(
            obj_type=obj_type,
            target=target,
            unit=unit
        )

    def _extract_variables(self, text: str) -> List[VariableSchema]:
        """
        Extract variable names, bounds, units, and data types.
        Handles ranges like:
        - "injection temperature can range from 180 to 250 degrees Celsius"
        - "pressure from 50 to 120 MPa"
        - "cooling time from 5 to 30 seconds"
        - "learning rate from 0.0001 to 0.1"
        - "batch size from 16 to 256"
        - "temperature: 180–250 °C" or "180 - 250 °C"
        - list without bounds: "modify motor speed, acceleration, and trajectory parameters"
        """
        variables: List[VariableSchema] = []
        found_names = set()

        # Pattern 1: <var_name> (can range|ranges|from|is|between) <num1> (to|-|–|and) <num2> [unit]
        # Example: "injection temperature can range from 180 to 250 degrees Celsius"
        # Example: "pressure from 50 to 120 MPa"
        # Example: "cooling time from 5 to 30 seconds"
        pattern_range = re.compile(
            r"(?:(?:the|and|,)\s+)?([a-zA-Z][a-zA-Z0-9_\s]{1,30}?)\s*"
            r"(?:can range|ranges|range|is|are|between)?\s*"
            r"(?:from|between|:|=)\s*"
            r"([+-]?\d+(?:\.\d+)?)\s*"
            r"(?:to|–|-|and)\s*"
            r"([+-]?\d+(?:\.\d+)?)\s*"
            r"([°a-zA-Z/%³²\s]{0,20}?)"
            r"(?=[,\.;\n]|and\s+[a-zA-Z]|\s*$)",
            re.IGNORECASE
        )

        # Pattern 2: <var_name>\s*[:=\(]\s*([+-]?\d+(?:\.\d+)?)\s*(?:to|-|–)\s*([+-]?\d+(?:\.\d+)?)\s*([°a-zA-Z/%³²\s]*?)[\)\],;\.]
        pattern_compact = re.compile(
            r"([a-zA-Z][a-zA-Z0-9_\s]{1,30}?)\s*"
            r"[:=\(]\s*"
            r"([+-]?\d+(?:\.\d+)?)\s*(?:to|–|-)\s*([+-]?\d+(?:\.\d+)?)\s*"
            r"([°a-zA-Z/%³²\s]*?)[\)\],;\.]",
            re.IGNORECASE
        )

        # Check Pattern 1 matches
        for m in pattern_range.finditer(text):
            raw_name, raw_lb, raw_ub, raw_unit = m.groups()
            name = self._clean_variable_name(raw_name)
            if not name or self._is_blacklisted_name(name):
                continue

            try:
                lb = float(raw_lb)
                ub = float(raw_ub)
                unit = self._clean_unit(raw_unit)
                var_type = self._infer_variable_type(name, lb, ub)

                if name.lower() not in found_names:
                    variables.append(VariableSchema(
                        name=name,
                        lower_bound=min(lb, ub),
                        upper_bound=max(lb, ub),
                        unit=unit,
                        var_type=var_type
                    ))
                    found_names.add(name.lower())
            except ValueError:
                continue

        # Check Pattern 2 matches for any not already captured
        for m in pattern_compact.finditer(text):
            raw_name, raw_lb, raw_ub, raw_unit = m.groups()
            name = self._clean_variable_name(raw_name)
            if not name or self._is_blacklisted_name(name) or name.lower() in found_names:
                continue

            try:
                lb = float(raw_lb)
                ub = float(raw_ub)
                unit = self._clean_unit(raw_unit)
                var_type = self._infer_variable_type(name, lb, ub)

                variables.append(VariableSchema(
                    name=name,
                    lower_bound=min(lb, ub),
                    upper_bound=max(lb, ub),
                    unit=unit,
                    var_type=var_type
                ))
                found_names.add(name.lower())
            except ValueError:
                continue

        # If no bounded variables found, or for cases like the Robot Arm example:
        # "I can modify motor speed, acceleration, and trajectory parameters within predefined operating ranges."
        if not variables:
            # Look for variables explicitly listed without bounds
            param_list_patterns = [
                r"(?:modify|tune|control|adjust|optimize|vary|parameters?(?: include)?|factors?|inputs?)\s+([a-zA-Z0-9_,\s\(\)]+?)(?:within|\.|while|for|\band\b\s+each|\beach)",
            ]
            for pat in param_list_patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    raw_items = m.group(1)
                    # Split on comma or "and"
                    parts = re.split(r",|\band\b", raw_items)
                    for p in parts:
                        clean_p = self._clean_variable_name(p)
                        if clean_p and not self._is_blacklisted_name(clean_p) and clean_p.lower() not in found_names:
                            variables.append(VariableSchema(
                                name=clean_p,
                                lower_bound=None,  # Unknown bounds! Do NOT invent
                                upper_bound=None,
                                unit="",
                                var_type=VariableType.CONTINUOUS
                            ))
                            found_names.add(clean_p.lower())

        return variables

    def _clean_variable_name(self, raw_name: str) -> str:
        """Strip filler words from extracted variable names."""
        name = raw_name.strip()
        # Strip leading phrases repeatedly
        for _ in range(3):
            name = re.sub(
                r"^(?:i\s+can\s+(?:tune|modify|adjust|vary|change|set)|we\s+can\s+(?:tune|modify|adjust|vary)|"
                r"i\s+can|we\s+can|tune|modify|adjust|vary|the|a|an|and|or|can|range|from|with|parameters?|variables?)\s+",
                "", name, flags=re.IGNORECASE
            ).strip()
        name = re.sub(r"\s+(can|from|is|range|ranges|between|within)$", "", name, flags=re.IGNORECASE)
        name = re.sub(r"[^\w\s-]", "", name).strip()
        # Capitalize nicely
        if name:
            words = name.split()
            name = " ".join([w.capitalize() if w.lower() not in ["of", "in", "and", "to", "per"] else w for w in words])
        return name

    def _is_blacklisted_name(self, name: str) -> bool:
        """Filter out non-variable matches such as 'each experiment', 'budget', 'accuracy', etc."""
        name_lower = name.lower()
        blacklist = [
            "experiment", "experiments", "cost", "costs", "budget", "approximately",
            "percent", "percentage", "accuracy", "rate", "time", "hour", "minute",
            "second", "machine", "operating", "predefined", "range", "ranges"
        ]
        if name_lower in blacklist:
            return True
        if len(name_lower) < 2:
            return True
        return False

    def _clean_unit(self, raw_unit: str) -> str:
        """Clean and normalize physical unit strings."""
        if not raw_unit:
            return ""
        unit = raw_unit.strip()
        # Remove trailing words that got captured
        unit = re.split(r"\s+(?:and|each|cost|costs|with|where|while|I|we|for)\b", unit, flags=re.IGNORECASE)[0]
        unit = unit.strip(",.; ")

        # Map common unit phrases to standard symbols
        unit_map = {
            "degrees celsius": "°C",
            "celsius": "°C",
            "degree celsius": "°C",
            "deg c": "°C",
            "degrees fahrenheit": "°F",
            "seconds": "sec",
            "second": "sec",
            "minutes": "min",
            "minute": "min",
            "hours": "hr",
            "hour": "hr",
            "megapascals": "MPa",
            "megapascal": "MPa",
            "mpa": "MPa",
            "millimeters per second": "mm/s",
            "mm per s": "mm/s",
            "mm/s": "mm/s",
            "percent": "%",
            "percentage": "%"
        }
        return unit_map.get(unit.lower(), unit)

    def _infer_variable_type(self, name: str, lb: float, ub: float) -> VariableType:
        """Infer whether a variable is integer or continuous."""
        name_lower = name.lower()
        for hint in self.INTEGER_HINTS:
            if re.search(hint, name_lower):
                return VariableType.INTEGER

        # If bounds are whole numbers and name hints discrete
        if lb.is_integer() and ub.is_integer():
            if any(term in name_lower for term in ["layers", "batch", "epoch", "count", "steps", "units", "trees"]):
                return VariableType.INTEGER

        return VariableType.CONTINUOUS

    def _extract_constraints(self, text: str) -> List[str]:
        """Detect operational or quality constraints in the text."""
        constraints = []
        patterns = [
            # "maintaining positioning accuracy above 99%"
            r"(?:maintain\w*|keep\w*|ensure\w*|with|subject to|where)\s+([a-zA-Z\s]{3,35}?(?:above|below|at least|no more than|greater than|less than|>=|<=|>|<)\s*[\d\.]+%?)",
            # "accuracy > 99%"
            r"([a-zA-Z\s]{3,25}?\s*(?:>=|<=|>|<)\s*[\d\.]+%?)"
        ]

        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                c = m.group(1).strip()
                if c and c not in constraints and len(c) > 4:
                    constraints.append(c)

        return constraints

    def _extract_budget(self, text: str) -> Optional[int]:
        """Extract evaluation budget (e.g. 100 experiments, 80 evaluations)."""
        patterns = [
            r"(?:can perform|perform|budget of|budget is|run|execute|afford|limit(?:ed)? to)\s*[:=]?\s*(\d+)\s*(?:experiments|evaluations|trials|runs|iterations|tests|samples)?",
            r"(\d+)\s*(?:experiments|evaluations|trials|runs|tests|iterations)\s*(?:total|budget|in total|can be performed|allowed)",
            r"budget\s*[:=]?\s*(\d+)"
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    val = int(m.group(1))
                    if 1 <= val <= 10000:
                        return val
                except ValueError:
                    continue
        return None

    def _extract_cost_and_expensiveness(self, text: str, budget: Optional[int]) -> Tuple[Optional[str], Optional[bool]]:
        """
        Extract cost per evaluation and determine if evaluations are considered expensive.
        Never fabricates numerical values.
        """
        cost_str = None
        is_expensive = None

        # Currency costs
        currency_patterns = [
            r"((?:₹|\$|€|£|Rs\.?|INR|USD)\s*[\d,]+(?:\.\d+)?)\s*(?:per|each|\/)?\s*(?:experiment|evaluation|run|trial|sample)?",
            r"(?:costs?|costing|priced at)\s*(?:approximately|around|about)?\s*((?:₹|\$|€|£|Rs\.?|INR|USD)\s*[\d,]+(?:\.\d+)?)",
            r"(?:costs?|costing)\s*(?:approximately|around|about)?\s*([\d,]+(?:\.\d+)?\s*(?:rupees|dollars|euros|inr|usd))"
        ]

        for pat in currency_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                cost_str = m.group(1).strip()
                is_expensive = True
                break

        # Time-based costs
        time_patterns = [
            r"(?:takes|duration|time per (?:run|experiment) is|requires)\s*(?:around|about|approximately)?\s*(\d+\s*(?:minutes|mins|hours|hrs|days|weeks))",
            r"(?:simulation|experiment|run|training)\s+takes\s+(several minutes|hours|days|\d+\s*(?:minutes|mins|hours|hrs))"
        ]
        for pat in time_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                if not cost_str:
                    cost_str = f"~{m.group(1).strip()} per evaluation"
                is_expensive = True
                break

        # Keyword based expensiveness
        if is_expensive is None:
            if re.search(r"\b(expensive|time-consuming|computationally heavy|physical experiment|destructive testing)\b", text, re.IGNORECASE):
                is_expensive = True
            elif budget is not None and budget <= 150:
                is_expensive = True

        return cost_str, is_expensive

    def _extract_noise(self, text: str) -> Optional[float]:
        """
        Extract explicit noise level.
        CRITICAL RULE: If not explicitly specified, return None (Unknown).
        NEVER invent noise.
        """
        patterns = [
            r"noise\s*(?:level|std|sigma)?\s*[:=]?\s*([0-1](?:\.\d+)?)",
            r"standard deviation\s*(?:of)?\s*([0-1](?:\.\d+)?)",
            r"noise\s*=\s*([0-1](?:\.\d+)?)"
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
        return None

    def _extract_known_optimum(self, text: str) -> Optional[float]:
        """
        Extract known optimum if explicitly stated.
        CRITICAL RULE: If not explicitly provided, return None (Unknown).
        """
        patterns = [
            r"(?:known|true|global)\s+optimum\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)",
            r"target\s*optimum\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)"
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
        return None

    def _detect_domain(self, text: str) -> str:
        """Infer real-world domain from problem context."""
        text_lower = text.lower()
        domain_scores = {}
        for domain, keywords in self.DOMAIN_PATTERNS.items():
            score = 0
            for kw in keywords:
                if re.search(r"\b" + kw, text_lower):
                    score += 1
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return "General Engineering / Science"

    def _identify_missing_fields(
        self,
        variables: List[VariableSchema],
        noise: Optional[float],
        budget: Optional[int],
        cost_str: Optional[str],
        dimension: Optional[int],
        constraints: List[str]
    ) -> List[MissingFieldInfo]:
        """
        Examine extracted data and identify missing fields with transparent explanations.
        Never silently gloss over missing parameters.
        """
        missing: List[MissingFieldInfo] = []

        # 1. Variables missing bounds
        vars_without_bounds = [v.name for v in variables if v.lower_bound is None or v.upper_bound is None]
        if vars_without_bounds:
            missing.append(MissingFieldInfo(
                field_name="variable_bounds",
                display_name=f"Variable Bounds ({', '.join(vars_without_bounds)})",
                impact="Without explicit lower and upper bounds, Bayesian Optimization cannot construct the bounded search space [lb, ub]^d.",
                severity="warning",
                suggested_action="Specify valid operational ranges [min, max] for each variable."
            ))

        # 2. No variables detected at all
        if not variables:
            missing.append(MissingFieldInfo(
                field_name="variables",
                display_name="Optimization Variables",
                impact="No parameters or decision variables were detected in the description.",
                severity="critical",
                suggested_action="List the specific parameters you wish to tune with their names and ranges."
            ))

        # 3. Noise level missing
        if noise is None:
            missing.append(MissingFieldInfo(
                field_name="noise",
                display_name="Observation Noise Level",
                impact="Noise information determines surrogate model kernel regularization (Gaussian Process alpha/nugget). Without noise data, standard noiseless or default small nugget assumptions must be applied cautiously.",
                severity="info",
                suggested_action="Specify whether measurements are deterministic, low noise, moderate noise, or noisy."
            ))

        # 4. Evaluation budget missing
        if budget is None:
            missing.append(MissingFieldInfo(
                field_name="evaluation_budget",
                display_name="Evaluation Budget",
                impact="The total number of allowed evaluations guides algorithm selection: small budgets (< 100) favor standard sample-efficient GP-EI/UCB, while larger budgets (> 200) support scalable trust-region methods (TuRBO).",
                severity="warning",
                suggested_action="Provide the maximum number of experiments or function evaluations you can afford."
            ))

        # 5. Evaluation cost missing
        if cost_str is None:
            missing.append(MissingFieldInfo(
                field_name="evaluation_cost",
                display_name="Evaluation Cost / Duration",
                impact="Cost data helps assess whether the surrogate model computational overhead (1-5 seconds per iteration) is justified compared to the cost of running real experiments.",
                severity="info",
                suggested_action="Optionally specify monetary cost or time duration per experiment."
            ))

        return missing
