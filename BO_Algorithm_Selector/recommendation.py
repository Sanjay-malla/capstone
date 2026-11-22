"""
recommendation.py
Algorithm Suitability Evaluation and Recommendation Engine for the Bayesian Optimization Platform.

Supports:
1. Mode A: Analytical suitability scoring and evidence-based recommendation for real-world natural language problems.
2. Mode B: Empirical recommendation generation from executed benchmark experiments.
"""

from typing import Dict, Any, List, Optional
from problem_schema import (
    ProblemSchema,
    AlgorithmSuitability,
    SuitabilityLevel,
    RecommendationResult,
    VariableType
)
from characterization import ProblemCharacterization


def evaluate_algorithm_suitability(
    char: ProblemCharacterization,
    problem: ProblemSchema
) -> RecommendationResult:
    """
    Evaluates suitability of GP-EI, GP-UCB, REMBO, TuRBO, and Random Search
    based on multidimensional problem characteristics:
    - Dimensionality
    - Search space / variable types
    - Evaluation budget
    - Evaluation cost & expensiveness
    - Observation noise
    - Operational constraints

    Does NOT use simplistic lookup tables. Uses principled Bayesian Optimization principles.
    Scores are explicitly labeled as platform-defined suitability scores.
    """
    dim = char.dimensionality if char.dimensionality is not None else 4
    budget = char.budget if char.budget is not None else 100
    is_expensive = problem.expensive_evaluation if problem.expensive_evaluation is not None else True
    has_integer = any(v.var_type == VariableType.INTEGER for v in problem.variables)
    has_constraints = bool(problem.constraints)
    noise_val = problem.noise

    suitabilities: List[AlgorithmSuitability] = []

    # -------------------------------------------------------------
    # 1. GP-EI (Gaussian Process with Expected Improvement)
    # -------------------------------------------------------------
    gp_ei_strengths = []
    gp_ei_drawbacks = []
    gp_ei_score = 50.0

    if dim <= 8:
        gp_ei_score += 35.0
        gp_ei_strengths.append(f"Ideal dimensionality (d={dim} <= 8) where global GP covariance modeling is exact and computationally tractable.")
    elif dim <= 15:
        gp_ei_score += 20.0
        gp_ei_strengths.append(f"Moderate low dimensionality (d={dim}); standard GP modeling remains effective.")
    elif dim <= 30:
        gp_ei_score -= 10.0
        gp_ei_drawbacks.append(f"Moderate dimension (d={dim}); GP distance metric concentration begins to degrade acquisition precision.")
    else:
        gp_ei_score -= 30.0
        gp_ei_drawbacks.append(f"High dimensionality (d={dim} > 30); standard GP cubic complexity and empty-space phenomenon degrade performance.")

    if is_expensive:
        gp_ei_score += 10.0
        gp_ei_strengths.append("High sample efficiency: Expected Improvement extracts maximum information from expensive trials.")

    if budget <= 120:
        gp_ei_score += 5.0
        gp_ei_strengths.append(f"Well-suited for compact budgets (N={budget}) where global exploration is critical early on.")
    else:
        gp_ei_drawbacks.append(f"At larger budgets (N={budget}), O(N^3) kernel inversion creates noticeable computational overhead per step.")

    if has_integer:
        gp_ei_score -= 5.0
        gp_ei_drawbacks.append("Discrete/integer variables require continuous relaxation or rounding approximations.")

    gp_ei_score = max(10.0, min(95.0, gp_ei_score))
    if gp_ei_score >= 75:
        gp_ei_level = SuitabilityLevel.HIGH
        gp_ei_key = "Strong fit for low-dimensional expensive black-box optimization."
    elif gp_ei_score >= 50:
        gp_ei_level = SuitabilityLevel.MEDIUM
        gp_ei_key = "Viable approach, though scaling limits or variable relaxations apply."
    else:
        gp_ei_level = SuitabilityLevel.LOW
        gp_ei_key = "Limited suitability due to high dimensionality or search space characteristics."

    suitabilities.append(AlgorithmSuitability(
        algorithm="GP-EI",
        suitability=gp_ei_level,
        score=round(gp_ei_score, 1),
        key_reason=gp_ei_key,
        detailed_explanation=(
            "Gaussian Process with Expected Improvement (GP-EI) constructs a non-parametric surrogate "
            "over the objective function, computing the analytic expectation of improvement over the current best value. "
            "It offers exemplary sample efficiency in low dimensions (d <= 15) when evaluations are costly, "
            "though its global acquisition optimization scales poorly as dimensionality grows."
        ),
        strengths_for_problem=gp_ei_strengths,
        drawbacks_for_problem=gp_ei_drawbacks
    ))

    # -------------------------------------------------------------
    # 2. GP-UCB (Gaussian Process with Upper Confidence Bound)
    # -------------------------------------------------------------
    gp_ucb_strengths = []
    gp_ucb_drawbacks = []
    gp_ucb_score = 48.0

    if dim <= 8:
        gp_ucb_score += 33.0
        gp_ucb_strengths.append(f"Effective in low dimensions (d={dim}) with rigorous sub-linear regret bounds.")
    elif dim <= 15:
        gp_ucb_score += 18.0
        gp_ucb_strengths.append(f"Good sample efficiency in moderate-low dimensional spaces (d={dim}).")
    elif dim <= 30:
        gp_ucb_score -= 10.0
        gp_ucb_drawbacks.append(f"Dimension (d={dim}) induces volume dispersion across the search space.")
    else:
        gp_ucb_score -= 30.0
        gp_ucb_drawbacks.append(f"High dimension (d={dim}) limits global confidence bound exploration.")

    if noise_val is not None and noise_val > 0.05:
        gp_ucb_score += 8.0
        gp_ucb_strengths.append("UCB beta parameter allows principled tuning of exploration under noisy conditions.")

    if is_expensive:
        gp_ucb_score += 8.0
        gp_ucb_strengths.append("High sample efficiency saves expensive experimental evaluations.")

    if has_integer:
        gp_ucb_score -= 5.0
        gp_ucb_drawbacks.append("Discrete variables require continuous mapping approximations.")

    gp_ucb_score = max(10.0, min(95.0, gp_ucb_score))
    if gp_ucb_score >= 75:
        gp_ucb_level = SuitabilityLevel.HIGH
        gp_ucb_key = "Effective low-dimensional surrogate with adjustable exploration guarantees."
    elif gp_ucb_score >= 50:
        gp_ucb_level = SuitabilityLevel.MEDIUM
        gp_ucb_key = "Moderate suitability; balances mean and uncertainty with theoretical regret bounds."
    else:
        gp_ucb_level = SuitabilityLevel.LOW
        gp_ucb_key = "Sub-optimal in high dimensions where uncertainty estimates become uninformative."

    suitabilities.append(AlgorithmSuitability(
        algorithm="GP-UCB",
        suitability=gp_ucb_level,
        score=round(gp_ucb_score, 1),
        key_reason=gp_ucb_key,
        detailed_explanation=(
            "GP-UCB uses an optimistic upper confidence bound acquisition function (mu(x) + beta * sigma(x)). "
            "It is mathematically proven to achieve sub-linear cumulative regret in reproducing kernel Hilbert spaces. "
            "It excels when explicit control over exploration-exploitation trade-offs is desired, especially under noise."
        ),
        strengths_for_problem=gp_ucb_strengths,
        drawbacks_for_problem=gp_ucb_drawbacks
    ))

    # -------------------------------------------------------------
    # 3. REMBO (Random Embedding Bayesian Optimization)
    # -------------------------------------------------------------
    rembo_strengths = []
    rembo_drawbacks = []
    rembo_score = 30.0

    if dim >= 25:
        rembo_score += 45.0
        rembo_strengths.append(f"Explicitly architected for high dimensions (d={dim} >= 25) with low effective subspace.")
    elif dim >= 12:
        rembo_score += 20.0
        rembo_strengths.append(f"Can compress moderate dimension (d={dim}) into a lower-dimensional latent space.")
    else:
        rembo_score -= 20.0
        rembo_drawbacks.append(f"Overkill for low dimensions (d={dim}); random projection introduces boundary distortion without benefits.")

    if has_integer:
        rembo_score -= 15.0
        rembo_drawbacks.append("Random continuous projection matrix A is poorly suited for discrete/integer structures.")

    if has_constraints:
        rembo_score -= 10.0
        rembo_drawbacks.append("Boundary clipping and constraint handling in the embedded subspace can be challenging.")

    rembo_score = max(5.0, min(90.0, rembo_score))
    if rembo_score >= 70:
        rembo_level = SuitabilityLevel.HIGH
        rembo_key = "Top candidate for high-dimensional problems with suspected low effective dimension."
    elif rembo_score >= 45:
        rembo_level = SuitabilityLevel.MEDIUM
        rembo_key = "Viable if intrinsic dimensionality is suspected to be much lower than the full parameter count."
    else:
        rembo_level = SuitabilityLevel.LOW
        rembo_key = "Not recommended for low-dimensional problems due to projection artifacts."

    suitabilities.append(AlgorithmSuitability(
        algorithm="REMBO",
        suitability=rembo_level,
        score=round(rembo_score, 1),
        key_reason=rembo_key,
        detailed_explanation=(
            "Random Embedding Bayesian Optimization (REMBO) addresses the curse of dimensionality by projecting "
            "a low-dimensional latent space z in R^d_e into the high-dimensional ambient space x in R^D via a random Gaussian matrix A. "
            "It is theoretically guaranteed to locate the optimum if the objective has an effective subspace d_e << D, "
            "but introduces severe boundary clipping inefficiencies when applied to small or discrete parameter spaces."
        ),
        strengths_for_problem=rembo_strengths,
        drawbacks_for_problem=rembo_drawbacks
    ))

    # -------------------------------------------------------------
    # 4. TuRBO (Trust Region Bayesian Optimization)
    # -------------------------------------------------------------
    turbo_strengths = []
    turbo_drawbacks = []
    turbo_score = 65.0  # TuRBO is versatile across both moderate and higher dimensions

    if dim <= 4:
        turbo_score += 15.0
        turbo_strengths.append(f"Reliable local exploration in compact spaces (d={dim}); adjusts trust region adaptively.")
    elif dim <= 20:
        turbo_score += 25.0
        turbo_strengths.append(f"Highly effective in 5-20 dimensions, balancing local exploitation without over-exploring.")
    elif dim <= 100:
        turbo_score += 30.0
        turbo_strengths.append(f"Proven state-of-the-art scalability for high-dimensional BO (d={dim}) via local trust regions.")

    if budget >= 80:
        turbo_score += 10.0
        turbo_strengths.append(f"Budget of {budget} evaluations allows trust-region expansion and contraction mechanisms to fully converge.")
    else:
        turbo_score += 2.0

    if is_expensive:
        turbo_strengths.append("High sample efficiency within active trust regions prevents wasting expensive evaluations.")

    turbo_score = max(20.0, min(98.0, turbo_score))
    if turbo_score >= 75:
        turbo_level = SuitabilityLevel.HIGH
        turbo_key = "Outstanding scalable trust-region BO for moderate-to-high dimensions and complex landscapes."
    elif turbo_score >= 50:
        turbo_level = SuitabilityLevel.MEDIUM
        turbo_key = "Good candidate; local trust regions adapt to feedback dynamically."
    else:
        turbo_level = SuitabilityLevel.LOW
        turbo_key = "Sub-optimal for simple unimodal spaces with ultra-low budgets (< 15 evaluations)."

    suitabilities.append(AlgorithmSuitability(
        algorithm="TuRBO",
        suitability=turbo_level,
        score=round(turbo_score, 1),
        key_reason=turbo_key,
        detailed_explanation=(
            "Trust Region Bayesian Optimization (TuRBO) fits local Gaussian Process models inside dynamic bounding boxes "
            "(trust regions) centered around the best observed point. By expanding the region after consecutive improvements "
            "and shrinking it after repeated failures, TuRBO avoids the classic pathology where global GPs over-explore "
            "uninformative boundary regions in high dimensions."
        ),
        strengths_for_problem=turbo_strengths,
        drawbacks_for_problem=turbo_drawbacks
    ))

    # -------------------------------------------------------------
    # 5. Random Search (Baseline)
    # -------------------------------------------------------------
    rs_strengths = ["Zero computational overhead: no GP fitting or acquisition optimization."]
    rs_drawbacks = [
        "Lacks learning mechanisms: does not exploit past evaluation history.",
        "Very poor sample efficiency: unacceptable when each evaluation carries real financial or time cost."
    ]
    rs_score = 25.0
    if not is_expensive:
        rs_score += 10.0
        rs_strengths.append("Acceptable as a preliminary fast exploration when evaluations are virtually free.")

    suitabilities.append(AlgorithmSuitability(
        algorithm="Random Search",
        suitability=SuitabilityLevel.BASELINE,
        score=rs_score,
        key_reason="Essential scientific baseline; lacks surrogate learning and sample efficiency.",
        detailed_explanation=(
            "Random Search samples uniformly from the hypercube without building a predictive model. "
            "While computationally instantaneous, it requires asymptotically exponential samples to locate narrow optima "
            "and should only serve as a benchmark baseline."
        ),
        strengths_for_problem=rs_strengths,
        drawbacks_for_problem=rs_drawbacks
    ))

    # Sort algorithms by score (excluding Random Search which is baseline)
    active_algos = [s for s in suitabilities if s.algorithm != "Random Search"]
    active_algos.sort(key=lambda s: s.score, reverse=True)
    top_algo = active_algos[0]

    # Synthesize evidence-based reasoning
    reasoning: List[str] = []
    reasoning.append(f"Problem dimensionality is d = {dim}, categorized as {char.dimension_category.lower()}.")
    reasoning.append(f"Search space consists of {char.variable_types_summary.lower()} variables.")

    if is_expensive:
        cost_mention = f" ({problem.evaluation_cost})" if problem.evaluation_cost else ""
        reasoning.append(f"Evaluations are characterized as expensive{cost_mention}, making sample-efficient Bayesian Optimization mandatory.")

    if budget:
        reasoning.append(f"Evaluation budget of {budget} trials aligns well with {top_algo.algorithm}'s convergence trajectory.")

    for strength in top_algo.strengths_for_problem:
        reasoning.append(strength)

    # Scientific limitations (MUST BE TRANSPARENT)
    limitations: List[str] = [
        "Recommendation is derived purely from problem characteristics extracted from natural language text.",
        "Because no executable simulator, Python objective function, or dataset was provided, empirical performance has NOT been validated numerically.",
        "Unknown landscape characteristics (such as multimodal barriers, sharp ridges, or hidden discontinuities) may favor alternative acquisition strategies."
    ]
    if problem.noise is None:
        limitations.append("Measurement noise was not specified in the input; standard default noise assumptions were applied.")

    confidence = "High" if (char.dimensionality is not None and char.budget is not None) else "Medium"

    return RecommendationResult(
        recommended_algorithm=top_algo.algorithm,
        suitability_level=top_algo.suitability,
        suitability_score=top_algo.score,
        confidence=confidence,
        reasoning=reasoning,
        limitations=limitations,
        suitability_table=suitabilities,
        problem_summary=char.to_dict()
    )


def generate_recommendation(
    best_result: Dict[str, Any],
    ranking_df,
    characteristics: Dict[str, Any],
    selected_algorithms: List[str]
) -> Dict[str, Any]:
    """
    Mode B empirical recommendation based on executed benchmark results.
    Preserves and refines the existing benchmark recommendation functionality.
    """
    algorithm = best_result["algorithm"]
    score = ranking_df.iloc[0]["Score"]
    best_value = ranking_df.iloc[0]["Best Value"]
    regret = ranking_df.iloc[0]["Regret"]
    runtime = ranking_df.iloc[0]["Runtime (s)"]

    dim = characteristics.get("dimension", 10)
    modality = characteristics.get("modality", "Medium")
    effective_dim = characteristics.get("effective_dimension", dim)
    noise = characteristics.get("noise", 0.0)

    reasons = []
    confidence = "Medium"

    if dim > 30:
        reasons.append(f"High dimensionality (d={dim}): surrogate scalability and trust-region or embedding mechanisms are critical.")
        if "TuRBO" in algorithm or "REMBO" in algorithm:
            reasons.append("TuRBO / REMBO effectively mitigate the curse of dimensionality.")
            confidence = "High"
        else:
            confidence = "Medium"
    else:
        reasons.append(f"Moderate/low dimensionality (d={dim}): surrogate models capture the response surface accurately.")
        confidence = "High"

    if "High" in str(modality):
        reasons.append("Highly multimodal landscape with many local attractors.")
        if "TuRBO" in algorithm:
            reasons.append("TuRBO's local trust regions prevent getting trapped in poor global regions.")
        elif "GP" in algorithm:
            reasons.append("GP uncertainty modeling enables escaping shallow local minima.")

    if effective_dim < dim * 0.5:
        reasons.append(f"Estimated effective dimension ({effective_dim}) is significantly lower than full dimension ({dim}).")
        if "REMBO" in algorithm:
            reasons.append("REMBO efficiently exploits low effective dimensionality.")
            confidence = "High"

    if noise > 0.05:
        reasons.append(f"Measurement noise present (sigma = {noise:.2f}): Gaussian Process prior acts as a natural regularizer.")

    # Algorithm specific empirical achievement
    if "TuRBO" in algorithm:
        reasons.append("TuRBO achieved the highest composite score across convergence, regret, and efficiency.")
    elif "REMBO" in algorithm:
        reasons.append("REMBO achieved the top composite benchmark performance.")
    elif "GP-EI" in algorithm:
        reasons.append("GP-EI delivered the best trade-off between improvement per sample and runtime.")
    elif "GP-UCB" in algorithm:
        reasons.append("GP-UCB demonstrated superior exploration dynamics and regret minimization.")
    else:
        reasons.append("Random Search performed best; consider increasing the evaluation budget or adjusting hyperparameter bounds.")
        confidence = "Low"

    reason_text = "\n".join([f"• {r}" for r in reasons])

    return {
        "algorithm": algorithm,
        "score": round(score, 1),
        "confidence": confidence,
        "reason": reason_text,
        "best_value": best_value,
        "regret": regret,
        "runtime": runtime,
        "disclaimer": "Empirical recommendation based on synthetic benchmark execution with identical random seeds and budgets."
    }


def format_recommendation(rec: Dict[str, Any]) -> str:
    """Format recommendation for clean display."""
    return f"""🏆 RECOMMENDED ALGORITHM: {rec['algorithm']}

Composite Score: {rec['score']} / 100
Confidence Level: {rec['confidence']}

Key Evidence:
{rec['reason']}

---
Notice: {rec.get('disclaimer', '')}"""