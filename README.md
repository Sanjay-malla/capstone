# Project Name

> A web-based project built using Python and Streamlit.

## 🌐 Demo

A static HTML demo is available to preview the interface and basic functionality of the project.

**[View Demo](YOUR_HTML_DEMO_LINK_HERE)**

⚠️ **Note:** The HTML demo provides a preview of the project. To access the complete functionality, interactive features, and the full application, you need to run the project locally using Streamlit.

---

# Bayesian Optimization Algorithm Selection Platform

An AI-powered, natural-language-driven Bayesian Optimization (BO) decision support and benchmarking platform built for researchers, machine learning practitioners, and engineering scientists.

---

## 1. Project Overview

Bayesian Optimization is the premier paradigm for sample-efficient global optimization of expensive, noisy, black-box functions. However, selecting the appropriate BO algorithm (e.g., standard Gaussian Process with Expected Improvement vs. Trust-Region BO vs. Random Embeddings) requires deep specialized expertise in dimensionality, surrogate scalability, and landscape characteristics.

The **Bayesian Optimization Algorithm Selection Platform** bridges this gap:
- **Mode A (Real-World Problem Analysis):** Allows users to describe an engineering or machine learning problem in **natural language**. The platform extracts the optimization structure, checks parameter validity, handles missing information without fabrication, characterizes the problem geometry, evaluates algorithmic suitability, and provides evidence-based recommendations.
- **Mode B (Benchmark Validation):** Preserves and enhances full empirical benchmarking across standard synthetic test functions (Sphere, Rosenbrock, Rastrigin, Ackley, Schwefel) with reproducible controls (dimensions 2–100, noise, seed, budget) and interactive Plotly analytics.

---

## 2. Problem Statement

In traditional optimization platforms, practitioners are forced to manually configure technical parameters (kernel parameters, acquisition equations, surrogate types) before understanding whether their problem formulation is mathematically well-posed. Furthermore, many automated tools fabricate numerical optimization curves or invent missing variables when users provide incomplete real-world specifications.

This platform enforces **strict scientific honesty**:
1. **Zero Fabrication Principle:** If measurement noise, budget, or bounds are not specified in the input text, they are classified as `Unknown` and flagged with domain-specific explanations.
2. **Honest Objective Boundaries:** The system clearly communicates that natural-language analysis provides algorithmic recommendation and structural characterization, but **numerical benchmarking requires an executable objective function, physical simulator, or compatible dataset**.

---

## 3. Key Features

- **Natural Language Parsing:** Automatically extracts objective direction (minimize/maximize), target metrics, variables, continuous/integer/categorical types, lower/upper bounds, physical units, evaluation budgets, and monetary or temporal evaluation costs.
- **Transparent Missing Information Handling:** Highlights missing parameters (e.g., unspecified observation noise) with technical impact explanations and interactive refinement options.
- **Dual Problem Characterization:**
  - *Extracted Facts:* Dimensionality, variable types, search space bounds, cost profile, and operational constraints.
  - *Latent Properties:* Modality, smoothness, and effective dimension are honestly designated as `Unknown / Requires objective evaluation` unless empirically measured.
- **Intelligent Algorithm Suitability Matrix:** Evaluates 5 core algorithms (`GP-EI`, `GP-UCB`, `REMBO`, `TuRBO`, `Random Search`) against multi-dimensional criteria with platform-defined suitability scores.
- **Evidence-Based Recommendations:** Generates transparent recommendations with explicit rationale and scientific limitations.
- **Physical Simulator Proxy Connector:** Includes an executable simulation adapter for domain problems (such as injection molding defect rate minimization).
- **Synthetic Benchmark Validation Suite:** Supports rigorous, fair empirical benchmarking on Rastrigin, Ackley, Rosenbrock, Sphere, and Schwefel with identical initial samples and seeds.
- **Publication-Quality Visualizations:** Interactive Plotly convergence trajectories, log-scale regret curves, multi-metric score comparisons, and runtime profiling.
- **In-Memory Architecture:** Fully self-contained processing without database requirements, user authentication, or permanent tracking.
- **Comprehensive Data Export:** Instant one-click export of structured analysis and benchmark runs to standard CSV and JSON formats.

---

## 4. Architecture

```
                                  USER INPUT
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
   Mode A: Natural Language                              Mode B: Synthetic Benchmark
   Problem Description                                   Function Selection
            │                                                     │
            ▼                                                     │
   problem_parser.py                                              │
   ├─ Direction & Target Metric                                   │
   ├─ Variables, Bounds & Units                                   │
   ├─ Budget, Cost & Noise                                        │
   └─ Operational Constraints                                     │
            │                                                     │
            ▼                                                     │
   Missing Information Handler                                    │
   (Flag unspecified items; no fabrication)                       │
            │                                                     │
            ▼                                                     │
   characterization.py                                            ▼
   ├─ Verified Structural Facts                          characterization.py
   └─ Latent Geometry (Unknown)                          (Gradient-based effective dim)
            │                                                     │
            ▼                                                     │
   recommendation.py                                              ▼
   (Multi-factor Suitability Matrix)                     algorithms.py
            │                                            (GP-EI, GP-UCB, REMBO, TuRBO, RS)
            ▼                                                     │
   objective_adapter.py                                           ▼
   (Connect simulator or state analysis-only)            evaluation.py
            │                                            (Regret, Efficiency, Score)
            └──────────────────────────┬──────────────────────────┘
                                       ▼
                             Streamlit Dashboard
                         (visualization.py + app.py)
                                       │
                                       ▼
                            CSV & JSON Export Engine
```

---

## 5. Natural-Language Workflow

1. **User Input:** The user enters a natural-language description of an engineering or ML process.
2. **Entity & Range Extraction:** Regex and rule-based extractors capture parameters (e.g., `180 to 250 degrees Celsius` → `lower_bound: 180.0, upper_bound: 250.0, unit: '°C'`).
3. **Type & Cost Inference:** Identifies variable types (Continuous vs. Integer) and evaluation expensiveness from monetary (`₹5,000`, `$500`) or duration cues (`20 minutes`, `several minutes`).
4. **Validation:** Checks whether bounds, budget, or noise are missing.
5. **Characterization:** Generates verified structural characteristics and isolates unverified landscape properties.
6. **Suitability Scoring:** Evaluates algorithmic trade-offs across GP-EI, GP-UCB, REMBO, TuRBO, and Random Search.
7. **Recommendation & Export:** Formulates transparent recommendations and enables CSV/JSON download.

---

## 6. Problem Schema

Structured data representation defined in `problem_schema.py`:

```python
{
    "problem_description": str,
    "objective": {
        "type": "minimize" | "maximize" | "unknown",
        "target": str,
        "unit": str
    },
    "variables": [
        {
            "name": str,
            "lower_bound": float | None,
            "upper_bound": float | None,
            "unit": str,
            "type": "continuous" | "integer" | "categorical",
            "categories": list[str]
        }
    ],
    "constraints": list[str],
    "dimension": int | None,
    "noise": float | None,               # None indicates Unknown (never fabricated)
    "evaluation_budget": int | None,
    "evaluation_cost": str | None,
    "expensive_evaluation": bool | None,
    "known_optimum": float | None,
    "domain": str,
    "additional_information": str,
    "missing_fields": list[dict]
}
```

---

## 7. Problem Characterization

The characterization module (`characterization.py`) strictly segregates what is knowable from problem text versus what requires function evaluations:

| Dimension / Property | Real-World Text Mode | Benchmark Function Mode |
| :--- | :--- | :--- |
| **Dimensionality ($d$)** | Extracted from variable count | Configured by user ($2 \le d \le 100$) |
| **Variable Subtypes** | Continuous, Discrete, Mixed | Continuous $[lb, ub]^d$ |
| **Search Space** | Box-constrained / unbounded | Bounded hypercube |
| **Cost Profile** | Inferred from currency/time | Simulated |
| **Observation Noise** | Extracted value or `Unknown` | User-configured Gaussian $\sigma$ |
| **Modality** | `Unknown / Requires objective evaluation` | High / Medium / Low (known math) |
| **Smoothness** | `Unknown / Requires objective evaluation` | Analytical class ($C^2$, etc.) |
| **Effective Dimension** | `Unknown / Requires gradient analysis` | Numerical finite-difference estimate |

---

## 8. Algorithm Selection Framework

The platform evaluates five distinct algorithms without hardcoded shortcuts:

1. **GP-EI (Gaussian Process + Expected Improvement)**
   - *Best For:* Low-dimensional ($d \le 15$), highly expensive black-box functions with compact budgets ($N \le 150$).
   - *Limitation:* $O(N^3)$ computational cost and over-exploration of boundaries in high dimensions ($d > 25$).
2. **GP-UCB (Gaussian Process + Upper Confidence Bound)**
   - *Best For:* Low/moderate dimensions ($d \le 15$) where explicit control over exploration ($\beta$) and sub-linear theoretical regret bounds are valued under noisy observations.
   - *Limitation:* Poor scaling when dimensionality dilutes confidence bounds across empty space.
3. **REMBO (Random Embedding Bayesian Optimization)**
   - *Best For:* High ambient dimensions ($d \ge 25$) that possess a low effective intrinsic subspace ($d_e \ll D$).
   - *Limitation:* Boundary clipping distortions and projection artifacts on low-dimensional or discrete spaces.
4. **TuRBO (Trust Region Bayesian Optimization)**
   - *Best For:* High-dimensional ($15 \le d \le 100+$) and multimodal problems with budgets from 80 to 500+. Local trust regions dynamically expand and contract based on consecutive successes and failures.
   - *Limitation:* Local search can potentially miss distant basins if trust region collapses prematurely.
5. **Random Search (Baseline)**
   - *Role:* Required scientific baseline. Zero surrogate overhead, but near-zero sample efficiency.

---

## 9. Benchmarking Suite

In Mode B, algorithms are evaluated under identical conditions:
- **Same Benchmark Function:** Sphere, Rosenbrock, Rastrigin, Ackley, Schwefel.
- **Same Dimensionality:** Configurable from 2 to 100.
- **Same Noise Level:** Gaussian $\mathcal{N}(0, \sigma^2)$ with $\sigma \in [0.0, 1.0]$.
- **Same Initial Design:** Identical Sobol/uniform pseudo-random points.
- **Same Budget & Seed:** Enforces fair, reproducible comparison.

---

## 10. Evaluation Metrics

When benchmarking is executed, metrics are calculated and normalized:

$$\text{Composite Score} = 0.40 \cdot S_{\text{BestValue}} + 0.25 \cdot S_{\text{Regret}} + 0.10 \cdot S_{\text{Runtime}} + 0.15 \cdot S_{\text{SampleEfficiency}} + 0.10 \cdot S_{\text{ConvergenceRate}}$$

- **Best Value (40%):** Minimum objective value observed (min-max inverted).
- **Regret (25%):** Distance to known global optimum $f(x^*) - f^*$, or empirical regret relative to best observed value if $f^*$ is unknown (min-max inverted).
- **Runtime (10%):** Total wall-clock execution time (min-max inverted).
- **Sample Efficiency (15%):** Mean magnitude of objective reduction per improving step (min-max scaled).
- **Convergence Rate (10%):** Ratio of initial regret eliminated across the run (min-max scaled).

*Notice: The composite score is a platform-defined evaluation score, not a universal physical constant.*

---

## 11. Real-World Analysis Limitations

When running in **Mode A (Real-World Analysis)**:
- Problem analysis and algorithm suitability recommendations are generated from extracted characteristics and Bayesian Optimization theory.
- **Numerical benchmarking is not executed** unless an executable simulator (such as the built-in injection molding simulator proxy) or an external objective function is explicitly attached.
- Empirical performance on novel physical systems cannot be guaranteed without real trial data.

---

## 12. Streamlit Dashboard

The web interface is built using Streamlit with custom CSS styling:
- **Full-Width Hero Section:** Overview and platform purpose.
- **Sidebar Navigation:** Seamless switching between Real-World Analysis (Mode A), Benchmark Validation (Mode B), and Architectural Documentation.
- **Analysis Status Component:** Real-time feedback showing parsing, characterization, and suitability steps.
- **Variable Table & Missing Information Banners:** Clean, actionable tables highlighting bounds, units, and missing items.
- **Interactive Plotly Visualizations:** Dynamic convergence curves, regret progression, score charts, and runtime comparisons.
- **In-Memory Download Buttons:** One-click CSV and JSON exports.

---

## 13. Installation

Ensure Python 3.11+ is installed.

```bash
# 1. Clone or navigate to the repository
cd "g:/capston project/BO_Algorithm_Selector"

# 2. Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 14. Running the Application

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

Open your web browser and navigate to `http://localhost:8501`.

---

## 15. Example Inputs

### Example 1: Injection Molding Process (Manufacturing)
> *"I operate an injection molding machine and want to minimize the percentage of defective parts. The injection temperature can range from 180 to 250 degrees Celsius, pressure from 50 to 120 MPa, injection speed from 20 to 100 mm/s, and cooling time from 5 to 30 seconds. Each experiment costs approximately ₹5,000 and I can perform 100 experiments."*

- **Extracted:** Minimize defect rate, 4 continuous parameters with bounds and units, budget = 100, cost = ₹5,000, noise = Unknown.
- **Recommended Algorithm:** `TuRBO` or `GP-EI` (High suitability due to low dimensionality and expensive trials).

### Example 2: Deep Learning Hyperparameters (Machine Learning)
> *"I want to maximize the validation accuracy of a machine learning model. I can tune learning rate from 0.0001 to 0.1, batch size from 16 to 256, dropout from 0 to 0.5, and number of layers from 2 to 8. Training one model takes around 20 minutes and I can run 80 experiments."*

- **Extracted:** Maximize validation accuracy, mixed continuous/integer variables, budget = 80, cost = ~20 min/eval.
- **Recommended Algorithm:** `TuRBO` / `GP-EI` with discrete handling.

### Example 3: Robot Arm Trajectory (Robotics & Control)
> *"I want to minimize the movement time of a robot arm while maintaining positioning accuracy above 99%. I can modify motor speed, acceleration, and trajectory parameters within predefined operating ranges. Each simulation takes several minutes."*

- **Extracted:** Minimize movement time, constraint = accuracy > 99%, 3 variables (bounds missing!).
- **Missing Info Trigger:** Correctly alerts user that bounds are missing without inventing artificial numbers.

---

## 16. Project Structure

```
BO_Algorithm_Selector/
│
├── app.py                   # Main Streamlit web application & UI controllers
├── problem_schema.py        # Data classes for problems, variables, missing fields & recommendations
├── problem_parser.py        # Robust natural language parsing engine
├── characterization.py      # Problem characterization layer (facts vs. unverified latent properties)
├── recommendation.py        # Multi-factor suitability matrix and recommendation engine
├── objective_adapter.py     # Executable objective abstraction & simulator proxies
├── benchmarks.py            # Synthetic mathematical test functions (Sphere, Rastrigin, etc.)
├── algorithms.py            # BO implementations: GP-EI, GP-UCB, REMBO, TuRBO, Random Search
├── evaluation.py            # Metric calculation, min-max normalization & multi-metric scoring
├── visualization.py         # Publication-quality interactive Plotly charts
├── examples.py              # Pre-configured engineering problem templates
├── requirements.txt         # Core dependencies
└── README.md                # Comprehensive documentation
```

---

## 17. Future Enhancements

- **LLM API Plugin:** Optional integration with Gemini or OpenAI API for advanced linguistic disambiguation of domain jargon.
- **Constrained Bayesian Optimization:** Native implementation of expected improvement with constraints ($cEI$) using auxiliary Gaussian Process classifiers.
- **Batch BO (q-EI):** Support for parallel asynchronous evaluations on high-performance compute clusters.
- **Multi-Fidelity BO:** Support for combining low-cost simulations with high-cost physical experimental validation.sting, consider giving the repository a star!
