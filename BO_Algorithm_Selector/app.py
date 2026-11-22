"""
app.py
Bayesian Optimization Algorithm Selection Platform
AI-Powered Natural-Language Problem Analysis & Experimental Benchmark Dashboard
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import json
import io
from typing import Dict, Any, List, Optional

# Core engine imports
from benchmarks import get_benchmark, list_benchmarks
from algorithms import run_algorithm, list_algorithms
from evaluation import rank_algorithms, calculate_metrics
from characterization import (
    get_problem_characteristics,
    characterize_parsed_problem,
    ProblemCharacterization
)
from recommendation import (
    evaluate_algorithm_suitability,
    generate_recommendation,
    format_recommendation
)
from problem_schema import (
    ProblemSchema,
    VariableSchema,
    ObjectiveSchema,
    ObjectiveType,
    VariableType,
    SuitabilityLevel,
    RecommendationResult
)
from problem_parser import ProblemParser
from objective_adapter import ObjectiveConnector, BaseObjectiveAdapter, InjectionMoldingSimulator
from examples import EXAMPLES, list_example_names, get_example
from visualization import (
    create_suitability_chart,
    create_convergence_plot,
    create_regret_plot,
    create_score_plot,
    create_runtime_efficiency_plot
)


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bayesian Optimization Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------------------------------------
# CUSTOM CSS FOR PROFESSIONAL RESEARCH / ANALYTICS UI
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Typography & Font Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 95rem;
    }

    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #1e3a8a 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 2.2rem 2.5rem;
        color: #f8fafc;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        max-width: 58rem;
        line-height: 1.6;
        margin-bottom: 0;
    }

    /* Metric & Characterization Cards */
    .feature-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .feature-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .card-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.25rem;
    }
    .card-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
    }
    .card-subtext {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    /* Recommendation Showcase Banner */
    .recommendation-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 1.8rem 2rem;
        color: #ffffff;
        margin: 1.5rem 0;
        box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.25);
    }
    .rec-header {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #93c5fd;
        margin-bottom: 0.4rem;
    }
    .rec-algo-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.8rem;
    }
    .rec-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 0.25rem 0.85rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-high { background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
    .badge-medium { background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; }
    .badge-low { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .badge-baseline { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
    .badge-warning { background: #fff7ed; color: #9a3412; border: 1px solid #ffedd5; }
    .badge-info { background: #f0fdfa; color: #115e59; border: 1px solid #ccfbf1; }

    /* Callout & Notice Boxes */
    .info-callout {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    .warning-callout {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    .honest-notice {
        background: #f1f5f9;
        border: 1px dashed #94a3b8;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        color: #334155;
        margin: 1rem 0;
    }

    /* Streamlit Button Tweaks */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .primary-btn>button {
        background: #1e3a8a !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.65rem 1.5rem !important;
    }
    .primary-btn>button:hover {
        background: #1e40af !important;
        box-shadow: 0 4px 6px -1px rgba(30, 58, 138, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
def initialize_session_state():
    # Mode A State (Natural Language Real-World Analysis)
    if "nl_input" not in st.session_state:
        st.session_state.nl_input = EXAMPLES["Injection Molding Process"]["description"]
    if "parsed_problem" not in st.session_state:
        st.session_state.parsed_problem = None
    if "problem_characterization" not in st.session_state:
        st.session_state.problem_characterization = None
    if "suitability_recommendation" not in st.session_state:
        st.session_state.suitability_recommendation = None
    if "nl_active_tab" not in st.session_state:
        st.session_state.nl_active_tab = 0

    # Mode B State (Synthetic Benchmark Validation)
    if "bench_results" not in st.session_state:
        st.session_state.bench_results = None
    if "bench_ranking_df" not in st.session_state:
        st.session_state.bench_ranking_df = None
    if "bench_characteristics" not in st.session_state:
        st.session_state.bench_characteristics = None
    if "bench_recommendation" not in st.session_state:
        st.session_state.bench_recommendation = None
    if "bench_obj" not in st.session_state:
        st.session_state.bench_obj = None

    # Simulation Proxy State for Mode A
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = None
    if "sim_ranking_df" not in st.session_state:
        st.session_state.sim_ranking_df = None


# -----------------------------------------------------------------------------
# EXPORT GENERATION HELPERS
# -----------------------------------------------------------------------------
def generate_mode_a_export_json(
    problem: ProblemSchema,
    char: ProblemCharacterization,
    rec: RecommendationResult
) -> str:
    data = {
        "platform": "Bayesian Optimization Algorithm Selection Platform",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "mode": "Real-World Problem Analysis (Natural Language)",
        "problem": problem.to_dict(),
        "characterization": char.to_dict(),
        "recommendation": rec.to_dict(),
        "scientific_integrity_notice": (
            "No numerical optimization was executed. Recommendation is derived analytically "
            "from extracted problem characteristics and algorithmic capability theory."
        )
    }
    return json.dumps(data, indent=2)


def generate_mode_a_export_csv(
    problem: ProblemSchema,
    rec: RecommendationResult
) -> str:
    output = io.StringIO()
    output.write("# PROBLEM SPECIFICATION\n")
    output.write(f"Description,{problem.problem_description.replace(',', ';')}\n")
    output.write(f"Objective Direction,{problem.objective.obj_type.value}\n")
    output.write(f"Objective Target,{problem.objective.target}\n")
    output.write(f"Dimension,{problem.dimension or 'Unknown'}\n")
    output.write(f"Budget,{problem.evaluation_budget or 'Unknown'}\n")
    output.write(f"Cost,{problem.evaluation_cost or 'Unknown'}\n")
    output.write(f"Noise,{problem.noise if problem.noise is not None else 'Unknown'}\n\n")

    output.write("# OPTIMIZATION VARIABLES\n")
    var_df = pd.DataFrame([v.to_dict() for v in problem.variables])
    if not var_df.empty:
        var_df.to_csv(output, index=False)
    output.write("\n# ALGORITHM SUITABILITY MATRIX\n")
    suit_rows = [
        {
            "Algorithm": s.algorithm,
            "Suitability": s.suitability.value,
            "Platform Score": s.score,
            "Key Reason": s.key_reason
        } for s in rec.suitability_table
    ]
    pd.DataFrame(suit_rows).to_csv(output, index=False)

    output.write(f"\n# RECOMMENDATION\n")
    output.write(f"Recommended Algorithm,{rec.recommended_algorithm}\n")
    output.write(f"Confidence Level,{rec.confidence}\n")
    output.write(f"Platform Score,{rec.suitability_score}\n")
    return output.getvalue()


def generate_mode_b_export_csv(
    ranking_df: pd.DataFrame,
    characteristics: Dict[str, Any],
    recommendation: Dict[str, Any]
) -> str:
    output = io.StringIO()
    output.write("# BENCHMARK EXPERIMENTAL RANKING\n")
    ranking_df.to_csv(output, index=False)
    output.write("\n# PROBLEM CHARACTERISTICS\n")
    for k, v in characteristics.items():
        output.write(f"{k},{str(v).replace(',', ';')}\n")
    output.write("\n# EMPIRICAL RECOMMENDATION\n")
    output.write(f"Recommended Algorithm,{recommendation['algorithm']}\n")
    output.write(f"Composite Score,{recommendation['score']}\n")
    output.write(f"Confidence,{recommendation['confidence']}\n")
    return output.getvalue()


# -----------------------------------------------------------------------------
# MODE A: REAL-WORLD PROBLEM ANALYSIS CONTROLLER
# -----------------------------------------------------------------------------
def analyze_natural_language_problem(text: str, noise_override: Optional[float] = None):
    """
    Executes the analytical pipeline:
    NL Text -> ProblemParser -> ProblemSchema -> ProblemCharacterization -> RecommendationResult
    """
    parser = ProblemParser()
    parsed = parser.parse(text)

    # Optional noise override if user specified it in missing info modal
    if noise_override is not None:
        parsed.noise = noise_override
        # Remove noise from missing fields list
        parsed.missing_fields = [m for m in parsed.missing_fields if m.field_name != "noise"]

    char = characterize_parsed_problem(parsed)
    rec = evaluate_algorithm_suitability(char, parsed)

    st.session_state.parsed_problem = parsed
    st.session_state.problem_characterization = char
    st.session_state.suitability_recommendation = rec
    st.session_state.sim_results = None  # Reset simulation cache on new analysis


# -----------------------------------------------------------------------------
# MODE B: SYNTHETIC BENCHMARK OPTIMIZATION CONTROLLER
# -----------------------------------------------------------------------------
def run_benchmark_experiment(
    benchmark_name: str,
    dimension: int,
    noise: float,
    n_initial: int,
    budget: int,
    seed: int,
    selected_algos: List[str]
):
    benchmark = get_benchmark(benchmark_name)

    def noisy_objective(x):
        return benchmark.evaluate(x, noise)

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, algo_name in enumerate(selected_algos):
        status_text.markdown(f"**Benchmarking `{algo_name}`...** (Algorithm {i+1} of {len(selected_algos)})")
        try:
            result = run_algorithm(
                algo_name, noisy_objective, benchmark.bounds, dimension,
                n_initial, budget, noise, seed
            )
            results.append(result)
        except Exception as e:
            st.error(f"Execution error for {algo_name}: {str(e)}")
            results.append({
                "algorithm": algo_name,
                "evaluations": [],
                "objective_values": [],
                "best_values": [],
                "best_value": float("inf"),
                "best_point": [],
                "runtime": 0.0
            })
        progress_bar.progress((i + 1) / len(selected_algos))

    status_text.markdown("**Computing performance metrics, regret, and normalized scores...**")
    time.sleep(0.3)
    progress_bar.empty()
    status_text.empty()

    char = get_problem_characteristics(benchmark_name, dimension, noise)
    ranking_df = rank_algorithms(results, benchmark.global_optimum, budget)
    best_algo_name = ranking_df.iloc[0]["Algorithm"]
    best_result = next((r for r in results if r["algorithm"] == best_algo_name), results[0])
    recommendation = generate_recommendation(best_result, ranking_df, char, selected_algos)

    st.session_state.bench_results = results
    st.session_state.bench_ranking_df = ranking_df
    st.session_state.bench_characteristics = char
    st.session_state.bench_recommendation = recommendation
    st.session_state.bench_obj = benchmark


# -----------------------------------------------------------------------------
# MAIN APPLICATION INTERFACE
# -----------------------------------------------------------------------------
def main():
    initialize_session_state()

    # 1. SIDEBAR NAVIGATION & SETTINGS
    with st.sidebar:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1.2rem;">
            <div style="font-size:1.8rem;">🧠</div>
            <div>
                <div style="font-size:1.15rem;font-weight:700;color:#0f172a;">Bayesian Opt</div>
                <div style="font-size:0.75rem;color:#64748b;font-weight:500;">Algorithm Selection Platform</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        mode = st.radio(
            "Navigation Mode",
            ["🎯 Real-World Problem Analysis", "🔬 Benchmark Validation", "📖 Architecture & Reference"],
            index=0,
            help="Select between Natural Language Problem Analysis (Mode A) and Benchmark Validation (Mode B)"
        )

        st.divider()

        if mode == "🎯 Real-World Problem Analysis":
            st.markdown("### 📋 Quick Example Presets")
            selected_preset = st.selectbox(
                "Load Sample Problem",
                list_example_names(),
                index=0,
                help="Select a representative problem template from industry or research"
            )
            if st.button("📥 Load Preset into Input", use_container_width=True):
                st.session_state.nl_input = EXAMPLES[selected_preset]["description"]
                st.rerun()

            st.caption(f"**Domain:** {EXAMPLES[selected_preset]['category']}")
            st.caption(f"**Characteristics:** {EXAMPLES[selected_preset]['notes']}")

        elif mode == "🔬 Benchmark Validation":
            st.markdown("### ⚙️ Benchmark Setup")
            bench_name = st.selectbox(
                "Benchmark Function",
                list_benchmarks(),
                index=2,  # Default to Rastrigin as standard validation
                help="Standard synthetic mathematical test function"
            )

            bench_dim = st.number_input(
                "Dimensions (d)",
                min_value=2, max_value=100, value=50, step=1,
                help="Number of decision variables (2-100)"
            )

            bench_noise = st.number_input(
                "Observation Noise (std)",
                min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%.2f",
                help="Gaussian noise added to function evaluations"
            )

            bench_budget = st.number_input(
                "Evaluation Budget",
                min_value=10, max_value=500, value=100, step=10,
                help="Total function evaluations permitted per algorithm"
            )

            bench_initial = st.number_input(
                "Initial Samples",
                min_value=2, max_value=50, value=10, step=1,
                help="Random design points evaluated before active BO acquisition"
            )

            bench_seed = st.number_input(
                "Random Seed",
                min_value=0, max_value=10000, value=42, step=1,
                help="Random seed for fair reproducibility across all algorithms"
            )

            st.divider()
            st.markdown("### 🤖 Algorithms")
            all_algos = list_algorithms()
            selected_bench_algos = []
            algo_cols = st.columns(2)
            for i, algo in enumerate(all_algos):
                with algo_cols[i % 2]:
                    if st.checkbox(algo, value=True, key=f"bench_algo_{algo}"):
                        selected_bench_algos.append(algo)

            demo_mode = st.checkbox("⚡ Fast Demo Mode", value=False, help="Restricts budget & dimension for instantaneous demo runs")
            if demo_mode:
                bench_budget = min(bench_budget, 30)
                bench_initial = min(bench_initial, 5)
                bench_dim = min(bench_dim, 15)

            st.divider()
            run_bench_btn = st.button("🚀 RUN BENCHMARK", type="primary", use_container_width=True)

        st.divider()
        if st.button("🔄 Reset All States", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.markdown("""
        <div style="font-size:0.75rem;color:#94a3b8;text-align:center;margin-top:1.5rem;">
            In-Memory Processing • No Database • No Authentication
        </div>
        """, unsafe_allow_html=True)

    # 2. HERO CONTAINER
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">
            <span>🧠</span> Bayesian Optimization Algorithm Selection Platform
        </div>
        <div class="hero-subtitle">
            An AI-powered decision support and benchmarking platform. Describe real-world optimization problems
            in natural language to extract parameter bounds, evaluate algorithmic suitability, and receive
            evidence-based recommendations without numerical fabrication.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MODE A: REAL-WORLD PROBLEM ANALYSIS (NATURAL LANGUAGE)
    # -------------------------------------------------------------------------
    if mode == "🎯 Real-World Problem Analysis":
        st.markdown("### 📝 Natural Language Problem Description")
        st.markdown(
            "Enter your engineering or scientific optimization problem below. The platform will automatically extract "
            "the objective direction, variables, bounds, units, budget, and operational constraints."
        )

        user_text = st.text_area(
            "Optimization Problem Statement",
            value=st.session_state.nl_input,
            height=130,
            placeholder="e.g. I operate an injection molding machine and want to minimize the percentage of defective parts...",
            help="Describe your objective, factors, ranges, constraints, budget, and measurement characteristics."
        )
        st.session_state.nl_input = user_text

        col_btn, col_note = st.columns([1, 3])
        with col_btn:
            analyze_btn = st.button("🔍 Analyze Problem", type="primary", use_container_width=True)
        with col_note:
            st.markdown("""
            <div style="font-size:0.85rem;color:#64748b;padding-top:0.45rem;">
                ⚡ <b>Zero Fabrication Principle:</b> Missing parameters remain <code>Unknown</code> and are never artificially populated.
            </div>
            """, unsafe_allow_html=True)

        if analyze_btn:
            if not user_text.strip():
                st.warning("Please enter a problem description before analyzing.")
            else:
                with st.status("Analyzing optimization problem...", expanded=True) as status_box:
                    st.write("✓ Understanding natural-language problem description")
                    time.sleep(0.15)
                    st.write("✓ Extracting optimization variables, bounds, and physical units")
                    time.sleep(0.15)
                    st.write("✓ Detecting objective direction and target metrics")
                    time.sleep(0.15)
                    st.write("✓ Validating information & checking for missing parameters")
                    time.sleep(0.15)
                    st.write("✓ Characterizing optimization problem structure and search space")
                    time.sleep(0.15)
                    st.write("✓ Evaluating algorithm suitability: GP-EI, GP-UCB, REMBO, TuRBO, Random Search")
                    time.sleep(0.15)
                    analyze_natural_language_problem(user_text)
                    status_box.update(label="Problem analysis complete!", state="complete", expanded=False)
                st.rerun()

        # Display Parsed Analysis if available
        if st.session_state.parsed_problem is not None:
            problem: ProblemSchema = st.session_state.parsed_problem
            char: ProblemCharacterization = st.session_state.problem_characterization
            rec: RecommendationResult = st.session_state.suitability_recommendation

            st.divider()

            # Extracted High-Level Metric Cards
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                obj_icon = "📉" if problem.objective.obj_type == ObjectiveType.MINIMIZE else "📈"
                st.markdown(f"""
                <div class="feature-card">
                    <div class="card-label">Objective</div>
                    <div class="card-value">{obj_icon} {problem.objective.obj_type.value.capitalize()}</div>
                    <div class="card-subtext">{problem.objective.target}</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                dim_str = str(problem.dimension) if problem.dimension is not None else "Unknown"
                st.markdown(f"""
                <div class="feature-card">
                    <div class="card-label">Variables (d)</div>
                    <div class="card-value">🎛️ {dim_str}</div>
                    <div class="card-subtext">{char.variable_types_summary}</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                budget_str = f"{problem.evaluation_budget} runs" if problem.evaluation_budget else "Unknown"
                st.markdown(f"""
                <div class="feature-card">
                    <div class="card-label">Evaluation Budget</div>
                    <div class="card-value">⏱️ {budget_str}</div>
                    <div class="card-subtext">Max allowable trials</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                cost_str = problem.evaluation_cost if problem.evaluation_cost else "Unknown"
                st.markdown(f"""
                <div class="feature-card">
                    <div class="card-label">Cost per Eval</div>
                    <div class="card-value">💰 {cost_str}</div>
                    <div class="card-subtext">{char.evaluation_cost_profile}</div>
                </div>
                """, unsafe_allow_html=True)
            with m5:
                noise_str = f"σ = {problem.noise:.2f}" if problem.noise is not None else "Unknown"
                st.markdown(f"""
                <div class="feature-card">
                    <div class="card-label">Observed Noise</div>
                    <div class="card-value">📡 {noise_str}</div>
                    <div class="card-subtext">{char.noise_profile}</div>
                </div>
                """, unsafe_allow_html=True)

            # Missing Information Handler
            if problem.missing_fields:
                st.markdown("#### ⚠️ Unspecified Problem Details")
                st.markdown(
                    "To maintain scientific accuracy, missing fields are flagged rather than assumed. "
                    "Review how these unspecified parameters influence algorithm suitability:"
                )

                for missing in problem.missing_fields:
                    severity_badge = f'<span class="badge badge-warning">{missing.severity.upper()}</span>'
                    st.markdown(f"""
                    <div class="warning-callout">
                        <div style="font-weight:700;color:#92400e;margin-bottom:0.25rem;">
                            {severity_badge} <b>{missing.display_name}</b> was not explicitly specified
                        </div>
                        <div style="font-size:0.88rem;color:#78350f;margin-bottom:0.4rem;">
                            <b>Impact:</b> {missing.impact}
                        </div>
                        <div style="font-size:0.82rem;color:#b45309;">
                            💡 <b>Recommendation:</b> {missing.suggested_action}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Interactive Quick-Fill for Noise if missing
                has_noise_missing = any(m.field_name == "noise" for m in problem.missing_fields)
                if has_noise_missing:
                    with st.expander("🛠️ Optional: Provide Noise Estimate to Refine Selection", expanded=False):
                        st.markdown(
                            "If you know your experimental measurement variability, select it below to update recommendations:"
                        )
                        c_opt1, c_opt2 = st.columns([2, 1])
                        with c_opt1:
                            user_noise_choice = st.selectbox(
                                "Estimated Measurement Noise",
                                [
                                    "Remain Unknown (Assumes default noise-tolerant GP prior)",
                                    "Low Noise (σ ≈ 0.01 - Highly repeatable lab instruments)",
                                    "Moderate Noise (σ ≈ 0.10 - Standard industrial tolerance)",
                                    "High Noise (σ ≈ 0.25 - Noisy field or biological assays)"
                                ]
                            )
                        with c_opt2:
                            if st.button("Apply Noise Estimate", use_container_width=True):
                                if "Low Noise" in user_noise_choice:
                                    analyze_natural_language_problem(user_text, noise_override=0.01)
                                elif "Moderate Noise" in user_noise_choice:
                                    analyze_natural_language_problem(user_text, noise_override=0.10)
                                elif "High Noise" in user_noise_choice:
                                    analyze_natural_language_problem(user_text, noise_override=0.25)
                                st.rerun()

            # Tabs for Detailed Analysis
            tab_vars, tab_char, tab_suit, tab_rec, tab_sim = st.tabs([
                "🎛️ Extracted Variables",
                "📊 Problem Characterization",
                "⚖️ Algorithm Suitability",
                "🏆 Recommendation",
                "🧪 Simulator & Executable Objective"
            ])

            # Tab 1: Extracted Variables Table
            with tab_vars:
                st.markdown("#### Extracted Optimization Parameters")
                if problem.variables:
                    var_data = []
                    for v in problem.variables:
                        lb_display = f"{v.lower_bound:g}" if v.lower_bound is not None else "Unknown"
                        ub_display = f"{v.upper_bound:g}" if v.upper_bound is not None else "Unknown"
                        range_display = f"[{lb_display}, {ub_display}]" if (v.lower_bound is not None and v.upper_bound is not None) else "Unbounded / Incomplete"
                        var_data.append({
                            "Parameter Name": v.name,
                            "Type": v.var_type.value.capitalize(),
                            "Lower Bound": lb_display,
                            "Upper Bound": ub_display,
                            "Valid Range": range_display,
                            "Unit": v.unit if v.unit else "dimensionless"
                        })
                    var_df = pd.DataFrame(var_data)
                    st.dataframe(var_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No specific parameters were detected. Please ensure parameter ranges or names are listed.")

                if problem.constraints:
                    st.markdown("#### Detected Operational Constraints")
                    for c in problem.constraints:
                        st.markdown(f"- 🔒 **Constraint:** `{c}`")

            # Tab 2: Problem Characterization Dashboard
            with tab_char:
                st.markdown("#### Formal Problem Characterization")
                st.markdown(
                    "Problem characteristics are systematically partitioned between **verified extracted facts** "
                    "and properties that **cannot be inferred without numerical function evaluations**."
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("##### 📌 Verified Structural Characteristics")
                    st.markdown(f"- **Dimensionality:** {char.dimension_category}")
                    st.markdown(f"- **Search Space Geometry:** {char.search_space_type}")
                    st.markdown(f"- **Variable Subtypes:** {char.variable_types_summary}")
                    st.markdown(f"- **Objective Target:** {problem.objective.target} ({problem.objective.obj_type.value})")
                    st.markdown(f"- **Evaluation Cost Profile:** {char.evaluation_cost_profile}")
                    st.markdown(f"- **Measurement Noise:** {char.noise_profile}")
                    st.markdown(f"- **Constraint Regime:** {char.constraint_status}")

                with c2:
                    st.markdown("##### 🔬 Latent / Objective-Dependent Properties")
                    st.markdown(f"""
                    <div class="honest-notice">
                        <b>Scientific Integrity Notice:</b><br>
                        The following geometric properties cannot be scientifically determined from natural language text alone:<br><br>
                        • <b>Landscape Modality:</b> <code>{char.landscape_modality}</code><br>
                        • <b>Response Smoothness:</b> <code>{char.landscape_smoothness}</code><br>
                        • <b>Effective Dimensionality:</b> <code>{char.effective_dimension_info}</code><br><br>
                        <i>Any claims of smoothness, separability, or unimodality without numerical evaluations would be scientifically fabricated.</i>
                    </div>
                    """, unsafe_allow_html=True)

            # Tab 3: Algorithm Suitability Comparison
            with tab_suit:
                st.markdown("#### Multi-Algorithm Suitability Evaluation")
                st.markdown(
                    "Each Bayesian Optimization strategy is evaluated against the extracted characteristics. "
                    "Scores are clearly labeled as **Platform-Defined Suitability Scores (0-100)**."
                )

                st.plotly_chart(create_suitability_chart(rec), use_container_width=True)

                st.markdown("##### Detailed Comparison Table")
                suit_rows = []
                for s in rec.suitability_table:
                    suit_rows.append({
                        "Algorithm": s.algorithm,
                        "Suitability": s.suitability.value,
                        "Platform Score": f"{s.score:.1f} / 100",
                        "Key Assessment": s.key_reason,
                        "Primary Advantage": s.strengths_for_problem[0] if s.strengths_for_problem else "Standard implementation",
                        "Primary Limitation": s.drawbacks_for_problem[0] if s.drawbacks_for_problem else "None significant"
                    })
                suit_df = pd.DataFrame(suit_rows)
                st.dataframe(suit_df, use_container_width=True, hide_index=True)

            # Tab 4: Recommendation Card & Evidence
            with tab_rec:
                st.markdown(f"""
                <div class="recommendation-banner">
                    <div class="rec-header">Platform Recommendation</div>
                    <div class="rec-algo-title">🏆 {rec.recommended_algorithm}</div>
                    <div style="margin-bottom:1rem;">
                        <span class="rec-pill">Suitability: {rec.suitability_level.value}</span>
                        <span class="rec-pill">Platform Score: {rec.suitability_score:.1f} / 100</span>
                        <span class="rec-pill">Confidence: {rec.confidence}</span>
                    </div>
                    <div style="font-weight:600;font-size:1.05rem;margin-bottom:0.5rem;color:#f8fafc;">
                        Evidence-Based Rationale:
                    </div>
                    <ul style="margin-bottom:1rem;color:#e2e8f0;line-height:1.6;">
                        {''.join([f'<li>{r}</li>' for r in rec.reasoning])}
                    </ul>
                    <div style="border-top:1px solid rgba(255,255,255,0.25);padding-top:0.8rem;font-size:0.85rem;color:#cbd5e1;">
                        <b>Explicit Limitations:</b><br>
                        {'<br>'.join(['• ' + lim for lim in rec.limitations])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Tab 5: Simulator & Executable Objective
            with tab_vars if False else tab_sim:
                st.markdown("#### Executable Objective & Simulator Status")

                adapter = ObjectiveConnector.get_adapter(problem)
                status_info = ObjectiveConnector.get_status_message(problem, has_adapter=(adapter is not None))

                if adapter is not None:
                    st.success(f"✅ {status_info['status']}")
                    st.markdown(f"""
                    <div class="info-callout">
                        <b>{status_info['badge']}:</b> {status_info['message']}
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("##### 🧪 Optional: Run Numerical Benchmark on Physical Simulator Proxy")
                    st.markdown(
                        "Because this problem matches the **Injection Molding Process** domain, an executable physical "
                        "proxy simulator is available. You may run an actual empirical benchmark comparing all algorithms on this simulator:"
                    )

                    c_sim1, c_sim2 = st.columns([1, 2])
                    with c_sim1:
                        sim_budget = st.slider("Simulation Budget", min_value=20, max_value=120, value=60, step=10)
                        sim_seed = st.number_input("Simulator Seed", min_value=0, max_value=1000, value=42)
                        run_sim_btn = st.button("🚀 Execute Simulator Benchmark", type="primary", use_container_width=True)

                    if run_sim_btn:
                        with st.spinner("Benchmarking algorithms on physical proxy simulator..."):
                            sim_results = []
                            sim_algos = list_algorithms()
                            for algo in sim_algos:
                                res = run_algorithm(
                                    algo, adapter.evaluate, (180.0, 250.0), 4, 8, sim_budget, 0.05, sim_seed
                                )
                                sim_results.append(res)
                            sim_ranking = rank_algorithms(sim_results, global_optimum=0.8, budget=sim_budget)
                            st.session_state.sim_results = sim_results
                            st.session_state.sim_ranking_df = sim_ranking
                        st.rerun()

                    if st.session_state.sim_results is not None:
                        st.markdown("##### 📊 Simulator Benchmark Results")
                        st.dataframe(st.session_state.sim_ranking_df, use_container_width=True, hide_index=True)
                        st.plotly_chart(create_convergence_plot(st.session_state.sim_results, "Injection Molding Simulator"), use_container_width=True)
                else:
                    st.info(f"ℹ️ {status_info['status']}")
                    st.markdown(f"""
                    <div class="honest-notice">
                        <b>{status_info['badge']}</b><br><br>
                        {status_info['message']}
                    </div>
                    """, unsafe_allow_html=True)

            # Export Section for Mode A
            st.divider()
            st.markdown("### 💾 Export Analysis & Recommendation")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                csv_data = generate_mode_a_export_csv(problem, rec)
                st.download_button(
                    "📥 Download Analysis as CSV",
                    data=csv_data,
                    file_name="bo_problem_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_exp2:
                json_data = generate_mode_a_export_json(problem, char, rec)
                st.download_button(
                    "📥 Download Analysis as JSON",
                    data=json_data,
                    file_name="bo_problem_analysis.json",
                    mime="application/json",
                    use_container_width=True
                )

    # -------------------------------------------------------------------------
    # MODE B: SYNTHETIC BENCHMARK VALIDATION
    # -------------------------------------------------------------------------
    elif mode == "🔬 Benchmark Validation":
        st.markdown("### 🔬 Synthetic Benchmark Experimental Validation")
        st.markdown(
            "Validate and empirically compare Bayesian Optimization algorithms across synthetic mathematical functions. "
            "All algorithms are executed with **identical initial points, bounds, budgets, and random seeds** for rigorous scientific comparison."
        )

        # Validation check for inputs
        if run_bench_btn:
            if not selected_bench_algos:
                st.error("Please select at least one algorithm to run.")
            else:
                with st.spinner("Executing benchmark validation suite..."):
                    run_benchmark_experiment(
                        benchmark_name=bench_name,
                        dimension=int(bench_dim),
                        noise=float(bench_noise),
                        n_initial=int(bench_initial),
                        budget=int(bench_budget),
                        seed=int(bench_seed),
                        selected_algos=selected_bench_algos
                    )
                st.rerun()

        # Display Benchmark Results
        if st.session_state.bench_results is not None:
            results = st.session_state.bench_results
            ranking_df = st.session_state.bench_ranking_df
            b_char = st.session_state.bench_characteristics
            b_rec = st.session_state.bench_recommendation
            benchmark = st.session_state.bench_obj

            st.divider()

            # Benchmark Configuration Overview Cards
            bc1, bc2, bc3, bc4 = st.columns(4)
            with bc1:
                st.metric("Benchmark Function", benchmark.name, delta=f"Optimum: {benchmark.global_optimum:g}")
            with bc2:
                st.metric("Dimensions (d)", bench_dim, delta=f"Effective: ~{b_char['effective_dimension']}")
            with bc3:
                st.metric("Noise Level (σ)", f"{bench_noise:.2f}")
            with bc4:
                st.metric("Evaluation Budget", bench_budget, delta=f"Initial: {bench_initial}")

            # Empirical Recommendation Card
            st.markdown(f"""
            <div class="recommendation-banner">
                <div class="rec-header">Empirical Benchmark Winner</div>
                <div class="rec-algo-title">🏆 {b_rec['algorithm']}</div>
                <div style="margin-bottom:0.8rem;">
                    <span class="rec-pill">Score: {b_rec['score']} / 100</span>
                    <span class="rec-pill">Confidence: {b_rec['confidence']}</span>
                    <span class="rec-pill">Runtime: {b_rec['runtime']:.2f}s</span>
                </div>
                <div style="font-weight:600;font-size:1rem;color:#f8fafc;margin-bottom:0.4rem;">
                    Empirical Performance Rationale:
                </div>
                <div style="color:#e2e8f0;line-height:1.5;">
                    {b_rec['reason']}
                </div>
                <div style="border-top:1px solid rgba(255,255,255,0.2);margin-top:0.8rem;padding-top:0.6rem;font-size:0.8rem;color:#cbd5e1;">
                    {b_rec['disclaimer']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Ranking Table
            st.markdown("#### 🏅 Algorithm Performance Ranking")
            st.markdown("""
            Scores are normalized across **Best Value (40%)**, **Regret (25%)**, **Runtime (10%)**,
            **Sample Efficiency (15%)**, and **Convergence Rate (10%)**.
            """)
            st.dataframe(
                ranking_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Algorithm": st.column_config.TextColumn("Algorithm", width="medium"),
                    "Best Value": st.column_config.NumberColumn("Best Value", format="%.4f"),
                    "Score": st.column_config.NumberColumn("Score", format="%.1f")
                }
            )

            # Interactive Plotly Visualization Tabs
            st.divider()
            st.markdown("#### 📊 Interactive Performance Plots")
            t_conv, t_reg, t_score, t_time = st.tabs([
                "📈 Convergence Trajectory",
                "📉 Regret Progression",
                "🏆 Multi-Metric Scores",
                "⏱️ Runtime & Efficiency"
            ])

            with t_conv:
                st.plotly_chart(create_convergence_plot(results, benchmark.name), use_container_width=True)
            with t_reg:
                st.plotly_chart(create_regret_plot(results, benchmark.global_optimum), use_container_width=True)
            with t_score:
                st.plotly_chart(create_score_plot(ranking_df), use_container_width=True)
            with t_time:
                st.plotly_chart(create_runtime_efficiency_plot(ranking_df), use_container_width=True)

            # Export Buttons
            st.divider()
            col_b_dl1, col_b_dl2 = st.columns(2)
            with col_b_dl1:
                b_csv = generate_mode_b_export_csv(ranking_df, b_char, b_rec)
                st.download_button(
                    "📥 Download Benchmark Results (CSV)",
                    data=b_csv,
                    file_name=f"bo_benchmark_{benchmark.name}_{bench_dim}d.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_b_dl2:
                b_json = json.dumps({
                    "benchmark": benchmark.name,
                    "dimension": bench_dim,
                    "noise": bench_noise,
                    "budget": bench_budget,
                    "ranking": ranking_df.to_dict(orient="records"),
                    "recommendation": b_rec
                }, indent=2)
                st.download_button(
                    "📥 Download Benchmark Results (JSON)",
                    data=b_json,
                    file_name=f"bo_benchmark_{benchmark.name}_{bench_dim}d.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("👈 Configure benchmark parameters in the sidebar and click **RUN BENCHMARK** to execute.")

    # -------------------------------------------------------------------------
    # MODE C: SYSTEM ARCHITECTURE & SCIENTIFIC REFERENCE
    # -------------------------------------------------------------------------
    elif mode == "📖 Architecture & Reference":
        st.markdown("### 🏛️ System Architecture & Scientific Methodology")
        st.markdown("""
        The **Bayesian Optimization Algorithm Selection Platform** bridges natural language real-world problem statements
        with rigorous Bayesian Optimization theory and experimental validation.
        """)

        st.markdown("""
        #### 1. Architecture Flow
        ```
        Natural Language Input
                  │
                  ▼
        Problem Parser (problem_parser.py)
        ├─ Objective Direction & Target
        ├─ Variables, Bounds & Physical Units
        ├─ Budget, Cost & Noise Detection
        └─ Operational Constraints
                  │
                  ▼
        Missing Information Handler
        └─ Flag Unspecified Parameters Transparently (No Fabrication)
                  │
                  ▼
        Problem Characterization (characterization.py)
        ├─ Verified Facts (Dimension, Variable Types, Search Space)
        └─ Latent Properties (Modality, Smoothness: Marked as Unknown)
                  │
                  ▼
        Algorithm Suitability Evaluator (recommendation.py)
        ├─ GP-EI: Ideal for low-dim (d <= 15) expensive black-box
        ├─ GP-UCB: Robust exploration with theoretical regret bounds
        ├─ REMBO: High dimensions (d >= 25) with low effective subspace
        ├─ TuRBO: Scalable local trust regions for moderate/high dimensions
        └─ Random Search: Uniform baseline reference
                  │
                  ▼
        Recommendation Engine & Analytics Dashboard
        └─ Export CSV / JSON (In-Memory Processing)
        ```
        """)

        st.markdown("#### 2. Bayesian Optimization Algorithm Profiles")
        st.markdown("""
        | Algorithm | Surrogate Model | Acquisition / Strategy | Recommended Domain | Known Limitations |
        | :--- | :--- | :--- | :--- | :--- |
        | **GP-EI** | Gaussian Process (Matern 5/2) | Expected Improvement | Low dimensions ($d \le 15$), costly evaluations | Cubic $O(N^3)$ computational cost; over-explores boundaries in $d > 25$ |
        | **GP-UCB** | Gaussian Process (Matern 5/2) | Upper Confidence Bound | Moderate noise, regret-sensitive tasks | Requires careful tuning of exploration schedule $\\beta_t$ |
        | **REMBO** | GP in Random Embedding | Latent space $z \in \mathbb{R}^{d_e}$ | High ambient dimensions ($d \ge 25$) with low effective $d_e$ | Projection artifacts and boundary clipping on compact spaces |
        | **TuRBO** | Local Gaussian Process | Adaptive Trust Region | Scalable BO ($15 \le d \le 100+$), complex multimodal landscapes | Local model; can miss distant disjoint basins if trust region shrinks prematurely |
        | **Random Search** | None (Non-adaptive) | Uniform Random Sampling | Scientific baseline reference | Zero sample efficiency; exponential sample complexity |
        """)

        st.markdown("#### 3. Scientific Integrity Guarantees")
        st.markdown("""
        - **No Fabricated Benchmarks:** When analyzing natural language descriptions, the platform does not fabricate numerical optimization runs unless an executable simulator or objective is explicitly supplied.
        - **No Assumed Noise or Bounds:** Missing parameters are explicitly surfaced to the user as `Unknown`.
        - **In-Memory Architecture:** No external databases, user credentials, or permanent tracking are employed.
        """)


if __name__ == "__main__":
    main()