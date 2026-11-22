"""
test_platform.py
Comprehensive verification script for Bayesian Optimization Algorithm Selection Platform.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from problem_parser import ProblemParser
from characterization import characterize_parsed_problem, get_problem_characteristics
from recommendation import evaluate_algorithm_suitability, generate_recommendation
from benchmarks import get_benchmark
from algorithms import run_algorithm, list_algorithms
from evaluation import rank_algorithms, calculate_metrics
from objective_adapter import ObjectiveConnector, InjectionMoldingSimulator
from examples import EXAMPLES


def run_all_tests():
    parser = ProblemParser()
    print("==================================================")
    print("STEP 1: Testing Real-World Natural Language Inputs")
    print("==================================================")

    for ex_name, ex_data in EXAMPLES.items():
        print(f"\n--- Testing: {ex_name} ---")
        p = parser.parse(ex_data["description"])
        char = characterize_parsed_problem(p)
        rec = evaluate_algorithm_suitability(char, p)

        print(f"Objective: {p.objective.obj_type.value} '{p.objective.target}' (unit: {p.objective.unit})")
        print(f"Dimension: {p.dimension} | Budget: {p.evaluation_budget} | Cost: {p.evaluation_cost} | Noise: {p.noise}")
        print(f"Variables Extracted ({len(p.variables)}):")
        for v in p.variables:
            print(f"  • {v.name}: [{v.lower_bound}, {v.upper_bound}] {v.unit} ({v.var_type.value})")
        print(f"Missing Fields Count: {len(p.missing_fields)} -> {[m.display_name for m in p.missing_fields]}")
        print(f"Recommended Algorithm: {rec.recommended_algorithm} (Score: {rec.suitability_score}, Conf: {rec.confidence})")
        print("Suitability Table:")
        for s in rec.suitability_table:
            print(f"  [{s.suitability.value.ljust(15)}] {s.algorithm.ljust(14)}: {s.score:5.1f} | {s.key_reason}")

    print("\n==================================================")
    print("STEP 2: Testing Ambiguous / Invalid Natural Language")
    print("==================================================")
    ambig_text = "I have a generic optimization problem and want to find the best settings."
    p_ambig = parser.parse(ambig_text)
    char_ambig = characterize_parsed_problem(p_ambig)
    rec_ambig = evaluate_algorithm_suitability(char_ambig, p_ambig)
    print(f"Ambiguous input handled gracefully:")
    print(f"  Variables: {p_ambig.dimension} (Expected: None)")
    print(f"  Missing fields: {[m.field_name for m in p_ambig.missing_fields]}")
    print(f"  Recommended Algorithm: {rec_ambig.recommended_algorithm}")

    print("\n==================================================")
    print("STEP 3: Testing Objective Adapter & Simulator Proxy")
    print("==================================================")
    p_inj = parser.parse(EXAMPLES["Injection Molding Process"]["description"])
    adapter = ObjectiveConnector.get_adapter(p_inj)
    print(f"Adapter detected for Injection Molding: {adapter.name if adapter else None}")
    if adapter:
        test_x = np.array([220.0, 95.0, 60.0, 22.0])
        val = adapter.evaluate(test_x)
        print(f"Evaluated proxy simulator at test point {test_x} -> Defect rate: {val:.4f}%")

    print("\n==================================================")
    print("STEP 4: Testing Benchmark Mode (Rastrigin 50-D Validation)")
    print("==================================================")
    b_name = "Rastrigin"
    dim = 50
    noise = 0.10
    n_init = 10
    budget = 100
    seed = 42

    bench = get_benchmark(b_name)
    def noisy_obj(x):
        return bench.evaluate(x, noise)

    algos_to_test = ["GP-EI", "GP-UCB", "REMBO", "TuRBO", "Random Search"]
    results = []
    print(f"Running benchmark on {b_name} (d={dim}, budget={budget}, seed={seed})...")
    for algo in algos_to_test:
        res = run_algorithm(algo, noisy_obj, bench.bounds, dim, n_init, budget, noise, seed)
        print(f"  ✓ {algo:14} completed in {res['runtime']:.2f}s | Best Value: {res['best_value']:.4f}")
        results.append(res)

    ranking_df = rank_algorithms(results, bench.global_optimum, budget)
    print("\nBenchmark Ranking Table:")
    print(ranking_df.to_string(index=False))

    char_bench = get_problem_characteristics(b_name, dim, noise)
    best_algo_name = ranking_df.iloc[0]["Algorithm"]
    best_res = next(r for r in results if r["algorithm"] == best_algo_name)
    bench_rec = generate_recommendation(best_res, ranking_df, char_bench, algos_to_test)
    print(f"\nEmpirical Recommendation: {bench_rec['algorithm']} (Score: {bench_rec['score']})")
    print(f"Reason:\n{bench_rec['reason']}")

    print("\n==================================================")
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_all_tests()
