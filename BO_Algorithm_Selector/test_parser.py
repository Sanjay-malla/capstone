"""
test_parser.py
Unit tests for ProblemParser and ProblemSchema
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from problem_parser import ProblemParser

def test_examples():
    parser = ProblemParser()

    print("--- Test 1: Injection Molding ---")
    t1 = ("I operate an injection molding machine and want to minimize the percentage of "
          "defective parts. The injection temperature can range from 180 to 250 degrees Celsius, "
          "pressure from 50 to 120 MPa, injection speed from 20 to 100 mm/s, and cooling time "
          "from 5 to 30 seconds. Each experiment costs approximately ₹5,000 and I can perform 100 experiments.")
    p1 = parser.parse(t1)
    print("Objective:", p1.objective.to_dict())
    print("Dimension:", p1.dimension)
    for v in p1.variables:
        print(f"  Var: {v.name} -> [{v.lower_bound}, {v.upper_bound}] {v.unit} ({v.var_type.value})")
    print("Budget:", p1.evaluation_budget)
    print("Cost:", p1.evaluation_cost)
    print("Expensive:", p1.expensive_evaluation)
    print("Noise:", p1.noise)
    print("Missing count:", len(p1.missing_fields))

    print("\n--- Test 2: ML Hyperparameter Optimization ---")
    t2 = ("I want to maximize the validation accuracy of a machine learning model. "
          "I can tune learning rate from 0.0001 to 0.1, batch size from 16 to 256, "
          "dropout from 0 to 0.5, and number of layers from 2 to 8. "
          "Training one model takes around 20 minutes and I can run 80 experiments.")
    p2 = parser.parse(t2)
    print("Objective:", p2.objective.to_dict())
    print("Dimension:", p2.dimension)
    for v in p2.variables:
        print(f"  Var: {v.name} -> [{v.lower_bound}, {v.upper_bound}] {v.unit} ({v.var_type.value})")
    print("Budget:", p2.evaluation_budget)
    print("Cost:", p2.evaluation_cost)
    print("Expensive:", p2.expensive_evaluation)
    print("Noise:", p2.noise)

    print("\n--- Test 3: Robot Arm ---")
    t3 = ("I want to minimize the movement time of a robot arm while maintaining "
          "positioning accuracy above 99%. I can modify motor speed, acceleration, "
          "and trajectory parameters within predefined operating ranges. Each simulation takes several minutes.")
    p3 = parser.parse(t3)
    print("Objective:", p3.objective.to_dict())
    print("Dimension:", p3.dimension)
    print("Constraints:", p3.constraints)
    for v in p3.variables:
        print(f"  Var: {v.name} -> [{v.lower_bound}, {v.upper_bound}] {v.unit} ({v.var_type.value})")
    print("Cost:", p3.evaluation_cost)
    print("Missing fields:", [m.display_name for m in p3.missing_fields])

if __name__ == "__main__":
    test_examples()
