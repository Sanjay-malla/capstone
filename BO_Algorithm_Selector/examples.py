"""
examples.py
Pre-configured real-world optimization problem templates for the
Bayesian Optimization Algorithm Selection Platform.
"""

from typing import Dict, Any, List

EXAMPLES: Dict[str, Dict[str, Any]] = {
    "Injection Molding Process": {
        "title": "Injection Molding Defect Minimization",
        "category": "Manufacturing / Industrial",
        "description": (
            "I operate an injection molding machine and want to minimize the percentage of defective parts. "
            "The injection temperature can range from 180 to 250 degrees Celsius, pressure from 50 to 120 MPa, "
            "injection speed from 20 to 100 mm/s, and cooling time from 5 to 30 seconds. "
            "Each experiment costs approximately ₹5,000 and I can perform 100 experiments."
        ),
        "expected_dim": 4,
        "notes": "4 continuous parameters, expensive physical testing, unobserved noise."
    },
    "ML Hyperparameter Optimization": {
        "title": "Deep Learning Hyperparameter Tuning",
        "category": "Machine Learning / AI",
        "description": (
            "I want to maximize the validation accuracy of a machine learning model. "
            "I can tune learning rate from 0.0001 to 0.1, batch size from 16 to 256, "
            "dropout from 0 to 0.5, and number of layers from 2 to 8. "
            "Training one model takes around 20 minutes and I can run 80 experiments."
        ),
        "expected_dim": 4,
        "notes": "Mixed continuous (lr, dropout) and integer (batch size, layers), high time cost."
    },
    "Robot Arm Trajectory": {
        "title": "Robot Arm Trajectory & Speed",
        "category": "Robotics & Control",
        "description": (
            "I want to minimize the movement time of a robot arm while maintaining "
            "positioning accuracy above 99%. I can modify motor speed, acceleration, "
            "and trajectory parameters within predefined operating ranges. Each simulation takes several minutes."
        ),
        "expected_dim": 3,
        "notes": "Quality constraint (accuracy > 99%), missing numerical bounds (triggers missing info handler)."
    },
    "HVAC Building Energy": {
        "title": "Commercial HVAC Energy Minimization",
        "category": "Energy & Buildings",
        "description": (
            "I want to minimize the daily energy consumption of a commercial HVAC system. "
            "Chilled water supply temperature ranges from 44 to 54 degrees Fahrenheit, "
            "supply airflow rate from 1000 to 5000 CFM, and static pressure from 1.0 to 2.5 inches of water. "
            "I have a budget of 60 operational test hours."
        ),
        "expected_dim": 3,
        "notes": "Continuous physical parameters, moderate evaluation budget."
    },
    "EV Battery Range Optimization": {
        "title": "Electric Vehicle Thermal & Range Tuning",
        "category": "Automotive / EV",
        "description": (
            "I want to maximize the driving range of an electric vehicle in cold conditions. "
            "Battery coolant flow rate ranges from 5 to 25 liters per minute, "
            "cell preheat temperature from 15 to 35 degrees Celsius, "
            "and cabin climate setpoint from 18 to 24 degrees Celsius. "
            "Each dynamometer run costs $500 and the test budget is 50 runs."
        ),
        "expected_dim": 3,
        "notes": "3 continuous thermal parameters, monetary cost per test, compact budget."
    }
}


def list_example_names() -> List[str]:
    return list(EXAMPLES.keys())


def get_example(name: str) -> Dict[str, Any]:
    if name not in EXAMPLES:
        raise ValueError(f"Unknown example: {name}")
    return EXAMPLES[name]
