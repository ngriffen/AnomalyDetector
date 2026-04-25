#Install from GitHub
!pip install --no-cache-dir git+https://github.com/ngriffen/AnomalyDetector.git
!pip install scikit-learn

import pandas as pd
import numpy as np
from anomaly_detector.detector import AnomalyDetector

# 1. Setup a more complex "Real-World" Stress Dataset
data = {
    # BASIC: 'F' and 'Unk' are rare
    "gender": (["Male"] * 50 + ["Female"] * 50 + ["F"] * 2 + ["Unk"] * 1),

    # BASIC: ~60% Nulls
    "middle_name": ([None] * 63 + ["Lee"] * 40),

    # BASIC: Statistical Outliers (IQR)
    "salary": [55000, 58000, 60000, 57000, 56000, 2000000, -500] + ([55000] * 96),

    # BASIC: Mixed types
    "tenure_years": [5, 10, "2 years", 8, 4] + ([5] * 98),

    # BASIC: Logical Violations (Rule: 0-100)
    "performance_score": [85, 92, 150, -5, 80] + ([85] * 98),

    # AUTO: These rows are designed to be "Multivariate Anomalies"
    # Row 7 and 8 have normal ages and normal salaries,
    # but the COMBINATION is statistically weird for the rest of the group.
    "age": [25, 26, 24, 25, 26, 25, 20, 70] + ([30] * 95),
}

df = pd.DataFrame(data)

# BASIC: Create 2 Duplicate Rows
df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)

# 2. Configure Rules for Basic/Full modes
config = {
    "rare_threshold": 3,
    "null_threshold_pct": 15.0,
    "logical_rules": {
        "performance_score": {"min": 0, "max": 100},
        "salary": {"min": 0}
    },
    "patterns": {
        "gender": r"^(Male|Female)$" # Enforce full words only
    }
}

#BASIC MODE
print("RUNNING STEP 1: BASIC MODE")
# Only runs the 6 Basic statistical checks
basic_report = AnomalyDetector.Basic(df, **config).run()
print(basic_report.summary())

#AUTO MODE
print("\n\nRUNNING STEP 2: AUTO MODE")
# Uses ML (Isolation Forest) to find weird row combinations
# We set contamination higher to ensure it catches the age/salary combo anomalies
auto_report = AnomalyDetector.Auto(df, contamination=0.05).run()
print(auto_report.summary())

#FULL MODE
print("\n\nRUNNING STEP 3: FULL MODE")
# Runs EVERYTHING: Rules + Statistics + Machine Learning
full_report = AnomalyDetector.Full(df, **config).run()
print(full_report.summary())
