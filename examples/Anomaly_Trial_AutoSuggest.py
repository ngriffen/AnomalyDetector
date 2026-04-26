!pip install --no-cache-dir git+https://github.com/ngriffen/AnomalyDetector.git
!pip install scikit-learn

import pandas as pd
import numpy as np
import json
from anomaly_detector.detector import AnomalyDetector

# --- 1. SETUP DATA ---
data = {
    "gender": (["Male"] * 50 + ["Female"] * 50 + ["F"] * 2 + ["Unk"] * 1),
    "middle_name": ([None] * 63 + ["Lee"] * 40),
    "salary": [55000, 58000, 60000, 57000, 56000, 2000000, -500] + ([55000] * 96),
    "tenure_years": [5, 10, "2 years", 8, 4] + ([5] * 98),
    "performance_score": [85, 92, 150, -5, 80] + ([85] * 98),
    "age": [25, 26, 24, 25, 26, 25, 20, 70] + ([30] * 95),
}
df = pd.DataFrame(data)
df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)

# --- 2. THE "SUGGESTION" STEP ---
print("🔍 STEP 1: Analyzing data to suggest optimal thresholds...")
# This automatically finds the 5th/95th percentiles and dominant categories
suggested_config = AnomalyDetector.suggest_config(df)

# Print the suggested config in a pretty format so the user can inspect it
print("\n💡 SUGGESTED CONFIGURATION:")
print(json.dumps(suggested_config, indent=4))

# --- 3. THE "REFINEMENT" STEP ---
print("\n🛠️  STEP 2: Refining thresholds based on business knowledge...")

# Example: The suggested statistical range for performance_score, 
# but we know the strict legal limit is 0-100. Let's override it:
suggested_config["logical_rules"]["performance_score"] = {
    "min": 0, 
    "max": 100,
    "info": "Strict business limit: 0 to 100"
}

# Example: We want to be more aggressive with rare value detection
suggested_config["global_thresholds"]["rare_threshold"] = 5

# --- 4. THE "AUDIT" STEP ---
print("\n🚀 STEP 3: Running FULL MODE Audit with refined settings...")

# We pass the 'global_thresholds' as kwargs and the rest as the config dict
# Note: Ensure your detector.py is updated to parse the new nested dict structure!
detector = AnomalyDetector.Full(
    df, 
    **suggested_config["global_thresholds"], 
    logical_rules=suggested_config["logical_rules"],
    patterns=suggested_config["patterns"]
)

report = detector.run()

# --- 5. RESULTS ---
print("\n" + report.summary())

# Display the visual dashboard
report.visualize()
