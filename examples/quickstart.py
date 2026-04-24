import pandas as pd
import numpy as np
from anomaly_detector.detector import AnomalyDetector

# 1. Create a "Stress Test" Dataset (103 base rows)
data = {
    "gender": (["Male"] * 50 + ["Female"] * 50 + ["Femlae"] * 2 + ["M"] * 1),
    "notes": ([None] * 63 + ["verified"] * 40),
    
    # [NEW] Flags Section 5: Mixed int and str
    "reference_id": [101, 102, "PENDING", 104, 105] + ([999] * 98),
    
    # Flags Section 4: Statistical Outliers
    "salary": [50000, 52000, 48000, 51000, 49000, 999999, -50000] + ([50000] * 96),
    
    # Flags Section 5: Mixed types (int and str)
    "age": [25, 30, "thirty-five", 40, 22] + ([30] * 98),
    
    # Flags Section 6: Values violating the 0-100 logic
    "score": [85, 90, 150, -10, 88] + ([90] * 98)
}

df = pd.DataFrame(data)

# [NEW] Flags Section 3: Duplicate first 3 rows
df = pd.concat([df, df.iloc[[0, 1, 2]]], ignore_index=True)

# 2. Define Scalable Logical Rules
my_rules = {
    "score": {"min": 0, "max": 100},
    "age": {"min": 18, "max": 100},
    "reference_id": {"regex": r'^\d+$'} # Flags anything that isn't purely digits
}

# 3. Initialize and Run
detector = AnomalyDetector(
    df,
    rare_threshold=3,
    null_threshold_pct=10.0,
    logical_rules=my_rules
)

report = detector.run()

# 4. Final Output
print(report.summary())
