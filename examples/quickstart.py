"""
examples/quickstart.py
----------------------
A self-contained demo of StatGuard's rare-value detection.
Run from the repo root:

    python examples/quickstart.py
"""

import pandas as pd
from statguard import AnomalyDetector

# -------------------------------------------------------------------
# Build a realistic-ish dataset with some deliberate dirty values
# -------------------------------------------------------------------
import random
random.seed(42)

n = 200

genders    = (["Male"] * 90  + ["Female"] * 95 + ["Femlae"] * 3 + ["M"] * 2 + ["male"] * 10)[:n]
countries  = (["USA"]  * 110 + ["UK"] * 40 + ["Canada"] * 30 + ["Canda"] * 1 + ["ZZ"] * 1 + ["??"] * 2 + ["Australia"] * 16)[:n]
statuses   = (["active"] * 165 + ["inactive"] * 28 + ["ACTIVE"] * 2 + ["pending"] * 5)[:n]
ages       = [random.randint(18, 80) for _ in range(n)]   # numeric — high cardinality, skipped

df = pd.DataFrame(
    {
        "gender":  genders[:n],
        "country": countries[:n],
        "status":  statuses[:n],
        "age":     ages,
    }
)

print(f"Dataset shape: {df.shape}\n")

# -------------------------------------------------------------------
# Run the detector
# -------------------------------------------------------------------
detector = AnomalyDetector(df, rare_threshold=5)
report   = detector.run()

# Pretty-print the summary
print(report)

# -------------------------------------------------------------------
# Access results programmatically
# -------------------------------------------------------------------
print("\n--- Rare-value findings as a DataFrame ---")
print(report.rare_values_df().to_string(index=False))

# Export to CSV
report.to_csv("/tmp/statguard_rare_values.csv")
