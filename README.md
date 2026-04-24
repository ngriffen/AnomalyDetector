
# AnomalyDetector 🔍

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**AnomalyDetector** is a lightweight, modular Python suite designed to automate data auditing in pandas DataFrames. By shifting quality control to the earliest stage of the pipeline, it enables data scientists and analysts to catch statistical outliers, type inconsistencies, and logical rule violations before they corrupt downstream models or insights.

---

## 🚀 Features

- **Rare Value Detection**: Surfaces potential typos or uncommon categories by identifying values that fall below a specific frequency threshold.
- **Null Rate Analysis**: Automatically flags columns where missing data exceeds a user-defined percentage, highlighting gaps in data collection.
- **Duplicate Detection**: Identifies duplicate rows across the entire dataset or specific subsets, providing the exact row indices and attributes of the clones.
- **Statistical Outlier Detection**: Uses the Interquartile Range (IQR) method to mathematically identify "extreme" numerical values that could skew results.
- **Type Inconsistency Checking**: Flags "dirty" columns containing mixed Python types (e.g., strings inside numeric columns) that often cause pipeline crashes.
- **Scalable Logical Validation**: Executes custom, rule-based audits—such as range limits, regex patterns, and allowed-value lists—tailored to your specific business logic.
- **Consolidated Reporting**: Returns a clean ` DATA QUALITY AUDIT REPORT` object containing all detected findings.


---

## 📦 Installation

To install the package in development mode, clone the repository and run:

```bash
git clone [https://github.com/ngriffen/AnomalyDetector.git](https://github.com/ngriffen/AnomalyDetector.git)
cd AnomalyDetector
pip install -e .
