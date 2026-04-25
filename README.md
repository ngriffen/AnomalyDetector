
# AnomalyDetector 🔍

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**AnomalyDetector** is a lightweight, modular Python suite designed to automate data auditing in pandas DataFrames. By shifting quality control to the earliest stage of the pipeline, it enables data scientists and analysts to catch statistical outliers, type inconsistencies, and logical rule violations before they corrupt downstream models or insights. Returns an `ANOMALY REPORT` object containing all detected findings.

---

## Basic Features

- **Rare Value Detection**: Surfaces potential typos or uncommon categories by identifying values that fall below a specific frequency threshold.
- **Null Rate Analysis**: Automatically flags columns where missing data exceeds a user-defined percentage, highlighting gaps in data collection.
- **Duplicate Detection**: Identifies duplicate rows across the entire dataset or specific subsets, providing the exact row indices and attributes of the clones.
- **Statistical Outlier Detection**: Uses the Interquartile Range (IQR) method to mathematically identify "extreme" numerical values that could skew results.
- **Type Inconsistency Checking**: Flags "dirty" columns containing mixed Python types (e.g., strings inside numeric columns) that often cause pipeline crashes.
- **Scalable Logical Validation**: Executes custom, rule-based audits—such as range limits, regex patterns, and allowed-value lists—tailored to your specific business logic.

## Auto Mode (Machine Learning)
The Auto Mode moves beyond simple row-by-row rules to analyze the "hidden" relationships within your data using unsupervised machine learning.

- **Unsupervised Multivariate Detection**: Powered by the Isolation Forest algorithm, this mode detects anomalies that traditional rules miss. It identifies outliers where individual values might look normal, but their combination is statistically rare.

- **Root Cause Attribution**: Unlike standard "black box" ML, the engine calculates feature importance via deviance scoring to pinpoint exactly which columns caused a row to be flagged.

- **Complex Interaction Tagging**: The report distinguishes between Specific Suspects (single column outliers) and Complex Interactions (rows where the relationship between multiple features is the primary anomaly).

- **Zero-Config Intelligence**: Requires no manual rules or thresholds. The engine automatically learns the "shape" of your unique dataset to separate signal from noise.

---

## Mode Overview

| Mode  | Command                      | Function |
|-------|-----------------------------|----------|
| Basic | `AnomalyDetector.Basic(df)` | Catching typos, nulls, duplicates, and explicit rule violations. |
| Auto  | `AnomalyDetector.Auto(df)`  | Finding "weird" data patterns using unsupervised machine learning. |
| Full  | `AnomalyDetector.Full(df)`  | Both Basic and Auto for full anomaly detection capabilities. |

---

## Installation

To install the package in development mode, clone the repository and run:

```bash
git clone [https://github.com/ngriffen/AnomalyDetector.git](https://github.com/ngriffen/AnomalyDetector.git)
cd AnomalyDetector
pip install -e .
