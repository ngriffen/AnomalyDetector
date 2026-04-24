
# AnomalyDetector 🔍

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**AnomalyDetector** is a lightweight, modular Python suite designed to flag data quality issues in pandas DataFrames. It helps data scientists and analysts quickly identify rare categorical values, high null rates, and duplicate records before moving into the modeling phase.

---

## 🚀 Features

- **Rare Value Detection**: Identify values in categorical columns that appear below a specific frequency threshold.
- **Null Rate Analysis**: Flag columns where the percentage of missing data exceeds a user-defined limit.
- **Duplicate Detection**: Scan for redundant rows across the entire dataset or specific column subsets.
- **Consolidated Reporting**: Returns a clean `AnomalyReport` object containing all detected findings.

---

## 📦 Installation

To install the package in development mode, clone the repository and run:

```bash
git clone [https://github.com/ngriffen/AnomalyDetector.git](https://github.com/ngriffen/AnomalyDetector.git)
cd AnomalyDetector
pip install -e .
