"""
Tests for rare-value detection.
"""

import pandas as pd
import pytest

from statguard import AnomalyDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df():
    """Small DataFrame with deliberate rare values."""
    return pd.DataFrame(
        {
            "gender": (
                ["Male"] * 45 + ["Female"] * 48 + ["Femlae"] * 2 + ["Unknown"] * 5
            ),
            "country": (
                ["USA"] * 60 + ["UK"] * 20 + ["CA"] * 15 + ["ZZ"] * 1 + ["?"] * 1 + ["AU"] * 3
            ),
            "score": list(range(100)),       # numeric, high-cardinality — should be skipped
            "status": ["active"] * 98 + ["ACTIVE"] * 1 + ["inactive"] * 1,
        }
    )


# ---------------------------------------------------------------------------
# Basic detection
# ---------------------------------------------------------------------------

class TestRareValues:
    def test_flags_typo(self, sample_df):
        report = AnomalyDetector(sample_df, rare_threshold=5).run()
        flagged_values = [r["value"] for r in report.rare_values]
        assert "Femlae" in flagged_values, "'Femlae' typo should be flagged"

    def test_flags_low_count_category(self, sample_df):
        report = AnomalyDetector(sample_df, rare_threshold=5).run()
        flagged_values = [r["value"] for r in report.rare_values]
        assert "ZZ" in flagged_values
        assert "?" in flagged_values

    def test_does_not_flag_common_values(self, sample_df):
        report = AnomalyDetector(sample_df, rare_threshold=5).run()
        flagged_values = [r["value"] for r in report.rare_values]
        assert "Male" not in flagged_values
        assert "USA" not in flagged_values

    def test_skips_high_cardinality_column(self, sample_df):
        """The 'score' column has 100 unique values and should be ignored."""
        report = AnomalyDetector(sample_df, rare_threshold=5, rare_max_categories=50).run()
        flagged_cols = {r["column"] for r in report.rare_values}
        assert "score" not in flagged_cols

    def test_threshold_respected(self, sample_df):
        """With threshold=1 only values with count==1 should be flagged."""
        report = AnomalyDetector(sample_df, rare_threshold=1).run()
        flagged_values = [r["value"] for r in report.rare_values]
        assert "?" in flagged_values
        # 'Unknown' appears 5 times → should NOT be flagged at threshold=1
        assert "Unknown" not in flagged_values

    def test_count_and_pct_correct(self, sample_df):
        report = AnomalyDetector(sample_df, rare_threshold=5).run()
        zz = next(r for r in report.rare_values if r["value"] == "ZZ")
        assert zz["count"] == 1
        assert zz["pct"] == pytest.approx(1 / 100 * 100, rel=1e-2)


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

class TestReport:
    def test_rare_values_df_columns(self, sample_df):
        report = AnomalyDetector(sample_df).run()
        df = report.rare_values_df()
        expected_cols = {"column", "value", "count", "pct", "total_rows", "unique_vals"}
        assert expected_cols.issubset(set(df.columns))

    def test_summary_contains_flagged_column(self, sample_df):
        report = AnomalyDetector(sample_df).run()
        summary = report.summary()
        assert "gender" in summary
        assert "Femlae" in summary

    def test_empty_report_for_clean_data(self):
        clean = pd.DataFrame(
            {"color": ["red"] * 50 + ["blue"] * 50}
        )
        report = AnomalyDetector(clean, rare_threshold=3).run()
        assert report.rare_values == []
        assert "No rare values detected" in report.summary()

    def test_repr(self, sample_df):
        report = AnomalyDetector(sample_df).run()
        assert "AnomalyReport" in repr(report)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_raises_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            AnomalyDetector(pd.DataFrame())

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            AnomalyDetector([[1, 2], [3, 4]])

    def test_all_null_column_ignored(self):
        df = pd.DataFrame({"a": [None, None, None], "b": ["x"] * 3})
        report = AnomalyDetector(df, rare_threshold=5).run()
        flagged_cols = {r["column"] for r in report.rare_values}
        assert "a" not in flagged_cols

    def test_single_row_dataframe(self):
        df = pd.DataFrame({"x": ["only_one"]})
        report = AnomalyDetector(df, rare_threshold=5).run()
        assert any(r["value"] == "only_one" for r in report.rare_values)
