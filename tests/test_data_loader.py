"""
Unit tests for data_loader module.

These tests use synthetic data to validate loading, cleaning,
and validation logic without requiring the actual OULAD dataset.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import (
    clean_student_info,
    clean_student_vle,
    clean_assessments,
    get_student_keys,
    summarize_dataset,
)


class TestCleanStudentInfo(unittest.TestCase):
    """Tests for clean_student_info function."""

    def setUp(self):
        self.df = pd.DataFrame({
            "code_module": ["AAA", "AAA", "BBB", "BBB"],
            "code_presentation": ["2013J", "2013J", "2014B", "2014B"],
            "id_student": [1, 2, 3, 4],
            "gender": ["M", "F", "M", "F"],
            "region": ["London", "Scotland", "London", "Wales"],
            "highest_education": [
                "A Level or Equivalent", "HE Qualification",
                "Lower Than A Level", "Post Graduate Qualification",
            ],
            "imd_band": ["10-20%", None, "30-40%", "50-60%"],
            "age_band": ["0-35", "35-55", "0-35", "55<="],
            "num_of_prev_attempts": [0, 1, 0, 2],
            "studied_credits": [60, 120, 60, 240],
            "disability": ["N", "N", "Y", "N"],
            "final_result": ["Pass", "Withdrawn", "Fail", "Distinction"],
        })

    def test_dropout_label(self):
        result = clean_student_info(self.df)
        expected = [0, 1, 1, 0]
        self.assertEqual(result["is_dropout"].tolist(), expected)

    def test_completion_status(self):
        result = clean_student_info(self.df)
        expected = [1, 0, 0, 1]
        self.assertEqual(result["completion_status"].tolist(), expected)

    def test_missing_imd_filled(self):
        result = clean_student_info(self.df)
        self.assertFalse(result["imd_band"].isnull().any())
        self.assertEqual(result["imd_band"].iloc[1], "Unknown")

    def test_age_band_numeric(self):
        result = clean_student_info(self.df)
        expected = [0, 1, 0, 2]
        self.assertEqual(result["age_band_numeric"].tolist(), expected)

    def test_education_numeric(self):
        result = clean_student_info(self.df)
        expected = [2, 3, 1, 4]
        self.assertEqual(result["education_numeric"].tolist(), expected)


class TestCleanStudentVle(unittest.TestCase):
    """Tests for clean_student_vle function."""

    def setUp(self):
        self.df = pd.DataFrame({
            "code_module": ["AAA"] * 5,
            "code_presentation": ["2013J"] * 5,
            "id_student": [1, 1, 2, 2, 3],
            "id_site": [100, 101, 100, 102, 100],
            "date": [1, 5, 1, 10, "NA"],
            "sum_click": [5, 0, 3, 10, 2],
        })

    def test_zero_clicks_removed(self):
        result = clean_student_vle(self.df)
        self.assertTrue((result["clicks"] > 0).all())

    def test_column_renamed(self):
        result = clean_student_vle(self.df)
        self.assertIn("clicks", result.columns)
        self.assertNotIn("sum_click", result.columns)

    def test_invalid_date_removed(self):
        result = clean_student_vle(self.df)
        self.assertTrue(result["date"].dtype in [np.int64, np.int32, int])


class TestCleanAssessments(unittest.TestCase):
    """Tests for clean_assessments function."""

    def test_score_capped_at_100(self):
        sa = pd.DataFrame({
            "id_assessment": [1, 2, 3],
            "id_student": [10, 20, 30],
            "date_submitted": [5, 10, 15],
            "is_banked": [0, 0, 1],
            "score": [85, 120, None],
        })
        a = pd.DataFrame({
            "code_module": ["A", "A", "A"],
            "code_presentation": ["2013J"] * 3,
            "id_assessment": [1, 2, 3],
            "assessment_type": ["TMA", "TMA", "CMA"],
            "date": [10, 20, None],
            "weight": [25, 25, 50],
        })

        a_clean, sa_clean = clean_assessments(a, sa)
        self.assertTrue((sa_clean["score"] <= 100).all())
        self.assertFalse(sa_clean["score"].isnull().any())
        self.assertFalse(a_clean["date"].isnull().any())


class TestSummarizeDataset(unittest.TestCase):
    """Tests for summarize_dataset function."""

    def test_summary_structure(self):
        tables = {
            "t1": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            "t2": pd.DataFrame({"x": [1, None, 3]}),
        }
        summary = summarize_dataset(tables)
        self.assertEqual(len(summary), 2)
        self.assertIn("rows", summary.columns)
        self.assertIn("missing_pct", summary.columns)


if __name__ == "__main__":
    unittest.main()
