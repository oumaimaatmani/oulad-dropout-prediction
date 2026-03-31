"""
Unit tests for feature_builder module.

Uses synthetic data to validate feature engineering logic.
"""

import os
import unittest

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feature_builder import (
    build_registration_features,
    _compute_activity_diversity,
)


class TestRegistrationFeatures(unittest.TestCase):
    """Tests for registration feature construction."""

    def setUp(self):
        self.df = pd.DataFrame({
            "code_module": ["AAA", "AAA", "BBB"],
            "code_presentation": ["2013J", "2013J", "2014B"],
            "id_student": [1, 2, 3],
            "date_registration": [-10, 0, 5],
            "date_unregistration": [None, 50, None],
        })

    def test_registration_date(self):
        result = build_registration_features(self.df)
        self.assertEqual(result["registration_date"].iloc[0], -10)
        self.assertEqual(result["registration_date"].iloc[2], 5)

    def test_unregistration_flag(self):
        result = build_registration_features(self.df)
        expected = [0, 1, 0]
        self.assertEqual(result["has_unregistered"].tolist(), expected)


class TestActivityDiversity(unittest.TestCase):
    """Tests for Shannon entropy computation."""

    def test_single_activity_zero_entropy(self):
        df = pd.DataFrame({
            "code_module": ["A"] * 3,
            "code_presentation": ["X"] * 3,
            "id_student": [1] * 3,
            "activity_type": ["forumng"] * 3,
            "clicks": [10, 20, 30],
        })
        key_cols = ["code_module", "code_presentation", "id_student"]
        result = _compute_activity_diversity(df, key_cols)
        self.assertAlmostEqual(result["activity_diversity"].iloc[0], 0.0)

    def test_two_equal_activities(self):
        df = pd.DataFrame({
            "code_module": ["A", "A"],
            "code_presentation": ["X", "X"],
            "id_student": [1, 1],
            "activity_type": ["forumng", "quiz"],
            "clicks": [50, 50],
        })
        key_cols = ["code_module", "code_presentation", "id_student"]
        result = _compute_activity_diversity(df, key_cols)
        self.assertAlmostEqual(result["activity_diversity"].iloc[0], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
