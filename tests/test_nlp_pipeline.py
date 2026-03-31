"""
Unit tests for nlp_pipeline module.

Tests text generation and forum feature construction.
Sentiment classification tests are skipped if transformers is not installed.
"""

import os
import unittest

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nlp_pipeline import (
    student_to_text,
    generate_student_descriptions,
    build_forum_features,
)


class TestStudentToText(unittest.TestCase):
    """Tests for natural language description generation."""

    def test_high_engagement_student(self):
        row = pd.Series({
            "module_engagement_rate": 2.0,
            "learning_pace": -10,
            "average_score": 85,
            "completed_assessments_ratio": 0.95,
            "forum_engagement_ratio": 0.2,
        })
        stats = {"mean_engagement": 1.0, "mean_pace": 0, "mean_score": 50}
        text = student_to_text(row, stats)

        self.assertIn("very high", text)
        self.assertIn("ahead of schedule", text)
        self.assertIn("excellent", text)
        self.assertIn("nearly all assessments", text)
        self.assertIn("active participation", text)

    def test_low_engagement_student(self):
        row = pd.Series({
            "module_engagement_rate": 0.1,
            "learning_pace": 25,
            "average_score": 20,
            "completed_assessments_ratio": 0.1,
            "forum_engagement_ratio": 0.0,
        })
        stats = {"mean_engagement": 1.0, "mean_pace": 0, "mean_score": 50}
        text = student_to_text(row, stats)

        self.assertIn("minimal", text)
        self.assertIn("behind schedule", text)
        self.assertIn("poor", text)
        self.assertIn("few or no", text)
        self.assertIn("little to no", text)

    def test_output_is_string(self):
        row = pd.Series({
            "module_engagement_rate": 1.0,
            "learning_pace": 0,
            "average_score": 60,
            "completed_assessments_ratio": 0.5,
            "forum_engagement_ratio": 0.05,
        })
        stats = {"mean_engagement": 1.0, "mean_pace": 0, "mean_score": 50}
        text = student_to_text(row, stats)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 50)


class TestGenerateDescriptions(unittest.TestCase):
    """Tests for batch description generation."""

    def test_generates_correct_count(self):
        df = pd.DataFrame({
            "module_engagement_rate": [1.0, 0.5, 2.0],
            "learning_pace": [0, 5, -3],
            "average_score": [70, 40, 90],
            "completed_assessments_ratio": [0.8, 0.3, 1.0],
            "forum_engagement_ratio": [0.1, 0.0, 0.2],
        })
        descs = generate_student_descriptions(df)
        self.assertEqual(len(descs), 3)
        self.assertTrue(all(isinstance(d, str) for d in descs))


class TestBuildForumFeatures(unittest.TestCase):
    """Tests for forum engagement proxy features."""

    def test_forum_silence(self):
        df = pd.DataFrame({
            "total_clicks": [100, 50, 200],
            "engagement_forumng": [0, 10, 0],
            "engagement_oucontent": [80, 30, 150],
        })
        result = build_forum_features(df)
        expected_silence = [1, 0, 1]
        self.assertEqual(result["forum_silence"].tolist(), expected_silence)

    def test_forum_ratio(self):
        df = pd.DataFrame({
            "total_clicks": [100, 0],
            "engagement_forumng": [20, 0],
            "engagement_oucontent": [60, 0],
        })
        result = build_forum_features(df)
        self.assertAlmostEqual(
            result["forum_engagement_ratio"].iloc[0],
            20 / 101,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
