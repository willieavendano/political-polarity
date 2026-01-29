"""Tests for k-anonymity suppression in subset reporting."""

import pytest
import numpy as np
import pandas as pd

from polarity.reporting.subset_analysis import compute_subset_report


class TestKAnonymity:
    @pytest.fixture
    def sample_data(self):
        """Create sample data with known group sizes."""
        n = 100
        np.random.seed(42)

        df = pd.DataFrame({
            "text": [f"Document {i}" for i in range(n)],
            "source": (["outlet_a"] * 50 + ["outlet_b"] * 30 +
                       ["outlet_c"] * 15 + ["outlet_d"] * 5),
        })
        polarity_scores = np.random.uniform(-1, 1, n)
        pred_labels = np.random.choice([0, 1, 2], n)
        pred_proba = np.random.dirichlet([1, 1, 1], n)

        return df, polarity_scores, pred_labels, pred_proba

    def test_suppresses_small_groups(self, sample_data):
        df, scores, labels, proba = sample_data

        report = compute_subset_report(
            df=df,
            polarity_scores=scores,
            pred_labels=labels,
            pred_proba=proba,
            class_names=["left", "center", "right"],
            group_by_field="source",
            k_anonymity=30,
        )

        # outlet_a (50) and outlet_b (30) should be reported
        assert "outlet_a" in report["groups"]
        assert "outlet_b" in report["groups"]

        # outlet_c (15) and outlet_d (5) should be suppressed
        suppressed_names = [s["group"] for s in report["suppressed_groups"]]
        assert "outlet_c" in suppressed_names
        assert "outlet_d" in suppressed_names

    def test_all_groups_suppressed_with_high_k(self, sample_data):
        df, scores, labels, proba = sample_data

        report = compute_subset_report(
            df=df,
            polarity_scores=scores,
            pred_labels=labels,
            pred_proba=proba,
            class_names=["left", "center", "right"],
            group_by_field="source",
            k_anonymity=100,  # Only outlet_a has 50, none reach 100
        )

        assert len(report["groups"]) == 0
        assert len(report["suppressed_groups"]) == 4

    def test_no_suppression_with_low_k(self, sample_data):
        df, scores, labels, proba = sample_data

        report = compute_subset_report(
            df=df,
            polarity_scores=scores,
            pred_labels=labels,
            pred_proba=proba,
            class_names=["left", "center", "right"],
            group_by_field="source",
            k_anonymity=1,
        )

        assert len(report["groups"]) == 4
        assert len(report["suppressed_groups"]) == 0

    def test_group_size_in_report(self, sample_data):
        df, scores, labels, proba = sample_data

        report = compute_subset_report(
            df=df,
            polarity_scores=scores,
            pred_labels=labels,
            pred_proba=proba,
            class_names=["left", "center", "right"],
            group_by_field="source",
            k_anonymity=1,
        )

        assert report["groups"]["outlet_a"]["size"] == 50
        assert report["groups"]["outlet_b"]["size"] == 30
        assert report["groups"]["outlet_c"]["size"] == 15
        assert report["groups"]["outlet_d"]["size"] == 5

    def test_polarity_range(self, sample_data):
        df, scores, labels, proba = sample_data

        report = compute_subset_report(
            df=df,
            polarity_scores=scores,
            pred_labels=labels,
            pred_proba=proba,
            class_names=["left", "center", "right"],
            group_by_field="source",
            k_anonymity=1,
        )

        for group_name, data in report["groups"].items():
            pol = data["polarity"]
            assert -1 <= pol["min"] <= 1
            assert -1 <= pol["max"] <= 1
            assert pol["min"] <= pol["mean"] <= pol["max"]

    def test_missing_field_returns_empty(self):
        df = pd.DataFrame({"text": ["doc"], "source": ["a"]})
        report = compute_subset_report(
            df=df,
            polarity_scores=np.array([0.0]),
            pred_labels=np.array([1]),
            pred_proba=np.array([[0.2, 0.6, 0.2]]),
            class_names=["left", "center", "right"],
            group_by_field="nonexistent_field",
            k_anonymity=1,
        )
        # Should handle gracefully (group everything as "unknown")
        assert isinstance(report, dict)

    def test_subset_meta_extraction(self):
        """Test that subset_meta fields are properly extracted."""
        df = pd.DataFrame({
            "text": [f"doc {i}" for i in range(40)],
            "subset_meta": [{"region": "Northeast"}] * 35 + [{"region": "South"}] * 5,
        })

        report = compute_subset_report(
            df=df,
            polarity_scores=np.random.uniform(-1, 1, 40),
            pred_labels=np.random.choice([0, 1, 2], 40),
            pred_proba=np.random.dirichlet([1, 1, 1], 40),
            class_names=["left", "center", "right"],
            group_by_field="region",
            k_anonymity=30,
        )

        assert "Northeast" in report["groups"]
        suppressed_names = [s["group"] for s in report["suppressed_groups"]]
        assert "South" in suppressed_names
