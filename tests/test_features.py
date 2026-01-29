"""Tests for feature extraction."""

import pytest
import numpy as np

from polarity.features.tfidf_features import (
    build_tfidf_vectorizer,
    extract_discriminative_phrases,
)


class TestTfidfVectorizer:
    @pytest.fixture
    def sample_texts(self):
        return [
            "The government should increase healthcare spending",
            "Tax cuts stimulate economic growth and create jobs",
            "Both sides have valid points on this policy issue",
            "Universal healthcare is a fundamental right",
            "Free market solutions drive innovation",
            "The debate continues with no clear consensus",
        ]

    def test_vectorizer_returns_correct_shape(self, sample_texts):
        vectorizer, X = build_tfidf_vectorizer(
            sample_texts, max_features=100, ngram_range=(1, 2), min_df=1,
        )
        assert X.shape[0] == len(sample_texts)
        assert X.shape[1] <= 100
        assert X.shape[1] > 0

    def test_vectorizer_sparse_output(self, sample_texts):
        from scipy.sparse import issparse
        vectorizer, X = build_tfidf_vectorizer(
            sample_texts, max_features=100, min_df=1,
        )
        assert issparse(X)

    def test_feature_names_match_shape(self, sample_texts):
        vectorizer, X = build_tfidf_vectorizer(
            sample_texts, max_features=100, min_df=1,
        )
        feature_names = vectorizer.get_feature_names_out()
        assert len(feature_names) == X.shape[1]

    def test_transform_new_data(self, sample_texts):
        vectorizer, X_train = build_tfidf_vectorizer(
            sample_texts[:4], max_features=100, min_df=1,
        )
        X_new = vectorizer.transform(sample_texts[4:])
        assert X_new.shape[0] == 2
        assert X_new.shape[1] == X_train.shape[1]


class TestDiscriminativePhrases:
    def test_returns_phrases_per_class(self):
        texts = [
            "healthcare spending government programs",
            "tax cuts free market economy",
            "balanced view both perspectives",
        ] * 5

        from polarity.features.tfidf_features import build_tfidf_vectorizer
        vectorizer, X = build_tfidf_vectorizer(texts, max_features=100, min_df=1)
        y = np.array([0, 1, 2] * 5)
        feature_names = list(vectorizer.get_feature_names_out())

        phrases = extract_discriminative_phrases(
            X, y, feature_names, method="chi2", top_k=5,
            class_names=["left", "right", "center"],
        )
        assert "left" in phrases
        assert "right" in phrases
        assert "center" in phrases

    def test_phrases_are_tuples(self):
        texts = ["word1 word2", "word3 word4", "word5 word6"] * 5
        from polarity.features.tfidf_features import build_tfidf_vectorizer
        vectorizer, X = build_tfidf_vectorizer(texts, max_features=50, min_df=1)
        y = np.array([0, 1, 2] * 5)
        feature_names = list(vectorizer.get_feature_names_out())

        phrases = extract_discriminative_phrases(
            X, y, feature_names, method="chi2", top_k=3,
        )
        for cls_name, phrase_list in phrases.items():
            for item in phrase_list:
                assert isinstance(item, tuple)
                assert len(item) == 2
                assert isinstance(item[0], str)
                assert isinstance(item[1], float)
