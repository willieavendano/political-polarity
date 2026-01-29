"""Tests for preprocessing module."""

import pytest
import pandas as pd
import numpy as np

from polarity.data.preprocess import (
    compute_text_hash,
    deduplicate,
    normalize_text,
    filter_length,
)


class TestTextHash:
    def test_same_text_same_hash(self):
        h1 = compute_text_hash("Hello world")
        h2 = compute_text_hash("Hello world")
        assert h1 == h2

    def test_different_text_different_hash(self):
        h1 = compute_text_hash("Hello world")
        h2 = compute_text_hash("Goodbye world")
        assert h1 != h2

    def test_whitespace_normalized(self):
        h1 = compute_text_hash("Hello   world")
        h2 = compute_text_hash("Hello world")
        assert h1 == h2

    def test_case_insensitive(self):
        h1 = compute_text_hash("Hello World")
        h2 = compute_text_hash("hello world")
        assert h1 == h2


class TestDeduplicate:
    def test_removes_duplicates(self):
        df = pd.DataFrame({
            "text": ["Hello world", "Hello world", "Different text"],
            "doc_id": ["a", "b", "c"],
        })
        result = deduplicate(df)
        assert len(result) == 2

    def test_keeps_first(self):
        df = pd.DataFrame({
            "text": ["Hello world", "Hello world"],
            "doc_id": ["first", "second"],
        })
        result = deduplicate(df)
        assert result.iloc[0]["doc_id"] == "first"

    def test_no_duplicates(self):
        df = pd.DataFrame({
            "text": ["Text A", "Text B", "Text C"],
            "doc_id": ["a", "b", "c"],
        })
        result = deduplicate(df)
        assert len(result) == 3


class TestNormalizeText:
    def test_collapses_whitespace(self):
        result = normalize_text("Hello    world\n\n\ntest")
        assert "  " not in result

    def test_removes_urls(self):
        result = normalize_text("Check out https://example.com for more")
        assert "https" not in result
        assert "example.com" not in result

    def test_normalizes_quotes(self):
        result = normalize_text("\u201cHello\u201d")
        assert '"' in result

    def test_removes_boilerplate(self):
        result = normalize_text("Good article. Subscribe to our newsletter here")
        assert "subscribe" not in result.lower() or "newsletter" not in result.lower()

    def test_handles_empty_string(self):
        result = normalize_text("")
        assert result == ""


class TestFilterLength:
    def test_filters_short(self):
        df = pd.DataFrame({
            "text": ["short", "This is a much longer text that meets the minimum"],
        })
        result = filter_length(df, min_length=20, max_length=10000)
        assert len(result) == 1

    def test_filters_long(self):
        df = pd.DataFrame({
            "text": ["x" * 100, "x" * 10000],
        })
        result = filter_length(df, min_length=1, max_length=500)
        assert len(result) == 1

    def test_keeps_within_range(self):
        df = pd.DataFrame({
            "text": ["x" * 50, "x" * 100, "x" * 200],
        })
        result = filter_length(df, min_length=40, max_length=250)
        assert len(result) == 3
