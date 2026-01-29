"""Embedding-aware keyword extraction (KeyBERT-style, lightweight)."""

import logging
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class LightKeywordExtractor:
    """Lightweight keyword extractor using TF-IDF embeddings.

    This is a simplified KeyBERT-style approach that works without
    sentence-transformers. It uses TF-IDF vectors as document and
    candidate representations, finding candidates whose TF-IDF profile
    most closely matches the document.

    If sentence-transformers is available, it uses proper embeddings instead.
    """

    def __init__(self, use_embeddings: bool = True, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize extractor.

        Args:
            use_embeddings: Try to use sentence-transformers embeddings.
            model_name: Sentence transformer model name.
        """
        self._embedder = None
        self._use_embeddings = use_embeddings

        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(model_name)
                logger.info("Using sentence-transformers model: %s", model_name)
            except ImportError:
                logger.info(
                    "sentence-transformers not available; "
                    "falling back to TF-IDF cosine keyword extraction"
                )

    def extract_keywords(
        self,
        text: str,
        top_k: int = 20,
        ngram_range: tuple[int, int] = (1, 3),
        diversity: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Extract keywords from a single document.

        Args:
            text: Document text.
            top_k: Number of keywords to return.
            ngram_range: N-gram range for candidates.
            diversity: Diversity parameter (0=most relevant, 1=most diverse).

        Returns:
            List of (keyword, score) tuples.
        """
        if self._embedder is not None:
            return self._extract_with_embeddings(text, top_k, ngram_range, diversity)
        else:
            return self._extract_with_tfidf(text, top_k, ngram_range)

    def _extract_with_embeddings(
        self,
        text: str,
        top_k: int,
        ngram_range: tuple[int, int],
        diversity: float,
    ) -> list[tuple[str, float]]:
        """Extract using sentence embeddings (preferred method)."""
        # Get candidate n-grams
        candidates = self._get_candidates(text, ngram_range)
        if not candidates:
            return []

        # Encode document and candidates
        doc_embedding = self._embedder.encode([text])
        candidate_embeddings = self._embedder.encode(candidates)

        # Cosine similarity
        similarities = cosine_similarity(doc_embedding, candidate_embeddings)[0]

        if diversity > 0:
            # MMR-style diversification
            return self._mmr_diversify(
                candidates, similarities, candidate_embeddings,
                top_k=top_k, diversity=diversity,
            )
        else:
            top_idx = np.argsort(similarities)[::-1][:top_k]
            return [(candidates[i], float(similarities[i])) for i in top_idx]

    def _extract_with_tfidf(
        self,
        text: str,
        top_k: int,
        ngram_range: tuple[int, int],
    ) -> list[tuple[str, float]]:
        """Extract using TF-IDF cosine similarity (fallback)."""
        candidates = self._get_candidates(text, ngram_range)
        if not candidates:
            return []

        # Build TF-IDF for document and candidates together
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = [text] + candidates
        vectorizer = TfidfVectorizer(ngram_range=(1, 1))

        try:
            X = vectorizer.fit_transform(corpus)
        except ValueError:
            return []

        doc_vec = X[0:1]
        cand_vecs = X[1:]

        similarities = cosine_similarity(doc_vec, cand_vecs)[0]
        top_idx = np.argsort(similarities)[::-1][:top_k]

        return [(candidates[i], float(similarities[i])) for i in top_idx]

    @staticmethod
    def _get_candidates(text: str, ngram_range: tuple[int, int]) -> list[str]:
        """Extract candidate n-grams from text."""
        cv = CountVectorizer(ngram_range=ngram_range, stop_words="english")
        try:
            cv.fit([text])
        except ValueError:
            return []
        return list(cv.get_feature_names_out())

    @staticmethod
    def _mmr_diversify(
        candidates: list[str],
        doc_similarities: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int,
        diversity: float,
    ) -> list[tuple[str, float]]:
        """Maximal Marginal Relevance diversification.

        Balances relevance to the document with diversity among
        selected keywords.
        """
        # Candidate-to-candidate similarity
        cand_sim = cosine_similarity(candidate_embeddings)

        selected_idx: list[int] = []
        remaining = list(range(len(candidates)))

        # Start with most relevant
        best = int(np.argmax(doc_similarities))
        selected_idx.append(best)
        remaining.remove(best)

        for _ in range(min(top_k - 1, len(remaining))):
            if not remaining:
                break

            mmr_scores = []
            for idx in remaining:
                relevance = doc_similarities[idx]
                # Max similarity to already-selected
                if selected_idx:
                    redundancy = max(cand_sim[idx][s] for s in selected_idx)
                else:
                    redundancy = 0
                mmr = (1 - diversity) * relevance - diversity * redundancy
                mmr_scores.append((idx, mmr))

            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            best_idx = mmr_scores[0][0]
            selected_idx.append(best_idx)
            remaining.remove(best_idx)

        return [
            (candidates[i], float(doc_similarities[i]))
            for i in selected_idx
        ]


def extract_corpus_keywords(
    texts: list[str],
    labels: np.ndarray,
    class_names: list[str],
    top_k: int = 20,
    use_embeddings: bool = True,
) -> dict[str, list[tuple[str, float]]]:
    """Extract representative keywords per class from a corpus.

    For each class, concatenates texts and extracts keywords from
    the combined text.

    Args:
        texts: Document texts.
        labels: Integer class labels.
        class_names: Class name mapping.
        top_k: Keywords per class.
        use_embeddings: Use embedding-based extraction.

    Returns:
        Dictionary mapping class_name -> list of (keyword, score).
    """
    extractor = LightKeywordExtractor(use_embeddings=use_embeddings)
    results: dict[str, list[tuple[str, float]]] = {}

    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        if not mask.any():
            results[cls_name] = []
            continue

        # Concatenate texts (limit to avoid memory issues)
        cls_texts = [t for t, m in zip(texts, mask) if m]
        combined = " ".join(cls_texts[:200])  # limit for memory
        # Truncate to ~50k chars for extraction
        combined = combined[:50000]

        keywords = extractor.extract_keywords(combined, top_k=top_k)
        results[cls_name] = keywords
        logger.info(
            "Keywords for '%s' (%d docs): %s ...",
            cls_name, mask.sum(), [kw[0] for kw in keywords[:5]],
        )

    return results
