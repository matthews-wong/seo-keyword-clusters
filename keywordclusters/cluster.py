"""Keyword clustering via TF-IDF vectorization + KMeans/agglomerative.

The pipeline is:

1. Vectorize keywords with a TF-IDF model that blends word unigrams/bigrams
   with character n-grams. Character n-grams help short SEO phrases group by
   shared roots ("run"/"running"/"runner") even without exact token overlap.
2. Cluster the sparse vectors with KMeans (default) or agglomerative
   clustering. A fixed ``random_state`` makes KMeans fully reproducible.
3. Auto-label each cluster by its most distinctive TF-IDF terms.

Everything is deterministic for a given input + ``random_state``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_RANDOM_STATE = 42
DEFAULT_ALGORITHM = "kmeans"
DEFAULT_LABEL_TERMS = 3

# Stop words that add no *topical* signal for SEO keyword phrases: plain
# English function words plus generic commercial/intent modifiers. The intent
# modifiers ("best", "buy", "cheap", ...) describe search intent, not subject
# matter, and are classified separately in ``intent.py`` — leaving them in the
# vectors would blur otherwise-distinct topics that happen to share a modifier
# ("cheap running shoes" vs "cheap coffee maker"). Kept explicit rather than
# pulling a large external list.
STOP_WORDS: tuple[str, ...] = (
    # function words
    "a",
    "an",
    "and",
    "at",
    "for",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "with",
    # generic commercial / intent modifiers (topic-neutral)
    "best",
    "buy",
    "cheap",
    "top",
    "online",
    "review",
    "reviews",
    "near",
    "me",
    "how",
    "what",
    "vs",
)


@dataclass
class ClusterResult:
    """Result of clustering a list of keywords.

    Attributes:
        keywords: The input keywords, in original order.
        labels: Cluster id (int) assigned to each keyword, aligned to
            ``keywords`` by index.
        cluster_labels: Human-readable label per cluster id, derived from the
            cluster's top TF-IDF terms.
        n_clusters: Number of clusters produced.
    """

    keywords: list[str]
    labels: list[int]
    cluster_labels: dict[int, str]
    n_clusters: int
    _terms: dict[int, list[str]] = field(default_factory=dict, repr=False)

    def top_terms(self, cluster_id: int) -> list[str]:
        """Return the ranked distinctive terms backing a cluster's label."""
        return self._terms.get(cluster_id, [])

    def keywords_in(self, cluster_id: int) -> list[str]:
        """Return the keywords assigned to ``cluster_id`` in input order."""
        return [k for k, lbl in zip(self.keywords, self.labels) if lbl == cluster_id]


def build_vectorizer() -> TfidfVectorizer:
    """Build the TF-IDF vectorizer used for clustering.

    Combines word unigrams+bigrams with character n-grams (via ``char_wb``)
    so morphologically related keywords share features. ``sublinear_tf``
    dampens the effect of repeated tokens in longer phrases.
    """
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        stop_words=list(STOP_WORDS),
        sublinear_tf=True,
        min_df=1,
    )


def _make_model(algorithm: str, n_clusters: int, random_state: int):
    """Instantiate the requested clustering estimator."""
    if algorithm == "kmeans":
        # n_init=10 + fixed random_state => reproducible centroids.
        return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    if algorithm == "agglomerative":
        return AgglomerativeClustering(n_clusters=n_clusters)
    raise ValueError(
        f"Unknown algorithm {algorithm!r}; expected 'kmeans' or 'agglomerative'."
    )


def suggest_cluster_count(n_keywords: int) -> int:
    """Heuristic default cluster count when the user does not specify one.

    Roughly the square root of the keyword count, clamped to a sensible range.
    """
    if n_keywords <= 2:
        return max(1, n_keywords)
    return max(2, min(10, round(n_keywords**0.5)))


def _label_clusters(
    keywords: list[str],
    labels: np.ndarray,
    vectorizer: TfidfVectorizer,
    matrix,
    n_clusters: int,
    n_terms: int,
) -> tuple[dict[int, str], dict[int, list[str]]]:
    """Derive a readable label per cluster from its mean TF-IDF profile.

    For each cluster we average the TF-IDF rows of its members and take the
    highest-weighted feature names. This surfaces the terms that most
    characterize the cluster, which we join into a label.
    """
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    dense = matrix.toarray()

    cluster_labels: dict[int, str] = {}
    cluster_terms: dict[int, list[str]] = {}

    for cid in range(n_clusters):
        member_mask = labels == cid
        if not member_mask.any():
            cluster_labels[cid] = f"cluster {cid}"
            cluster_terms[cid] = []
            continue

        centroid = dense[member_mask].mean(axis=0)
        # Prefer single-word features for readable labels, but fall back to
        # whatever ranks highest if unigrams are exhausted.
        ranked = np.argsort(centroid)[::-1]
        terms: list[str] = []
        for idx in ranked:
            if centroid[idx] <= 0:
                break
            term = feature_names[idx]
            if term not in terms:
                terms.append(term)
            if len(terms) >= n_terms:
                break

        cluster_terms[cid] = terms
        cluster_labels[cid] = " / ".join(terms) if terms else f"cluster {cid}"

    return cluster_labels, cluster_terms


def cluster_keywords(
    keywords: list[str],
    n_clusters: int | None = None,
    algorithm: str = DEFAULT_ALGORITHM,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_label_terms: int = DEFAULT_LABEL_TERMS,
) -> ClusterResult:
    """Cluster keywords into topic groups and auto-label each group.

    Args:
        keywords: Non-empty list of keyword phrases.
        n_clusters: Desired number of clusters. If ``None``, a heuristic count
            is chosen from the number of keywords.
        algorithm: ``"kmeans"`` (default) or ``"agglomerative"``.
        random_state: Seed for reproducible KMeans results.
        n_label_terms: How many top terms to include in each cluster label.

    Returns:
        A :class:`ClusterResult`.

    Raises:
        ValueError: If ``keywords`` is empty.
    """
    if not keywords:
        raise ValueError("keywords must be a non-empty list.")

    if n_clusters is None:
        n_clusters = suggest_cluster_count(len(keywords))
    # Never request more clusters than we have keywords.
    n_clusters = max(1, min(n_clusters, len(keywords)))

    vectorizer = build_vectorizer()
    matrix = vectorizer.fit_transform(keywords)

    model = _make_model(algorithm, n_clusters, random_state)
    labels = model.fit_predict(matrix.toarray())

    cluster_labels, cluster_terms = _label_clusters(
        keywords, labels, vectorizer, matrix, n_clusters, n_label_terms
    )

    return ClusterResult(
        keywords=list(keywords),
        labels=[int(x) for x in labels],
        cluster_labels=cluster_labels,
        n_clusters=n_clusters,
        _terms=cluster_terms,
    )
