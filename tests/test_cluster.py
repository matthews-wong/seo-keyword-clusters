"""Tests for TF-IDF + KMeans keyword clustering.

Clustering with a fixed ``random_state`` is deterministic, so we can assert
that obviously related keywords land in the same cluster and that repeated
runs produce identical assignments.
"""

from __future__ import annotations

from keywordclusters.cluster import cluster_keywords, suggest_cluster_count

# Two clearly separable themes; with n_clusters=2 they must split by theme.
RUNNING = [
    "best running shoes for beginners",
    "buy running shoes online",
    "running shoes for flat feet",
    "cheap running shoes for men",
]
COFFEE = [
    "best espresso machine for home",
    "buy espresso machine online",
    "how to make espresso at home",
    "cheap coffee maker for office",
]
KEYWORDS = RUNNING + COFFEE


def test_related_keywords_share_a_cluster() -> None:
    result = cluster_keywords(KEYWORDS, n_clusters=2, random_state=42)

    running_labels = {
        lbl for kw, lbl in zip(result.keywords, result.labels) if kw in RUNNING
    }
    coffee_labels = {
        lbl for kw, lbl in zip(result.keywords, result.labels) if kw in COFFEE
    }

    # Each theme collapses to a single cluster...
    assert len(running_labels) == 1
    assert len(coffee_labels) == 1
    # ...and the two themes are different clusters.
    assert running_labels != coffee_labels


def test_clustering_is_deterministic_across_runs() -> None:
    first = cluster_keywords(KEYWORDS, n_clusters=2, random_state=42)
    second = cluster_keywords(KEYWORDS, n_clusters=2, random_state=42)
    assert first.labels == second.labels


def test_cluster_labels_reflect_theme_terms() -> None:
    result = cluster_keywords(KEYWORDS, n_clusters=2, random_state=42)

    # The label for the running cluster should mention a running-shoe term.
    running_cluster = next(
        lbl for kw, lbl in zip(result.keywords, result.labels) if kw in RUNNING
    )
    label = result.cluster_labels[running_cluster].lower()
    assert any(term in label for term in ("running", "shoes"))


def test_output_lengths_align_with_input() -> None:
    result = cluster_keywords(KEYWORDS, n_clusters=3, random_state=42)
    assert len(result.labels) == len(KEYWORDS)
    assert len(result.keywords) == len(KEYWORDS)
    assert set(result.cluster_labels) == set(result.labels)


def test_suggest_cluster_count_is_bounded() -> None:
    assert suggest_cluster_count(1) == 1
    assert suggest_cluster_count(100) <= 10
    assert suggest_cluster_count(16) >= 2


def test_agglomerative_algorithm_also_separates_themes() -> None:
    result = cluster_keywords(
        KEYWORDS, n_clusters=2, algorithm="agglomerative", random_state=42
    )
    running_labels = {
        lbl for kw, lbl in zip(result.keywords, result.labels) if kw in RUNNING
    }
    assert len(running_labels) == 1
