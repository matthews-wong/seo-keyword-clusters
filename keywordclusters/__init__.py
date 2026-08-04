"""seo-keyword-clusters.

An offline Python CLI that groups a seed list of SEO keywords into topic
clusters (TF-IDF + KMeans/agglomerative) and classifies the search intent of
each keyword with a transparent, rule-based classifier.

The package is intentionally small and dependency-light so results are fully
reproducible offline.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
