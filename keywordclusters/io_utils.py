"""Input/output helpers: load keywords and export results.

Kept separate from clustering/intent logic so the core stays free of file and
formatting concerns. Exports are built from a plain list of row dicts so the
same data drives CSV, Markdown, and the console view.
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

import pandas as pd

# Column order shared across every export format.
COLUMNS: tuple[str, ...] = (
    "keyword",
    "cluster",
    "cluster_label",
    "intent",
    "intent_signals",
)


def load_keywords(path: str | Path) -> list[str]:
    """Load keywords from a text file (one keyword per line).

    Blank lines and lines beginning with ``#`` (comments) are skipped, and
    surrounding whitespace is stripped. Duplicate keywords are removed while
    preserving first-seen order.

    Args:
        path: Path to the keyword list file.

    Returns:
        Ordered list of unique, cleaned keywords.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file contains no usable keywords.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Keyword file not found: {file_path}")

    seen: "OrderedDict[str, None]" = OrderedDict()
    for raw in file_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        seen.setdefault(line, None)

    if not seen:
        raise ValueError(f"No keywords found in {file_path}")

    return list(seen.keys())


def build_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with the canonical column order from row dicts."""
    df = pd.DataFrame(rows)
    return df.reindex(columns=list(COLUMNS))


def export_csv(rows: list[dict], path: str | Path) -> Path:
    """Write rows to a CSV file with the canonical column order.

    Args:
        rows: Row dicts keyed by :data:`COLUMNS`.
        path: Destination CSV path.

    Returns:
        The path written to.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in COLUMNS})
    return out_path


def export_markdown(rows: list[dict], cluster_labels: dict[int, str], path: str | Path) -> Path:
    """Write a Markdown report grouped by cluster.

    The report lists each cluster with its auto-generated label followed by a
    table of its keywords and their classified intent.

    Args:
        rows: Row dicts keyed by :data:`COLUMNS`.
        cluster_labels: Mapping of cluster id to human-readable label.
        path: Destination Markdown path.

    Returns:
        The path written to.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["# Keyword clusters & intent", ""]
    lines.append(f"Total keywords: **{len(rows)}** across **{len(cluster_labels)}** clusters.")
    lines.append("")

    for cid in sorted(cluster_labels):
        members = [r for r in rows if r["cluster"] == cid]
        lines.append(f"## Cluster {cid}: {cluster_labels[cid]}")
        lines.append("")
        lines.append("| Keyword | Intent | Signals |")
        lines.append("| --- | --- | --- |")
        for row in members:
            signals = row.get("intent_signals", "") or "-"
            lines.append(f"| {row['keyword']} | {row['intent']} | {signals} |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
