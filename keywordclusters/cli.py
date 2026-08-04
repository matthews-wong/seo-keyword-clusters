"""Command-line interface for seo-keyword-clusters.

Usage::

    seo-keyword-clusters --help
    seo-keyword-clusters run                       # uses the bundled sample list
    seo-keyword-clusters run -k my_keywords.txt -n 6
    seo-keyword-clusters run --csv out.csv --markdown out.md

The command loads keywords, clusters them, classifies intent per keyword, and
renders a rich console report. Optional flags export CSV / Markdown.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .cluster import (
    DEFAULT_ALGORITHM,
    DEFAULT_RANDOM_STATE,
    ClusterResult,
    cluster_keywords,
)
from .intent import classify_intent
from .io_utils import build_dataframe, export_csv, export_markdown, load_keywords

# The bundled sample list ships inside the repo's data/ directory.
DEFAULT_KEYWORDS = Path(__file__).resolve().parent.parent / "data" / "keywords_sample.txt"

INTENT_STYLES = {
    "transactional": "bold green",
    "commercial": "bold yellow",
    "navigational": "bold cyan",
    "informational": "bold blue",
}


def _build_rows(result: ClusterResult) -> list[dict]:
    """Combine clustering + intent into a list of export-ready row dicts."""
    rows: list[dict] = []
    for keyword, cluster_id in zip(result.keywords, result.labels):
        intent = classify_intent(keyword)
        rows.append(
            {
                "keyword": keyword,
                "cluster": cluster_id,
                "cluster_label": result.cluster_labels[cluster_id],
                "intent": intent.intent,
                "intent_signals": ", ".join(intent.signals),
            }
        )
    return rows


def _render(console: Console, result: ClusterResult, rows: list[dict]) -> None:
    """Render clusters and an intent summary to the console."""
    console.print(
        Panel.fit(
            f"[bold]seo-keyword-clusters[/bold] v{__version__}\n"
            f"{len(result.keywords)} keywords -> {result.n_clusters} clusters",
            border_style="magenta",
        )
    )

    for cid in sorted(result.cluster_labels):
        members = [r for r in rows if r["cluster"] == cid]
        table = Table(
            title=f"Cluster {cid}: {result.cluster_labels[cid]}",
            title_style="bold magenta",
            header_style="bold",
            show_lines=False,
        )
        table.add_column("Keyword")
        table.add_column("Intent")
        table.add_column("Signals", style="dim")
        for row in members:
            style = INTENT_STYLES.get(row["intent"], "")
            table.add_row(
                row["keyword"],
                f"[{style}]{row['intent']}[/{style}]" if style else row["intent"],
                row["intent_signals"] or "-",
            )
        console.print(table)

    _render_intent_summary(console, rows)


def _render_intent_summary(console: Console, rows: list[dict]) -> None:
    """Render an aggregate count of each intent across all keywords."""
    counts = Counter(row["intent"] for row in rows)
    summary = Table(title="Intent distribution", title_style="bold", header_style="bold")
    summary.add_column("Intent")
    summary.add_column("Keywords", justify="right")
    for intent in ("transactional", "commercial", "navigational", "informational"):
        style = INTENT_STYLES.get(intent, "")
        summary.add_row(f"[{style}]{intent}[/{style}]", str(counts.get(intent, 0)))
    console.print(summary)


@click.group()
@click.version_option(__version__, prog_name="seo-keyword-clusters")
def main() -> None:
    """Group SEO keywords into topic clusters and classify their intent."""


@main.command()
@click.option(
    "-k",
    "--keywords",
    "keywords_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a keyword file (one per line). Defaults to the bundled sample.",
)
@click.option(
    "-n",
    "--num-clusters",
    type=int,
    default=None,
    help="Number of clusters. Defaults to a sqrt-based heuristic.",
)
@click.option(
    "-a",
    "--algorithm",
    type=click.Choice(["kmeans", "agglomerative"]),
    default=DEFAULT_ALGORITHM,
    show_default=True,
    help="Clustering algorithm.",
)
@click.option(
    "--random-state",
    type=int,
    default=DEFAULT_RANDOM_STATE,
    show_default=True,
    help="Random seed for reproducible KMeans results.",
)
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Also export results to this CSV file.",
)
@click.option(
    "--markdown",
    "markdown_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Also export a grouped Markdown report to this file.",
)
def run(
    keywords_path: Path | None,
    num_clusters: int | None,
    algorithm: str,
    random_state: int,
    csv_path: Path | None,
    markdown_path: Path | None,
) -> None:
    """Cluster keywords and classify intent, then print/export the report."""
    console = Console()
    path = keywords_path or DEFAULT_KEYWORDS
    keywords = load_keywords(path)

    result = cluster_keywords(
        keywords,
        n_clusters=num_clusters,
        algorithm=algorithm,
        random_state=random_state,
    )
    rows = _build_rows(result)

    _render(console, result, rows)

    if csv_path is not None:
        export_csv(rows, csv_path)
        console.print(f"[green]Wrote CSV:[/green] {csv_path}")
    if markdown_path is not None:
        export_markdown(rows, result.cluster_labels, markdown_path)
        console.print(f"[green]Wrote Markdown:[/green] {markdown_path}")

    # Touch the DataFrame builder so pandas is exercised even without export;
    # this also validates that every row has the canonical columns.
    build_dataframe(rows)


if __name__ == "__main__":  # pragma: no cover
    main()
