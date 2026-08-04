"""Transparent, rule-based search-intent classification.

Every keyword is assigned one of four canonical search intents:

* ``transactional``  - the searcher wants to act now (buy, order, subscribe).
* ``commercial``     - the searcher is comparing before acting (best, review, vs).
* ``navigational``   - the searcher wants a specific site/brand (login, official).
* ``informational``  - the searcher wants to learn (how, what, guide).

The classifier is deliberately rule-based rather than learned: the mapping from
modifier lexicons to intent is fully inspectable, deterministic, and needs no
training data. Each classification also returns the exact modifiers that fired,
so the decision can be explained and audited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------
# Ordered from strongest/most-specific intent to weakest. When modifiers from
# several intents appear in one keyword, the earliest intent in PRIORITY wins.
# This mirrors real search behaviour: an action signal ("buy") dominates a
# comparison signal ("best"), which in turn dominates a plain topic word.

TRANSACTIONAL_MODIFIERS: tuple[str, ...] = (
    "buy",
    "order",
    "purchase",
    "price",
    "prices",
    "pricing",
    "cost",
    "cheap",
    "cheapest",
    "discount",
    "coupon",
    "coupons",
    "deal",
    "deals",
    "sale",
    "for sale",
    "shop",
    "subscription",
    "subscribe",
    "near me",
    "free shipping",
)

COMMERCIAL_MODIFIERS: tuple[str, ...] = (
    "best",
    "top",
    "review",
    "reviews",
    "comparison",
    "compare",
    "vs",
    "versus",
    "alternative",
    "alternatives",
    "affordable",
    "budget",
    "recommended",
)

NAVIGATIONAL_MODIFIERS: tuple[str, ...] = (
    "login",
    "log in",
    "sign in",
    "signin",
    "official",
    "website",
    "homepage",
    "portal",
    "dashboard",
    "app",
    "download",
    ".com",
)

INFORMATIONAL_MODIFIERS: tuple[str, ...] = (
    "how",
    "how to",
    "what",
    "what is",
    "why",
    "when",
    "where",
    "who",
    "guide",
    "tutorial",
    "tips",
    "ideas",
    "examples",
    "meaning",
    "definition",
    "explained",
    "learn",
)

# Map an intent label to its lexicon.
LEXICONS: dict[str, tuple[str, ...]] = {
    "transactional": TRANSACTIONAL_MODIFIERS,
    "commercial": COMMERCIAL_MODIFIERS,
    "navigational": NAVIGATIONAL_MODIFIERS,
    "informational": INFORMATIONAL_MODIFIERS,
}

# Resolution order when multiple intents match.
PRIORITY: tuple[str, ...] = (
    "transactional",
    "commercial",
    "navigational",
    "informational",
)

# Default when nothing matches: most bare "topic" queries are informational.
DEFAULT_INTENT = "informational"


@dataclass(frozen=True)
class IntentResult:
    """Outcome of classifying a single keyword.

    Attributes:
        intent: One of the four canonical intent labels.
        signals: The modifiers (in lexicon form) that triggered the decision.
                 Empty when the default intent was used.
    """

    intent: str
    signals: tuple[str, ...]


def _matches(text: str, modifier: str) -> bool:
    """Return True if ``modifier`` appears in ``text`` on word boundaries.

    Multi-word modifiers ("near me") and modifiers containing punctuation
    (".com") are matched as substrings/phrases; single tokens are matched on
    word boundaries so "how" does not fire inside "however".
    """
    if " " in modifier or not modifier.isalnum():
        return modifier in text
    return re.search(rf"\b{re.escape(modifier)}\b", text) is not None


def classify_intent(keyword: str) -> IntentResult:
    """Classify a single keyword's search intent.

    Args:
        keyword: The raw keyword phrase (case-insensitive).

    Returns:
        An :class:`IntentResult` with the resolved intent and the modifiers
        that fired. If no modifier matches, the intent falls back to
        :data:`DEFAULT_INTENT` with no signals.
    """
    text = keyword.lower().strip()

    matched: dict[str, list[str]] = {}
    for intent, modifiers in LEXICONS.items():
        hits = [m for m in modifiers if _matches(text, m)]
        if hits:
            matched[intent] = hits

    for intent in PRIORITY:
        if intent in matched:
            return IntentResult(intent=intent, signals=tuple(matched[intent]))

    return IntentResult(intent=DEFAULT_INTENT, signals=())


def classify_many(keywords: list[str]) -> list[IntentResult]:
    """Classify a list of keywords, preserving input order."""
    return [classify_intent(k) for k in keywords]
