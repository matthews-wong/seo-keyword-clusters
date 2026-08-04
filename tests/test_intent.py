"""Tests for the rule-based intent classifier.

These assert the classifier's contract: known modifier phrases map to the
expected intent, priority resolves conflicts deterministically, and the fired
signals are reported.
"""

from __future__ import annotations

import pytest

from keywordclusters.intent import DEFAULT_INTENT, classify_intent


@pytest.mark.parametrize(
    "keyword,expected",
    [
        # transactional
        ("buy running shoes online", "transactional"),
        ("espresso machine price", "transactional"),
        ("coffee maker near me", "transactional"),
        ("running shoes discount code", "transactional"),
        ("cheapest espresso machine", "transactional"),
        # commercial
        ("best running shoes for beginners", "commercial"),
        ("trail running shoes review", "commercial"),
        ("running shoes vs walking shoes", "commercial"),
        ("coffee grinder comparison", "commercial"),
        # navigational
        ("adidas running shoes official website", "navigational"),
        ("delonghi espresso machine official", "navigational"),
        # informational
        ("how to clean running shoes", "informational"),
        ("what is a burr grinder", "informational"),
        ("minimalist running shoes guide", "informational"),
    ],
)
def test_known_examples_map_to_expected_intent(keyword: str, expected: str) -> None:
    assert classify_intent(keyword).intent == expected


def test_transactional_beats_commercial_by_priority() -> None:
    # "best" (commercial) + "buy"/"price" (transactional) -> transactional wins.
    result = classify_intent("buy the best espresso machine at a good price")
    assert result.intent == "transactional"
    assert "buy" in result.signals


def test_no_modifier_falls_back_to_default_intent() -> None:
    result = classify_intent("running shoes for flat feet")
    assert result.intent == DEFAULT_INTENT
    assert result.signals == ()


def test_word_boundary_prevents_false_positive() -> None:
    # "however" must not trigger the "how" informational modifier.
    result = classify_intent("however running shoes fit")
    assert "how" not in result.signals


def test_signals_are_reported_for_a_match() -> None:
    result = classify_intent("best budget trail running shoes review")
    assert result.intent == "commercial"
    assert "best" in result.signals
    assert "review" in result.signals
