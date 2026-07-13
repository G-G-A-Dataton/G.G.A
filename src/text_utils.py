"""Shared text-matching helpers with Unicode-aware word boundaries."""

import re
from functools import lru_cache


_NON_WORD_PATTERN = re.compile(r"[^\w]+", flags=re.UNICODE)


def normalize_for_matching(text) -> str:
    """Case-fold text and normalize punctuation/whitespace to word separators."""
    if not isinstance(text, str):
        return ""
    return " ".join(_NON_WORD_PATTERN.sub(" ", text.casefold()).split())


@lru_cache(maxsize=8_192)
def _normalize_phrase(phrase: str) -> str:
    return normalize_for_matching(phrase)


def contains_phrase(text, phrase) -> bool:
    """Return whether `phrase` occurs as complete words in `text`."""
    if not isinstance(phrase, str):
        return False
    normalized_text = normalize_for_matching(text)
    normalized_phrase = _normalize_phrase(phrase)
    if not normalized_text or not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_text} "


def find_phrase_values(text, phrase_values) -> set:
    """Return mapped values for every complete phrase found in text."""
    normalized_text = normalize_for_matching(text)
    if not normalized_text:
        return set()
    bordered_text = f" {normalized_text} "
    return {
        value
        for phrase, value in phrase_values.items()
        if f" {_normalize_phrase(phrase)} " in bordered_text
    }


def first_phrase_value(text, phrase_values, default=""):
    """Return the first mapped value whose complete phrase occurs in text."""
    normalized_text = normalize_for_matching(text)
    if not normalized_text:
        return default
    bordered_text = f" {normalized_text} "
    for phrase, value in phrase_values.items():
        if f" {_normalize_phrase(phrase)} " in bordered_text:
            return value
    return default
