"""Character/word error-rate scoring with explicit normalization rules.

The normalization rules implemented here are part of the benchmark methodology
(see docs/METHODOLOGY.md) and are unit-tested against golden cases:

- Case-folded comparison.
- Punctuation (both ASCII and CJK variants) is dropped for CER and mapped to
  spaces for WER, so word boundaries survive.
- CER removes all whitespace (CJK-appropriate); WER splits on whitespace.
- Language choice decides the *primary* metric: CER for zh/yue/ja/ko, WER otherwise.
- No unicode folding beyond ``str.lower()`` is applied: the Russian ``ё``/``е``
  distinction is deliberately kept and must be handled by callers who want
  locale-aware scoring (documented as a known measurement artifact).

The authoritative prose for these rules is docs/METHODOLOGY.md §4.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# ASCII + CJK punctuation (kept in sync with docs/METHODOLOGY.md §4).
# Apostrophes are handled separately (deleted, not turned into word boundaries)
# because they are almost always intra-word in en/fr/de/it contractions.
_APOSTROPHE = re.compile(r"['’]")
_PUNCT = re.compile(r"[，。！？；：、,.!?;:\"()\[\]<>《》„“”«»…\-—–_/\\|~`@#$%^&*+=]+")
# SenseVoice-style tag tokens, e.g. <|zh|><|NEUTRAL|><|withitn|>
_TAG = re.compile(r"<\|[A-Za-z_]+\|>")

CJK_LANGS = frozenset({"zh", "yue", "ja", "ko"})


def strip_tags(text: str) -> str:
    """Remove auxiliary tag tokens emitted by some engines (SenseVoice family)."""
    return _TAG.sub("", text)


def _no_apostrophes(text: str) -> str:
    return _APOSTROPHE.sub("", text)


def normalize_cer(text: str) -> str:
    """Lowercase, drop punctuation and *all* whitespace (character-level input)."""
    t = _PUNCT.sub("", _no_apostrophes(text).strip().lower())
    return re.sub(r"\s+", "", t)


def normalize_words(text: str) -> list[str]:
    """Lowercase, punctuation to spaces, split on whitespace (word-level input)."""
    t = _PUNCT.sub(" ", _no_apostrophes(text).strip().lower())
    return [w for w in re.split(r"\s+", t) if w]


def _levenshtein(a: Sequence, b: Sequence) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str, *, strip_engine_tags: bool = True) -> float:
    """Character error rate after normalization.

    An empty normalized reference scores 1.0 (full error), by convention.
    Values above 1.0 are possible when the hypothesis inserts characters.
    """
    if strip_engine_tags:
        ref, hyp = strip_tags(ref), strip_tags(hyp)
    r, h = normalize_cer(ref), normalize_cer(hyp)
    if not r:
        return 1.0
    return _levenshtein(r, h) / len(r)


def wer(ref: str, hyp: str, *, strip_engine_tags: bool = True) -> float:
    """Word error rate after normalization (same conventions as ``cer``)."""
    if strip_engine_tags:
        ref, hyp = strip_tags(ref), strip_tags(hyp)
    r, h = normalize_words(ref), normalize_words(hyp)
    if not r:
        return 1.0
    return _levenshtein(r, h) / len(r)


def primary_error(ref: str, hyp: str, lang: str) -> float:
    """CER for CJK languages, WER otherwise — the metric reported in all tables."""
    return cer(ref, hyp) if lang.lower() in CJK_LANGS else wer(ref, hyp)
