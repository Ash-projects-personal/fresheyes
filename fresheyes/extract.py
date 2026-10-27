"""Claim extraction for v1: deterministic, sentence-level, transparent.

An agent work product is a report made of sentences. A sentence becomes an
atomic claim when it states something checkable: it carries a number, a
quoted passage, or a citation to a source document. Intro lines,
recommendations, and framing sentences carry none of those, so they are
counted as non-claim sentences and skipped, not guessed at.

This is deliberately not an LLM extractor. The extraction rules are visible
here, and anything they cannot handle is a known limit, written in the
README. An LLM extractor is a v2 item; it would propose claims, and the
deterministic checkers would still decide the verdicts.
"""

from __future__ import annotations

import re

from .models import Claim

# [S1], [S2] ... citation markers the report uses to point at source docs.
CITATION_RE = re.compile(r"\[(S\d+)\]")
# A parenthesised arithmetic formula, e.g. (8,214 / 12,480) or (3,912 + 2,847).
FORMULA_RE = re.compile(r"\(([\d\s,+\-*/().]+)\)")
# Numbers: 12,480 / 4.31 / 65.8% / $4.2M handled as plain numbers; the
# trailing % is kept as a flag on the claim, not part of the value.
# The lookbehind keeps digits inside tokens like "Q3" or "S1" from matching.
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])\$?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(%)?"
)
# Double-quoted passages.
QUOTE_RE = re.compile(r'"([^"]+)"')
# "Name Name ... said" attribution before a quote.
ATTRIBUTION_RE = re.compile(r"([A-Z][a-z]+ [A-Z][a-z]+)[^\"<>]{0,60}?\bsaid\b")
# "said Name Name" attribution after a quote.
ATTRIBUTION_AFTER_RE = re.compile(r"\bsaid\s+([A-Z][a-z]+ [A-Z][a-z]+)")

# Words that carry no meaning for keyword overlap.
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "was", "were", "is", "are", "by", "at", "from", "after", "under",
    "over", "than", "that", "this", "its", "it", "as", "out", "across",
    "per", "each", "all", "total", "tickets", "ticket",
}

_SENTENCE_END_RE = re.compile(r"([.!?][\"']?)\s+(?=[A-Z0-9\"'\(])")


def _split_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Split text into sentences, returning (sentence, start, end) spans."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _SENTENCE_END_RE.finditer(text):
        end = m.start() + len(m.group(1))  # end of the terminator
        spans.append((pos, end))
        pos = m.end()  # just past the whitespace after the terminator
    if pos < len(text):
        spans.append((pos, len(text)))
    out: list[tuple[str, int, int]] = []
    for start, end in spans:
        chunk = text[start:end]
        stripped = chunk.strip()
        if stripped:
            lead = len(chunk) - len(chunk.lstrip())
            out.append((stripped, start + lead, end))
    return out


def split_sentences(text: str) -> list[tuple[str, list[str]]]:
    """Split report text into (sentence, citations) pairs.

    Citation markers like [S1] sit between a sentence's terminator and the
    next sentence, so they are first replaced with equal-length spaces:
    that keeps every character offset intact (so citations can be mapped
    back to their sentence by span) while letting the sentence splitter see
    a clean boundary, and it stops the "1" in "[S1]" being read as a number.
    """
    lines = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        if raw.startswith("#"):
            continue  # title / headings are not claims
        if raw.lower().startswith("task:"):
            continue  # the task line is metadata, not a claim
        lines.append(raw)
    joined = " ".join(lines)
    cites = [(m.start(), m.group(1)) for m in CITATION_RE.finditer(joined)]
    chars = list(joined)
    for start, _ in cites:
        end = joined.find("]", start)
        for i in range(start, end + 1):
            chars[i] = " "
    masked = "".join(chars)
    out: list[tuple[str, list[str]]] = []
    for sent, s0, s1 in _split_with_spans(masked):
        sent_cites = [c for (off, c) in cites if s0 <= off < s1]
        out.append((sent, sent_cites))
    return out


def _parse_number(raw: str) -> float:
    return float(raw.replace(",", "").replace("$", ""))


def extract_claims(report_text: str) -> tuple[list[Claim], int]:
    """Extract atomic claims from a report.

    Returns (claims, non_claim_sentence_count).
    """
    sentences = split_sentences(report_text)
    claims: list[Claim] = []
    non_claims = 0
    for sent, citations in sentences:
        # Citations were already masked with spaces by split_sentences;
        # clean up the leftover spacing and stray punctuation.
        clean = re.sub(r"\s+([.,])", r"\1", sent)
        clean = re.sub(r"\s+", " ", clean).strip()
        quote_m = QUOTE_RE.search(clean)
        formula_m = None
        for m in FORMULA_RE.finditer(clean):
            body = m.group(1)
            if any(op in body for op in ("+", "-", "*", "/")) and re.search(r"\d", body):
                formula_m = m
                break
        numbers = NUMBER_RE.findall(clean)

        has_number = bool(numbers)
        has_quote = quote_m is not None
        has_citation = bool(citations)
        if not (has_number or has_quote or has_citation):
            non_claims += 1
            continue

        attribution = None
        if quote_m:
            before = clean[: quote_m.start()]
            am = ATTRIBUTION_RE.search(before)
            if am:
                attribution = am.group(1)
            else:
                after = clean[quote_m.end():]
                am2 = ATTRIBUTION_AFTER_RE.search(after)
                if am2:
                    attribution = am2.group(1)

        stated_value = None
        stated_is_percent = False
        if formula_m:
            # The stated value is the headline value the formula is meant
            # to prove: a percent if the claim carries one, else the first
            # number before the formula.
            before = clean[: formula_m.start()]
            nums_before = NUMBER_RE.findall(before)
            pct = [n for n in nums_before if n[1] == "%"]
            pick = pct[-1] if pct else (nums_before[0] if nums_before else None)
            if pick:
                stated_value = _parse_number(pick[0])
                stated_is_percent = pick[1] == "%"

        claim = Claim(
            idx=len(claims),
            text=clean,
            citations=citations,
            numbers=[_parse_number(raw) for raw, _ in numbers],
            quote=quote_m.group(1) if quote_m else None,
            attribution=attribution,
            formula=formula_m.group(1) if formula_m else None,
            stated_value=stated_value,
            stated_is_percent=stated_is_percent,
        )
        claims.append(claim)

    # Route each claim to its verification method.
    for claim in claims:
        if claim.formula is not None:
            claim.method = "compute"
        elif claim.quote is not None:
            claim.method = "quote"
        elif claim.numbers or claim.citations:
            claim.method = "source"
        else:
            claim.method = "none"
    return claims, non_claims


def content_words(text: str) -> set[str]:
    """Lowercased content words, stopwords removed, for keyword overlap."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}
