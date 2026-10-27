"""The checkers. These decide every verdict, and they decide it with
deterministic code, not with another model's opinion.

- ``check_source``: every number in the claim must appear in the cited
  source document next to the same words. A different number next to the
  same words is a contradiction, with the real passage quoted.
- ``check_quote``: the quoted words must appear verbatim in a source
  document, and the attribution must match the source's attribution.
- ``check_compute``: the stated calculation is re-run in the sandbox, its
  inputs are checked against the cited source, and the result is compared
  to the claimed value.

A claim nothing can be checked against is ``unverifiable``. That is a
first-class outcome, not a failure to hide: it means the agent's work
product gave the verifier nothing to stand on.
"""

from __future__ import annotations

import re

from .extract import NUMBER_RE, content_words
from .models import CheckResult, Claim, SourceDoc
from .retrieve import normalize_text
from .tools import ToolRegistry

_NUMBER_SPAN_RE = re.compile(
    r"(?<![A-Za-z0-9])\$?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*%?"
)


def _fmt(value: float) -> str:
    if value == int(value) and abs(value) >= 1000:
        return f"{int(value):,}"
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _numbers_with_spans(text: str) -> list[tuple[float, int, int]]:
    out = []
    for m in _NUMBER_SPAN_RE.finditer(text):
        out.append((float(m.group(1).replace(",", "")), m.start(), m.end()))
    return out


def _same_value(a: float, b: float) -> bool:
    if a == int(a) and b == int(b):
        return a == b
    return abs(a - b) <= 0.01


def _window_words(text: str, start: int, end: int, radius: int = 10) -> set[str]:
    """Content words of the line containing the number. The line is the
    natural unit: a number is checked against the words on its own line,
    not against words lines away that happen to sit within a character
    window."""
    return content_words(_line_at(text, start))


def _match_number(
    value: float, claim_words: set[str], doc_text: str
) -> tuple[str, str | None]:
    """Match one claimed number against one source document.

    Returns (status, evidence_line). Status is 'match', 'conflict', or
    'absent'.
    """
    best_match: tuple[int, str] | None = None
    best_conflict: tuple[int, str] | None = None
    for val, start, end in _numbers_with_spans(doc_text):
        line = _line_at(doc_text, start)
        overlap = len(claim_words & _window_words(doc_text, start, end))
        if _same_value(val, value):
            if best_match is None or overlap > best_match[0]:
                best_match = (overlap, line)
        else:
            if best_conflict is None or overlap > best_conflict[0]:
                best_conflict = (overlap, line)
    if best_match is not None and best_match[0] >= 1:
        return "match", best_match[1]
    if best_conflict is not None and best_conflict[0] >= 2:
        return "conflict", best_conflict[1]
    return "absent", None


def _line_at(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def _doc_texts(sources: list[SourceDoc], citations: list[str]) -> list[SourceDoc]:
    if citations:
        cited = [s for s in sources if s.doc_id in citations]
        if cited:
            return cited
    return sources


def check_source(claim: Claim, tools: ToolRegistry, sources: list[SourceDoc]) -> CheckResult:
    docs = _doc_texts(sources, claim.citations)
    claim_words = content_words(claim.text)
    evidence: list[str] = []
    statuses: list[str] = []
    for value in claim.numbers:
        found: tuple[str, str | None] | None = None
        for doc in docs:
            status, line = _match_number(value, claim_words, doc.text)
            if status == "match":
                found = (status, f"{doc.doc_id} — \"{line}\"")
                break
            if status == "conflict" and found is None:
                found = (status, f"{doc.doc_id} — \"{line}\"")
        if found is None:
            statuses.append("absent")
        else:
            statuses.append(found[0])
            if found[1] and found[1] not in evidence:
                evidence.append(found[1])
    if all(s == "match" for s in statuses):
        return CheckResult(
            idx=claim.idx, claim_text=claim.text, method="source",
            verdict="supported",
            reason="Every number in the claim matches the cited source.",
            evidence=evidence,
        )
    if any(s == "conflict" for s in statuses):
        return CheckResult(
            idx=claim.idx, claim_text=claim.text, method="source",
            verdict="contradicted",
            reason="The source states a different number for the same fact.",
            evidence=evidence,
        )
    top = tools.retrieve_passages(claim.text, top_k=1)
    if top and top[0]["score"] >= 0.5:
        evidence.append(f"Closest passage: {top[0]['doc_id']} — \"{top[0]['text']}\"")
    return CheckResult(
        idx=claim.idx, claim_text=claim.text, method="source",
        verdict="unverifiable",
        reason="No source document states the numbers in this claim.",
        evidence=evidence,
    )


def _attribution_before(text: str, quote_start: int) -> str | None:
    before = text[:quote_start]
    m = re.findall(r"([A-Z][a-z]+ [A-Z][a-z]+)", before)
    return m[-1] if m else None


def check_quote(claim: Claim, tools: ToolRegistry, sources: list[SourceDoc]) -> CheckResult:
    quote = claim.quote or ""
    norm_quote = normalize_text(quote)
    docs = _doc_texts(sources, claim.citations)
    for doc in docs:
        for line in doc.text.splitlines():
            line = line.strip()
            if not line:
                continue
            if norm_quote and norm_quote in normalize_text(line):
                source_attr = _attribution_before(line, line.find('"'))
                evidence = [f"{doc.doc_id} — \"{line}\""]
                if claim.attribution is None:
                    return CheckResult(
                        idx=claim.idx, claim_text=claim.text, method="quote",
                        verdict="supported",
                        reason="The quoted words appear verbatim in the source.",
                        evidence=evidence,
                        detail=f"Source attributes the quote to {source_attr}."
                        if source_attr else "",
                    )
                if source_attr is None:
                    return CheckResult(
                        idx=claim.idx, claim_text=claim.text, method="quote",
                        verdict="supported",
                        reason="The quoted words appear verbatim in the source.",
                        evidence=evidence,
                    )
                if claim.attribution == source_attr:
                    return CheckResult(
                        idx=claim.idx, claim_text=claim.text, method="quote",
                        verdict="supported",
                        reason="The quoted words appear verbatim, with the "
                        "same attribution as the source.",
                        evidence=evidence,
                    )
                return CheckResult(
                    idx=claim.idx, claim_text=claim.text, method="quote",
                    verdict="contradicted",
                    reason=f"The quote is real, but the source attributes it "
                    f"to {source_attr}, not {claim.attribution}.",
                    evidence=evidence,
                )
    # Quote not found verbatim: look for an altered version.
    top = tools.retrieve_passages(quote, top_k=3)
    if top:
        best = top[0]
        qwords = set(normalize_text(quote).split())
        pwords = set(normalize_text(best["text"]).split())
        overlap = len(qwords & pwords) / max(1, len(qwords))
        if overlap >= 0.8:
            return CheckResult(
                idx=claim.idx, claim_text=claim.text, method="quote",
                verdict="contradicted",
                reason="The closest source passage says the quote differently.",
                evidence=[f"{best['doc_id']} — \"{best['text']}\""],
            )
    return CheckResult(
        idx=claim.idx, claim_text=claim.text, method="quote",
        verdict="unverifiable",
        reason="The quoted words do not appear in any source document.",
        evidence=[],
    )


def check_compute(claim: Claim, tools: ToolRegistry, sources: list[SourceDoc]) -> CheckResult:
    formula = claim.formula or ""
    docs = _doc_texts(sources, claim.citations)
    run = tools.recompute(formula)
    if not run["ok"]:
        return CheckResult(
            idx=claim.idx, claim_text=claim.text, method="compute",
            verdict="unverifiable",
            reason=f"Could not re-run the stated calculation: {run['error']}",
            evidence=[],
        )
    result = run["result"]
    # First check the inputs exist in the cited source.
    for value in run["inputs"]:
        found = False
        for doc in docs:
            for val, _, _ in _numbers_with_spans(doc.text):
                if _same_value(val, value):
                    found = True
                    break
            if found:
                break
        if not found:
            return CheckResult(
                idx=claim.idx, claim_text=claim.text, method="compute",
                verdict="unverifiable",
                reason=f"Calculation input {_fmt(value)} is not in the cited "
                "source, so the calculation cannot be checked.",
                evidence=[],
            )
    stated = claim.stated_value
    if stated is None:
        return CheckResult(
            idx=claim.idx, claim_text=claim.text, method="compute",
            verdict="unverifiable",
            reason="The claim states a calculation but no value to check it against.",
            evidence=[f"Re-ran {formula} = {_fmt(result)}"],
        )
    if claim.stated_is_percent:
        computed = result * 100
        ok = abs(computed - stated) <= 0.1
        detail = f"Re-ran {formula} = {computed:.2f}%; claim says {stated}%."
    else:
        computed = result
        ok = abs(computed - stated) <= max(0.01, 0.001 * abs(stated))
        detail = f"Re-ran {formula} = {_fmt(computed)}; claim says {_fmt(stated)}."
    if ok:
        return CheckResult(
            idx=claim.idx, claim_text=claim.text, method="compute",
            verdict="supported",
            reason="The stated calculation re-runs to the claimed value.",
            evidence=[detail],
        )
    return CheckResult(
        idx=claim.idx, claim_text=claim.text, method="compute",
        verdict="contradicted",
        reason="The stated calculation re-runs to a different value.",
        evidence=[detail],
    )


def check_claim(claim: Claim, tools: ToolRegistry, sources: list[SourceDoc]) -> CheckResult:
    if claim.method == "compute":
        return check_compute(claim, tools, sources)
    if claim.method == "quote":
        return check_quote(claim, tools, sources)
    if claim.method == "source":
        return check_source(claim, tools, sources)
    return CheckResult(
        idx=claim.idx, claim_text=claim.text, method="none",
        verdict="unverifiable",
        reason="Nothing in this claim can be checked against a source.",
        evidence=[],
    )
