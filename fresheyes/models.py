"""FreshEyes: verify agent work, not web text.

Data model for a verification run. Everything here is plain data so the
workflow, the store, and the web app can share it without surprises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Verdict = Literal["supported", "contradicted", "unverifiable"]
Method = Literal["source", "quote", "compute", "none"]


@dataclass
class SourceDoc:
    """One of the agent's own source documents the work product was built from."""

    doc_id: str
    title: str
    text: str


@dataclass
class Claim:
    """One atomic claim extracted from an agent work product."""

    idx: int
    text: str
    method: Method = "none"
    citations: list[str] = field(default_factory=list)
    numbers: list[float] = field(default_factory=list)
    quote: str | None = None
    attribution: str | None = None
    formula: str | None = None
    stated_value: float | None = None
    stated_is_percent: bool = False


@dataclass
class CheckResult:
    """The verdict for one claim, with the evidence a human can audit."""

    idx: int
    claim_text: str
    method: Method
    verdict: Verdict
    reason: str
    evidence: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class VerifyReport:
    """The full result of verifying one agent work product."""

    run_id: str
    task: str
    report_text: str
    overall: str
    counts: dict[str, int]
    results: list[CheckResult] = field(default_factory=list)
    non_claim_sentences: int = 0
    duration_s: float = 0.0
    sources: list[SourceDoc] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "overall": self.overall,
            "counts": self.counts,
            "duration_s": round(self.duration_s, 2),
            "non_claim_sentences": self.non_claim_sentences,
            "results": [
                {
                    "idx": r.idx,
                    "claim": r.claim_text,
                    "method": r.method,
                    "verdict": r.verdict,
                    "reason": r.reason,
                    "evidence": r.evidence,
                    "detail": r.detail,
                }
                for r in self.results
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            "# FreshEyes evidence file",
            "",
            f"Task: {self.task}",
            "",
            f"Overall verdict: {self.overall}",
            "",
            f"Claims checked: {self.counts.get('total', 0)} "
            f"({self.counts.get('supported', 0)} supported, "
            f"{self.counts.get('contradicted', 0)} contradicted, "
            f"{self.counts.get('unverifiable', 0)} unverifiable). "
            f"Checked in {self.duration_s:.1f} seconds.",
            "",
            "Every verdict below comes with the evidence it was decided on. "
            "Nothing is marked supported without a quoted source passage or a "
            "re-executed calculation.",
            "",
        ]
        for r in self.results:
            lines.append(f"## Claim {r.idx + 1}: {r.verdict.upper()} [{r.method}]")
            lines.append("")
            lines.append(f"> {r.claim_text}")
            lines.append("")
            lines.append(f"Reason: {r.reason}")
            lines.append("")
            for e in r.evidence:
                lines.append(f"- Evidence: {e}")
            if r.detail:
                lines.append(f"- Detail: {r.detail}")
            lines.append("")
        return "\n".join(lines)
