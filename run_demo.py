"""Demo runner: verify both demo reports end to end and print the measured
results. Exits non-zero if the verdicts do not match the expected ones, so
this is the honest check, not a slide.

Expected:
- flawed report: 8 claims, 5 supported, 2 contradicted, 1 unverifiable,
  overall "has contradictions".
- clean report: 8 claims, 8 supported, overall "verified".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from fresheyes.models import SourceDoc  # noqa: E402
from fresheyes.workflow import verify_work  # noqa: E402

TASK = "Summarize Q3 2026 support operations from the source documents"

SOURCES = [
    ("S1", "support-metrics-q3-2026.txt"),
    ("S2", "incident-notes-q3-2026.txt"),
    ("S3", "survey-comments-q3-2026.txt"),
]


def load_sources() -> list[SourceDoc]:
    out = []
    for doc_id, name in SOURCES:
        text = (BASE / "demo" / "sources" / name).read_text()
        out.append(SourceDoc(doc_id=doc_id, title=name, text=text))
    return out


def run(report_file: str, expected: dict) -> bool:
    report_text = (BASE / "demo" / report_file).read_text()
    report = verify_work(TASK, report_text, load_sources())
    out_dir = BASE / "demo" / "output"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{report_file.replace('.md', '')}-evidence.json").write_text(
        json.dumps(report.to_dict(), indent=2)
    )
    (out_dir / f"{report_file.replace('.md', '')}-evidence.md").write_text(
        report.to_markdown()
    )
    print(f"\n=== {report_file} ===")
    print(f"Overall: {report.overall}  (checked in {report.duration_s:.2f}s)")
    print(f"Counts: {report.counts}")
    for r in report.results:
        print(f"  [{r.method:8}] {r.verdict:13} {r.claim_text}")
        print(f"             reason: {r.reason}")
        for e in r.evidence:
            print(f"             evidence: {e}")
        if r.detail:
            print(f"             detail: {r.detail}")
    ok = True
    for key, want in expected.items():
        got = report.counts.get(key) if key != "overall" else report.overall
        if got != want:
            print(f"  MISMATCH: expected {key}={want}, got {got}")
            ok = False
    return ok


def main() -> int:
    ok1 = run(
        "flawed-report.md",
        {"total": 8, "supported": 5, "contradicted": 2,
         "unverifiable": 1, "overall": "has contradictions"},
    )
    ok2 = run(
        "clean-report.md",
        {"total": 8, "supported": 8, "contradicted": 0,
         "unverifiable": 0, "overall": "verified"},
    )
    print()
    print("DEMO RESULT:", "PASS" if (ok1 and ok2) else "FAIL")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
