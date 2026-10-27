"""SQLite persistence for verification runs. Plain sqlite3, no ORM."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import CheckResult, VerifyReport

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    report_text TEXT NOT NULL,
    overall TEXT NOT NULL,
    counts_json TEXT NOT NULL,
    non_claim_sentences INTEGER NOT NULL DEFAULT 0,
    duration_s REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    idx INTEGER NOT NULL,
    claim_text TEXT NOT NULL,
    method TEXT NOT NULL,
    verdict TEXT NOT NULL,
    reason TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT ''
);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.executescript(SCHEMA)

    def save(self, report: VerifyReport) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(run_id, task, report_text, overall, counts_json, "
                "non_claim_sentences, duration_s) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    report.run_id,
                    report.task,
                    report.report_text,
                    report.overall,
                    json.dumps(report.counts),
                    report.non_claim_sentences,
                    report.duration_s,
                ),
            )
            conn.execute("DELETE FROM claims WHERE run_id = ?", (report.run_id,))
            for r in report.results:
                conn.execute(
                    "INSERT INTO claims (run_id, idx, claim_text, method, "
                    "verdict, reason, evidence_json, detail) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        report.run_id,
                        r.idx,
                        r.claim_text,
                        r.method,
                        r.verdict,
                        r.reason,
                        json.dumps(r.evidence),
                        r.detail,
                    ),
                )

    def get(self, run_id: str) -> VerifyReport | None:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            claim_rows = conn.execute(
                "SELECT * FROM claims WHERE run_id = ? ORDER BY idx",
                (run_id,),
            ).fetchall()
        results = [
            CheckResult(
                idx=c["idx"],
                claim_text=c["claim_text"],
                method=c["method"],
                verdict=c["verdict"],
                reason=c["reason"],
                evidence=json.loads(c["evidence_json"]),
                detail=c["detail"],
            )
            for c in claim_rows
        ]
        counts = json.loads(row["counts_json"])
        return VerifyReport(
            run_id=row["run_id"],
            task=row["task"],
            report_text=row["report_text"],
            overall=row["overall"],
            counts=counts,
            results=results,
            non_claim_sentences=row["non_claim_sentences"],
            duration_s=row["duration_s"],
        )

    def list_runs(self, limit: int = 20) -> list[dict]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT run_id, task, overall, counts_json, created_at "
                "FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "run_id": r["run_id"],
                "task": r["task"],
                "overall": r["overall"],
                "counts": json.loads(r["counts_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
