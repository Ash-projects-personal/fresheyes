"""FreshEyes web app: paste an agent's finished work, get a verdict.

One page to submit (task + report text, or file uploads, plus the source
documents the agent worked from), one verdict page per run with a per-claim
table and expandable evidence, and a downloadable evidence file (JSON or
Markdown) a human can audit. Black and white, simple, no decoration.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse

from .models import SourceDoc
from .store import Store
from .workflow import verify_work

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
STORE = Store(DATA_DIR / "fresheyes.db")

app = FastAPI(title="FreshEyes", version="1.0.0")

CSS = """
body{font-family:system-ui,-apple-system,sans-serif;max-width:860px;margin:40px auto;
padding:0 20px;color:#000;background:#fff;line-height:1.5}
h1{font-size:26px}h2{font-size:19px;margin-top:28px}
textarea,input[type=text]{width:100%;padding:8px;border:1px solid #000;
font-family:inherit;font-size:15px;box-sizing:border-box}
textarea{height:220px}
button{padding:10px 18px;background:#000;color:#fff;border:0;font-size:15px;cursor:pointer}
table{width:100%;border-collapse:collapse;margin-top:12px}
th,td{border:1px solid #ccc;padding:8px;text-align:left;vertical-align:top}
th{background:#f0f0f0}
.supported{color:#0a6b1f;font-weight:700}
.contradicted{color:#a00;font-weight:700}
.unverifiable{color:#7a5b00;font-weight:700}
.banner{border:2px solid #000;padding:14px 18px;font-size:18px;margin:16px 0}
details{margin-top:6px}
summary{cursor:pointer}
small{color:#555}
a{color:#000}
"""

INDEX_HTML = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>FreshEyes — verify agent work, not web text</title>
<style>{CSS}</style></head><body>
<h1>FreshEyes</h1>
<p><strong>Verify agent work, not web text.</strong> Give me an agent's
finished work product and the source documents it was built from. I break
the work into atomic claims, re-run its calculations, re-fetch its sources,
and return a verdict per claim, with the evidence quoted. Deterministic
re-execution is the authority here, not another model's opinion.</p>
<form action="/verify" method="post" enctype="multipart/form-data">
<label>Task the agent was given<br>
<input type="text" name="task" value="Summarize Q3 2026 support operations from the source documents"></label><br><br>
<label>Agent's finished work (paste the report)<br>
<textarea name="report_text"></textarea></label><br><br>
<label>Or upload the report as a file<br>
<input type="file" name="report_file"></label><br><br>
<label>Source documents (upload one or more .txt / .md files)<br>
<input type="file" name="source_files" multiple></label><br><br>
<button type="submit">Verify this work</button>
</form>
<p><small>What gets checked: numbers against the cited source, quoted words
against the source's wording and attribution, stated calculations by
re-running them. A claim nothing can be checked against is marked
unverifiable, not guessed at. Nothing is marked supported without a quoted
source passage or a re-executed calculation.</small></p>
<p><small><a href="/runs">Past verification runs</a></small></p>
</body></html>"""


def _verdict_page(report) -> str:
    rows = ""
    for r in report.results:
        ev = "".join(f"<li>{e}</li>" for e in r.evidence)
        detail = f"<p><small>{r.detail}</small></p>" if r.detail else ""
        rows += (
            f"<tr><td>{r.idx + 1}</td><td>{r.claim_text}</td>"
            f"<td>{r.method}</td>"
            f"<td class=\"{r.verdict}\">{r.verdict}</td>"
            f"<td>{r.reason}{detail}"
            f"<details><summary>Evidence</summary><ul>{ev or '<li>None found.</li>'}</ul>"
            f"</details></td></tr>"
        )
    c = report.counts
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>FreshEyes verdict — {report.run_id}</title>
<style>{CSS}</style></head><body>
<h1>FreshEyes verdict</h1>
<p><small>Task: {report.task}</small></p>
<div class="banner">Overall verdict: <strong>{report.overall}</strong><br>
<small>{c.get('total', 0)} claims checked: {c.get('supported', 0)} supported,
{c.get('contradicted', 0)} contradicted, {c.get('unverifiable', 0)} unverifiable.
{report.non_claim_sentences} non-claim sentences skipped.
Checked in {report.duration_s:.1f} seconds.</small></div>
<table><tr><th>#</th><th>Claim</th><th>Method</th><th>Verdict</th><th>Reason and evidence</th></tr>
{rows}</table>
<p><a href="/runs/{report.run_id}/evidence.json">Download evidence file (JSON)</a> |
<a href="/runs/{report.run_id}/evidence.md">Download evidence file (Markdown)</a> |
<a href="/">Verify another work product</a></p>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"status": "ok", "service": "fresheyes"}


@app.post("/verify", response_class=HTMLResponse)
async def verify_form(
    request: Request,
    task: str = Form(...),
    report_text: str = Form(""),
    report_file: UploadFile | None = File(None),
    source_files: list[UploadFile] = File(...),
):
    text = report_text.strip()
    if report_file is not None and report_file.filename:
        text = (await report_file.read()).decode("utf-8", "replace")
    if not text:
        raise HTTPException(400, "No report text or report file given.")
    sources = []
    for i, f in enumerate(source_files):
        if not f.filename:
            continue
        content = (await f.read()).decode("utf-8", "replace")
        doc_id = f"S{i + 1}"
        sources.append(SourceDoc(doc_id=doc_id, title=f.filename, text=content))
    if not sources:
        raise HTTPException(400, "Upload at least one source document.")
    report = verify_work(task, text, sources)
    STORE.save(report)
    return _verdict_page(report)


@app.post("/api/verify")
async def verify_api(payload: dict):
    task = payload.get("task", "")
    text = payload.get("report_text", "")
    raw_sources = payload.get("sources", [])
    if not text or not raw_sources:
        raise HTTPException(400, "report_text and sources are required.")
    sources = [
        SourceDoc(doc_id=s.get("doc_id", f"S{i + 1}"), title=s.get("title", ""),
                  text=s.get("text", ""))
        for i, s in enumerate(raw_sources)
    ]
    report = verify_work(task, text, sources)
    STORE.save(report)
    return report.to_dict()


@app.get("/runs", response_class=HTMLResponse)
def list_runs():
    runs = STORE.list_runs()
    items = "".join(
        f"<li><a href=\"/runs/{r['run_id']}\">{r['run_id']}</a> — "
        f"{r['task']} — <strong>{r['overall']}</strong> "
        f"({r['counts'].get('supported', 0)}/{r['counts'].get('total', 0)} supported) "
        f"<small>{r['created_at']}</small></li>"
        for r in runs
    )
    return (f"<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
            f"<title>FreshEyes runs</title><style>{CSS}</style></head><body>"
            f"<h1>Past verification runs</h1><ul>{items or '<li>None yet.</li>'}</ul>"
            f"<p><a href=\"/\">Verify another work product</a></p></body></html>")


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def get_run(run_id: str):
    report = STORE.get(run_id)
    if report is None:
        raise HTTPException(404, "Run not found.")
    return _verdict_page(report)


@app.get("/runs/{run_id}/evidence.json")
def get_evidence_json(run_id: str):
    report = STORE.get(run_id)
    if report is None:
        raise HTTPException(404, "Run not found.")
    return report.to_dict()


@app.get("/runs/{run_id}/evidence.md", response_class=PlainTextResponse)
def get_evidence_md(run_id: str):
    report = STORE.get(run_id)
    if report is None:
        raise HTTPException(404, "Run not found.")
    return report.to_markdown()


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
