# FreshEyes

**Verify agent work, not web text.**

I built this because agents now write reports and analyses that no human has
time to re-check by hand, and the usual answer, one model grading another
model's output, is weak. FreshEyes takes an agent's finished work product
and the source documents it was built from, breaks the work into atomic
claims, and checks each one by re-executing the work: it re-runs the stated
calculations in a sandbox, re-fetches the cited sources, and quotes the
matching passage. Every verdict comes with the evidence it was decided on.
Nothing is marked supported without a quoted source passage or a re-executed
calculation.

## What makes it different

The atomic-claim verification pipeline exists already. FACTS&EVIDENCE (NAACL
2025 demo) breaks long text into atomic claims and checks each against the
web, and I credit it below for the pipeline shape. What FreshEyes does that
it does not: the input is an **agent's work product**, not an arbitrary
document. The checks run against the **agent's own source documents**, not
the web. Calculations are **re-run**, not re-read. And the deliverable is an
**evidence file a human can audit**, not a credibility score. If this ever
becomes "paste text, get a score," it has missed the point.

Deterministic re-execution is the authority here, not another model's
opinion. There is no LLM in v1 at all. Claim extraction is sentence-level
and rule-based, and every verdict is decided by code you can read in
`fresheyes/checkers.py`. That is a design choice, not a missing piece: it
means every verdict in the demo is reproducible, and nothing is ever graded
by a model that could itself be wrong.

## How it works

1. **Extract.** The report is split into sentences. A sentence becomes an
   atomic claim when it states something checkable: a number, a quoted
   passage, or a citation like `[S1]` pointing at a source document. Intro
   lines and recommendations carry none of those, so they are counted as
   non-claim sentences and skipped.
2. **Route.** Each claim gets a method from what it carries: a parenthesised
   formula goes to the sandbox, a quoted passage goes to the quote checker,
   numbers with a citation go to the source checker.
3. **Check (LangGraph).** A LangGraph graph fans out one check node per claim
   (`extract -> route -> check -> aggregate`). Each check node calls tools:
   `source_lookup` re-fetches lines from the cited document,
   `retrieve_passages` runs TF-IDF retrieval over the agent's own documents,
   `recompute` runs the stated calculation in an AST sandbox that only
   allows plain arithmetic.
4. **Verdict.** Per claim: `supported`, `contradicted`, or `unverifiable`.
   Overall: `verified` only if every claim is supported; `has
   contradictions` if any claim is contradicted; `partially verified`
   otherwise. Unverifiable is a first-class outcome: it means the work gave
   the verifier nothing to stand on, and the system says so instead of
   guessing.

## Measured results (real runs, not typed by hand)

Run `python run_demo.py`. It verifies both demo reports end to end and
fails if the verdicts do not match. On my machine, 2026-10-25:

- **Flawed report** (8 claims, 3 planted errors): 5 supported, 2
  contradicted, 1 unverifiable. Overall: has contradictions. Checked in
  0.08 seconds. The wrong number (login issues claimed 2,974, source says
  2,847) is contradicted with the real passage quoted. The misattributed
  quote (credited to Priya Nair, source says Maya Chen) is contradicted
  with the source line quoted. The claim with no source (churn fell 18%)
  is unverifiable.
- **Clean report** (8 claims): 8 supported. Overall: verified. Checked in
  0.07 seconds. The same system says yes and no.

The demo evidence files those runs produced are in `demo/output/`.

## Run it

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_demo.py          # demo: both reports, asserts verdicts
.venv/bin/python -m unittest discover -s tests -v   # 12 unit tests
.venv/bin/python -m uvicorn fresheyes.app:app --host 127.0.0.1 --port 8001
```

Then open http://127.0.0.1:8001. Paste a task and a report, upload the source
documents, and get a verdict page with a per-claim table and expandable
evidence, plus downloadable evidence files (JSON and Markdown). Runs are
stored in SQLite at `data/fresheyes.db`. There is also a JSON API:
`POST /api/verify` with `task`, `report_text`, and `sources`.

## What it does not do (limits, stated plainly)

- Claim extraction is sentence-level and rule-based in v1. It handles
  numbers, quotes, `[S1]`-style citations, and parenthesised formulas. A
  claim phrased without any of those is skipped, and the run reports how
  many sentences were skipped.
- It can only check claims that are checkable. A claim about a private
  conversation or a feeling is unverifiable, and the system says so.
- It checks the work against the documents you give it. If the agent's
  sources are wrong, FreshEyes will faithfully confirm the wrong numbers.
  Garbage in, verified garbage out.
- It does not fix agents. It finds where their work breaks, with evidence.
- English text only in v1.

## Credits

The atomic-claim pipeline shape (break a text into atomic claims, check each
against evidence, return per-claim verdicts) follows FACTS&EVIDENCE, the
NAACL 2025 demo system. FreshEyes applies that shape to a different
problem: verifying an agent's finished work against its own sources, with
deterministic re-execution, instead of verifying general text against the
web.
