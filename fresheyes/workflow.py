"""The checker workflow, as a LangGraph graph.

    extract -> route -> check (one node per claim, fanned out) -> aggregate

Extract pulls atomic claims out of the agent's work product. Route has
already happened inside extraction (the method is chosen by what the claim
carries: a formula, a quote, or numbers with a citation). The check nodes
run in parallel, one per claim, each calling the tools to re-fetch sources
or re-run a calculation. Aggregate turns the per-claim verdicts into the
overall verdict. Deterministic code decides every verdict; the graph is the
orchestration, not the judge.
"""

from __future__ import annotations

import operator
import time
import uuid
from dataclasses import asdict
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .checkers import check_claim
from .extract import extract_claims
from .models import CheckResult, Claim, SourceDoc, VerifyReport
from .tools import ToolRegistry


class VerifyState(TypedDict):
    run_id: str
    task: str
    report_text: str
    sources: list[SourceDoc]
    claims: list[Claim]
    non_claim_sentences: int
    results: Annotated[list[CheckResult], operator.add]
    overall: str
    started_at: float


def _extract(state: VerifyState) -> dict:
    claims, non_claims = extract_claims(state["report_text"])
    return {"claims": claims, "non_claim_sentences": non_claims}


def _route(state: VerifyState) -> list[Send]:
    # A Send target only receives its payload, not the graph state, so the
    # claim and the sources travel in the payload as plain dicts.
    sources = [asdict(s) for s in state["sources"]]
    return [
        Send("check_one", {"claim": asdict(c), "sources": sources})
        for c in state["claims"]
    ]


def _check_one(state: dict) -> dict:
    claim = Claim(**state["claim"])
    sources = [SourceDoc(**s) for s in state["sources"]]
    tools = ToolRegistry(sources)
    result = check_claim(claim, tools, sources)
    return {"results": [result]}


def _aggregate(state: VerifyState) -> dict:
    results = sorted(state["results"], key=lambda r: r.idx)
    if not results:
        overall = "nothing checkable"
    elif all(r.verdict == "supported" for r in results):
        overall = "verified"
    elif any(r.verdict == "contradicted" for r in results):
        overall = "has contradictions"
    else:
        overall = "partially verified"
    # NOTE: do not return "results" here. The channel is an append
    # reducer, so returning the full list would duplicate every result.
    return {"overall": overall}


def build_graph():
    graph = StateGraph(VerifyState)
    graph.add_node("extract", _extract)
    graph.add_node("check_one", _check_one)
    graph.add_node("aggregate", _aggregate)
    graph.add_edge(START, "extract")
    graph.add_conditional_edges("extract", _route, ["check_one"])
    graph.add_edge("check_one", "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile()


def verify_work(task: str, report_text: str, sources: list[SourceDoc]) -> VerifyReport:
    """Run the full workflow on one agent work product."""
    run_id = uuid.uuid4().hex[:12]
    started = time.time()
    graph = build_graph()
    final = graph.invoke(
        {
            "run_id": run_id,
            "task": task,
            "report_text": report_text,
            "sources": sources,
            "claims": [],
            "non_claim_sentences": 0,
            "results": [],
            "overall": "",
            "started_at": started,
        }
    )
    duration = time.time() - started
    results = sorted(final["results"], key=lambda r: r.idx)
    counts = {
        "total": len(results),
        "supported": sum(1 for r in results if r.verdict == "supported"),
        "contradicted": sum(1 for r in results if r.verdict == "contradicted"),
        "unverifiable": sum(1 for r in results if r.verdict == "unverifiable"),
    }
    return VerifyReport(
        run_id=run_id,
        task=task,
        report_text=report_text,
        overall=final["overall"],
        counts=counts,
        results=results,
        non_claim_sentences=final["non_claim_sentences"],
        duration_s=duration,
        sources=sources,
    )
