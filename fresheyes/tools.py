"""Tools the checker workflow calls. These are the only ways the workflow
touches the source documents: it re-fetches from the agent's own documents
instead of trusting the report's copy of the facts.

- ``source_lookup``: open one cited document and return the lines matching
  a query (the "re-fetch from the system of record" tool).
- ``retrieve_passages``: TF-IDF retrieval over all source documents.
- ``recompute``: run an arithmetic expression in the sandbox.
"""

from __future__ import annotations

from .models import SourceDoc
from .retrieve import Retriever
from .sandbox import evaluate, formula_inputs


class ToolRegistry:
    def __init__(self, sources: list[SourceDoc]):
        self.sources = {s.doc_id: s for s in sources}
        self.retriever = Retriever(sources)

    def source_lookup(self, doc_id: str, query: str) -> list[str]:
        """Return the lines of one cited document that mention the query's
        content words, in document order."""
        doc = self.sources.get(doc_id)
        if doc is None:
            return []
        from .extract import content_words

        qwords = content_words(query)
        hits = []
        for line in doc.text.splitlines():
            line = line.strip()
            if not line:
                continue
            lwords = content_words(line)
            if qwords & lwords:
                hits.append(line)
        return hits

    def retrieve_passages(self, query: str, top_k: int = 3) -> list[dict]:
        return [
            {"doc_id": p.doc_id, "title": p.title, "text": p.text, "score": p.score}
            for p in self.retriever.search(query, top_k=top_k)
        ]

    def recompute(self, expression: str) -> dict:
        try:
            result = evaluate(expression)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "result": None}
        return {
            "ok": True,
            "error": None,
            "result": result,
            "inputs": formula_inputs(expression),
        }
