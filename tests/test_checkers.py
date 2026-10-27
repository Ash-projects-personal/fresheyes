"""Unit tests for FreshEyes's deterministic pieces.

Run: .venv/bin/python -m pytest tests/ -q
(or: .venv/bin/python -m unittest discover -s tests -v)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fresheyes.extract import extract_claims  # noqa: E402
from fresheyes.models import Claim, SourceDoc  # noqa: E402
from fresheyes.sandbox import evaluate, formula_inputs  # noqa: E402
from fresheyes.tools import ToolRegistry  # noqa: E402
from fresheyes.checkers import check_claim  # noqa: E402


class TestSandbox(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertAlmostEqual(evaluate("8,214 / 12,480"), 8214 / 12480)
        self.assertAlmostEqual(evaluate("(3,912 + 2,847)"), 3912 + 2847)
        self.assertAlmostEqual(evaluate("2 * (3 + 4)"), 14.0)

    def test_rejects_code(self):
        for bad in ("__import__('os')", "x + 1", "open('f')", "[1, 2]",
                    "1; 2", "os.system('ls')"):
            with self.assertRaises(ValueError, msg=bad):
                evaluate(bad)

    def test_division_by_zero(self):
        with self.assertRaises(ValueError):
            evaluate("1 / 0")

    def test_formula_inputs(self):
        self.assertEqual(formula_inputs("8,214 / 12,480"), [8214.0, 12480.0])


class TestExtraction(unittest.TestCase):
    def test_claims_and_non_claims(self):
        text = (
            "This report summarizes Q3 support operations.\n\n"
            "Q3 ticket volume was 12,480 [S1].\n"
            "Recommendation: keep staffing levels steady."
        )
        claims, non_claims = extract_claims(text)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].text, "Q3 ticket volume was 12,480.")
        self.assertEqual(claims[0].citations, ["S1"])
        self.assertEqual(claims[0].numbers, [12480.0])
        self.assertEqual(non_claims, 2)

    def test_quote_claim_split_after_closing_quote(self):
        text = (
            "Maya Chen said: \"The failover took too long.\" [S2]\n"
            "Customer churn fell 18% in Q3."
        )
        claims, _ = extract_claims(text)
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0].method, "quote")
        self.assertEqual(claims[0].attribution, "Maya Chen")
        self.assertEqual(claims[1].method, "source")

    def test_compute_claim(self):
        text = "65.8% of tickets were resolved fast (8,214 / 12,480) [S1]."
        claims, _ = extract_claims(text)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].method, "compute")
        self.assertEqual(claims[0].formula, "8,214 / 12,480")
        self.assertEqual(claims[0].stated_value, 65.8)
        self.assertTrue(claims[0].stated_is_percent)


class TestCheckers(unittest.TestCase):
    def setUp(self):
        self.sources = [
            SourceDoc(doc_id="S1", title="metrics",
                      text="Ticket volume was 12,480 tickets.\n"
                      "Login issues totaled 2,847 tickets."),
            SourceDoc(doc_id="S2", title="notes",
                      text="Maya Chen (on-call engineer): "
                      "\"The failover took too long.\""),
        ]
        self.tools = ToolRegistry(self.sources)

    def test_number_match(self):
        claim = Claim(idx=0, text="Ticket volume was 12,480.",
                      method="source", citations=["S1"], numbers=[12480.0])
        self.assertEqual(check_claim(claim, self.tools, self.sources).verdict,
                         "supported")

    def test_number_conflict(self):
        claim = Claim(idx=0, text="Login issues totaled 2,974 tickets.",
                      method="source", citations=["S1"], numbers=[2974.0])
        self.assertEqual(check_claim(claim, self.tools, self.sources).verdict,
                         "contradicted")

    def test_number_absent_is_unverifiable(self):
        claim = Claim(idx=0, text="Customer churn fell 18%.",
                      method="source", citations=[], numbers=[18.0])
        self.assertEqual(check_claim(claim, self.tools, self.sources).verdict,
                         "unverifiable")

    def test_quote_attribution(self):
        good = Claim(idx=0,
                     text="Maya Chen said: \"The failover took too long.\"",
                     method="quote", citations=["S2"],
                     quote="The failover took too long.",
                     attribution="Maya Chen")
        self.assertEqual(check_claim(good, self.tools, self.sources).verdict,
                         "supported")
        bad = Claim(idx=0,
                    text="Priya Nair said: \"The failover took too long.\"",
                    method="quote", citations=["S2"],
                    quote="The failover took too long.",
                    attribution="Priya Nair")
        self.assertEqual(check_claim(bad, self.tools, self.sources).verdict,
                         "contradicted")

    def test_compute_rerun(self):
        claim = Claim(idx=0,
                      text="65.8% were resolved fast (8,214 / 12,480).",
                      method="compute", citations=["S1"],
                      formula="8,214 / 12,480",
                      stated_value=65.8, stated_is_percent=True)
        # Inputs are in S1 (8,214 is not in the small fixture text, so the
        # checker reports unverifiable on missing inputs, not a wrong
        # verdict). Use inputs that exist in the fixture instead.
        claim2 = Claim(idx=0,
                       text="Total tickets were 15,327 (12,480 + 2,847).",
                       method="compute", citations=["S1"],
                       formula="12,480 + 2,847",
                       stated_value=15327.0, stated_is_percent=False)
        self.assertEqual(check_claim(claim2, self.tools, self.sources).verdict,
                         "supported")
        self.assertEqual(check_claim(claim, self.tools, self.sources).verdict,
                         "unverifiable")


if __name__ == "__main__":
    unittest.main()
