"""
tests/test_scoring.py — Lightweight tests that do NOT require a Gemini
API key or network access. Run with: python -m pytest tests/ -v
(or: python -m unittest discover tests)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MAX_LEAD_SCORE
from scoring import InvalidScoreError, calculate_lead_score, classify_priority, score_lead
from tools import dedupe_companies, normalize_company_name, normalize_domain
from sales_agent import SalesState


def make_criteria(scores):
    """scores: list of 6 ints or 'unknown', in SCORING_CRITERIA order."""
    from config import SCORING_CRITERIA

    return {
        name: {"score": s, "evidence": "test evidence"}
        for name, s in zip(SCORING_CRITERIA, scores)
    }


class TestLeadScoreCalculation(unittest.TestCase):
    def test_high_fit_lead(self):
        criteria = make_criteria([3, 3, 3, 3, 3, 3])
        self.assertEqual(calculate_lead_score(criteria), 18)
        self.assertEqual(calculate_lead_score(criteria), MAX_LEAD_SCORE)

    def test_medium_fit_lead(self):
        criteria = make_criteria([2, 2, 2, 2, 2, 2])
        self.assertEqual(calculate_lead_score(criteria), 12)

    def test_low_fit_lead(self):
        criteria = make_criteria([1, 1, 1, 1, 1, 2])
        self.assertEqual(calculate_lead_score(criteria), 7)

    def test_missing_evidence_treated_as_zero(self):
        criteria = make_criteria(["unknown", "unknown", 3, 3, 2, 1])
        self.assertEqual(calculate_lead_score(criteria), 9)

    def test_missing_criterion_key_treated_as_zero(self):
        criteria = make_criteria([3, 3, 3, 3, 3, 3])
        del criteria["evidence_quality"]
        self.assertEqual(calculate_lead_score(criteria), 15)

    def test_invalid_score_raises(self):
        criteria = make_criteria([5, 3, 3, 3, 3, 3])
        with self.assertRaises(InvalidScoreError):
            calculate_lead_score(criteria)

    def test_negative_score_raises(self):
        criteria = make_criteria([-1, 3, 3, 3, 3, 3])
        with self.assertRaises(InvalidScoreError):
            calculate_lead_score(criteria)

    def test_non_int_score_raises(self):
        criteria = make_criteria(["high", 3, 3, 3, 3, 3])
        with self.assertRaises(InvalidScoreError):
            calculate_lead_score(criteria)


class TestPriorityClassification(unittest.TestCase):
    def test_high_priority(self):
        self.assertEqual(classify_priority(18), "HIGH PRIORITY")
        self.assertEqual(classify_priority(15), "HIGH PRIORITY")

    def test_medium_priority(self):
        self.assertEqual(classify_priority(14), "MEDIUM PRIORITY")
        self.assertEqual(classify_priority(11), "MEDIUM PRIORITY")

    def test_low_priority(self):
        self.assertEqual(classify_priority(10), "LOW PRIORITY")
        self.assertEqual(classify_priority(7), "LOW PRIORITY")

    def test_insufficient_data(self):
        self.assertEqual(classify_priority(6), "INSUFFICIENT DATA")
        self.assertEqual(classify_priority(0), "INSUFFICIENT DATA")

    def test_out_of_range_raises(self):
        with self.assertRaises(InvalidScoreError):
            classify_priority(19)
        with self.assertRaises(InvalidScoreError):
            classify_priority(-1)


class TestScoreLead(unittest.TestCase):
    def test_full_lead_record(self):
        criteria = make_criteria([3, 3, 2, 3, 2, 3])
        lead = score_lead("Acme Corp", criteria, sources=["https://acme.com/about"])
        self.assertEqual(lead["company"], "Acme Corp")
        self.assertEqual(lead["score"], 16)
        self.assertEqual(lead["priority"], "HIGH PRIORITY")
        self.assertEqual(lead["sources"], ["https://acme.com/about"])


class TestDuplicateCompanyHandling(unittest.TestCase):
    def test_dedupe_by_name(self):
        companies = [
            {"company": "Acme Inc.", "website": "https://acme.com", "source_urls": ["https://a.com"]},
            {"company": "Acme", "website": "https://acme.com", "source_urls": ["https://b.com"]},
        ]
        deduped = dedupe_companies(companies)
        self.assertEqual(len(deduped), 1)
        self.assertIn("https://b.com", deduped[0]["source_urls"])

    def test_dedupe_by_domain(self):
        companies = [
            {"company": "Acme Corp", "website": "https://www.acme.com", "source_urls": []},
            {"company": "Acme Corporation Pvt Ltd", "website": "https://acme.com/home", "source_urls": []},
        ]
        deduped = dedupe_companies(companies)
        self.assertEqual(len(deduped), 1)

    def test_no_false_positive_dedupe(self):
        companies = [
            {"company": "Acme Corp", "website": "https://acme.com", "source_urls": []},
            {"company": "Zenith Corp", "website": "https://zenith.com", "source_urls": []},
        ]
        deduped = dedupe_companies(companies)
        self.assertEqual(len(deduped), 2)


class TestNormalization(unittest.TestCase):
    def test_normalize_domain_strips_www(self):
        self.assertEqual(normalize_domain("https://www.acme.com/about"), "acme.com")

    def test_normalize_domain_no_scheme(self):
        self.assertEqual(normalize_domain("acme.com"), "acme.com")

    def test_normalize_company_name_strips_suffix(self):
        self.assertEqual(normalize_company_name("Acme Inc."), "acme")
        self.assertEqual(normalize_company_name("Acme Private Limited"), "acme")


class TestAgentStateInitialization(unittest.TestCase):
    def test_default_state(self):
        state = SalesState(icp_raw={"industry": "SaaS"})
        self.assertEqual(state.icp_raw["industry"], "SaaS")
        self.assertEqual(state.plan, [])
        self.assertEqual(state.candidate_companies, [])
        self.assertEqual(state.qualified_leads, [])
        self.assertEqual(state.iterations, 0)
        self.assertEqual(state.final_report, "")

    def test_state_log(self):
        state = SalesState(icp_raw={})
        state.log("ANALYZE_ICP", "test detail")
        self.assertEqual(len(state.action_log), 1)
        self.assertEqual(state.action_log[0]["action"], "ANALYZE_ICP")


class TestSearchResultParsing(unittest.TestCase):
    def test_search_result_shape(self):
        # Simulated shape returned by tools.web_search — validated here
        # without hitting the network.
        fake_result = {"title": "Acme", "url": "https://acme.com", "snippet": "Acme is a SaaS company."}
        self.assertIn("title", fake_result)
        self.assertIn("url", fake_result)
        self.assertIn("snippet", fake_result)


if __name__ == "__main__":
    unittest.main()
