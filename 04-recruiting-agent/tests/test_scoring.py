import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config, scoring

def test_weights_sum_to_one():
    assert abs(sum(config.SCORING_WEIGHTS.values()) - 1.0) < 0.001

def test_skill_score_all_match():
    assert scoring.calculate_skill_score([{"status": "MATCH"}, {"status": "MATCH"}], 0.5) == 50.0

def test_total_score_never_exceeds_100():
    components = {k: 999 for k in config.SCORING_WEIGHTS}
    breakdown = scoring.calculate_total_score(components)
    assert breakdown["total"] <= 100

def test_shortlist_respects_coverage_threshold():
    evaluations = [
        {"candidate_id": "a", "score_breakdown": {"total": 95}, "mandatory_coverage": 0.4},
        {"candidate_id": "b", "score_breakdown": {"total": 80}, "mandatory_coverage": 0.8},
    ]
    shortlist = scoring.create_shortlist(evaluations, size=3, min_coverage=0.7)
    ids = [e["candidate_id"] for e in shortlist]
    assert "a" not in ids and "b" in ids

def _run_all():
    tests = [o for n, o in list(globals().items()) if n.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  ✓ {t.__name__}")
    print(f"\n{passed} passed")

if __name__ == "__main__":
    _run_all()
