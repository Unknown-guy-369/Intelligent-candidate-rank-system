import csv
import json
import tempfile
import unittest
from pathlib import Path

from app.scoring import build_reasoning, rank_features, score_candidate


def make_features(candidate_id="CAND_0000001", tier=5, risk="pass", career=1.0):
    return {
        "candidate_id": candidate_id,
        "estimated_tier": tier,
        "profile": {
            "current_title": "Senior Machine Learning Engineer",
            "current_company": "ProductCo",
            "current_industry": "SaaS",
            "years_of_experience": 7.0,
            "location": "Pune",
            "country": "India",
        },
        "signals": {
            "open_to_work": True,
            "recruiter_response_rate": 0.8,
            "notice_period_days": 30,
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "last_active_date": "2026-06-01",
            "inactive_days": 28,
            "github_activity_score": 70,
            "saved_by_recruiters_30d": 4,
        },
        "matches": {
            "python": ["python"],
            "embeddings": ["embeddings"],
            "vector_search": ["faiss"],
            "search_ranking": ["ranking", "retrieval"],
            "evaluation": ["ndcg"],
            "production": ["shipped"],
        },
        "top_relevant_skills": [
            {"name": "Python", "proficiency": "expert", "proficiency_score": 4, "duration_months": 72, "endorsements": 20},
            {"name": "FAISS", "proficiency": "advanced", "proficiency_score": 3, "duration_months": 36, "endorsements": 8},
        ],
        "evidence_snippets": [
            {
                "title": "Senior Machine Learning Engineer",
                "company": "ProductCo",
                "matched_terms": "ranking, retrieval, faiss, ndcg",
                "snippet": "Shipped ranking systems.",
            }
        ],
        "feature_scores": {
            "backend_data": 0.5,
            "behavioral_availability": 0.9,
            "career_evidence": career,
            "embeddings": 1.0,
            "evaluation": 1.0,
            "experience_band": 1.0,
            "llm_finetune": 0.3,
            "location_logistics": 1.0,
            "product_context": 1.0,
            "production": 1.0,
            "python": 1.0,
            "search_ranking": 1.0,
            "semantic_alignment": 1.0,
            "title_alignment": 1.0,
            "vector_search": 1.0,
        },
        "risk": {
            "risk_score": 0 if risk == "pass" else 60,
            "recommendation": risk,
            "flag_names": [] if risk == "pass" else ["very_long_notice_period"],
            "severity_counts": {},
        },
    }


class ScoringTests(unittest.TestCase):
    def test_stronger_evidence_scores_higher(self):
        strong = score_candidate(make_features(career=1.0))
        weak = score_candidate(make_features(candidate_id="CAND_0000002", tier=2, career=0.1))
        self.assertGreater(strong["score"], weak["score"])

    def test_reject_recommendation_scores_zero(self):
        rejected = score_candidate(make_features(tier=0, risk="reject_or_near_zero"))
        self.assertEqual(rejected["score"], 0.0)

    def test_temporal_paradox_flag_scores_zero(self):
        features = make_features()
        features["risk"]["flag_names"] = ["last_active_before_signup"]

        self.assertEqual(score_candidate(features)["score"], 0.0)

    def test_reasoning_uses_profile_evidence(self):
        features = make_features()
        reasoning = build_reasoning(features, score_candidate(features))
        self.assertIn("ProductCo", reasoning)
        self.assertIn("Python", reasoning)
        self.assertNotIn("ranking, retrieval, faiss, ndcg", reasoning.lower())

    def test_very_long_notice_period_is_strong_penalty(self):
        quick = make_features(candidate_id="CAND_0000001")
        slow = make_features(candidate_id="CAND_0000002")
        slow["signals"]["notice_period_days"] = 120
        slow["risk"]["risk_score"] = 30
        slow["risk"]["recommendation"] = "moderate_penalty"
        slow["risk"]["flag_names"] = ["very_long_notice_period"]

        self.assertGreater(score_candidate(quick)["score"] - score_candidate(slow)["score"], 0.2)

    def test_services_only_flag_caps_score(self):
        features = make_features()
        features["risk"]["flag_names"] = ["services_or_consulting_only_history"]

        self.assertLessEqual(score_candidate(features)["score"], 0.62)

    def test_rank_features_writes_sorted_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features_path = root / "features.jsonl"
            rows = [
                make_features(candidate_id="CAND_0000002", tier=3, career=0.4),
                make_features(candidate_id="CAND_0000001", tier=5, career=1.0),
                make_features(candidate_id="CAND_0000003", tier=0, risk="reject_or_near_zero", career=1.0),
            ]
            features_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            output = root / "submission.csv"
            audit = root / "audit.jsonl"

            report = rank_features(features_path, output, audit_path=audit, top_n=3)

            self.assertEqual(report["selected_records"], 3)
            with output.open(newline="", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))
            self.assertEqual(csv_rows[0]["candidate_id"], "CAND_0000001")
            scores = [float(row["score"]) for row in csv_rows]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertTrue(audit.exists())


if __name__ == "__main__":
    unittest.main()
