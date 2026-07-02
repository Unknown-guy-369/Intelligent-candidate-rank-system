import json
import tempfile
import unittest
from pathlib import Path

from app.features import extract_candidate_features, extract_file
from app.preprocess import clean_candidate
from app.risk import assess_candidate_risk


class FeatureTests(unittest.TestCase):
    def make_candidate(self, title="Senior ML Engineer", description=None):
        raw = {
            "candidate_id": "CAND_0000001",
            "profile": {
                "anonymized_name": "Test User",
                "headline": f"{title} | Python, FAISS, Search",
                "summary": "Senior engineer building production retrieval systems for real users.",
                "location": "Pune",
                "country": "India",
                "years_of_experience": 7.0,
                "current_title": title,
                "current_company": "ProductCo",
                "current_company_size": "201-500",
                "current_industry": "SaaS",
            },
            "career_history": [
                {
                    "company": "ProductCo",
                    "title": title,
                    "start_date": "2020-01-01",
                    "end_date": None,
                    "duration_months": 72,
                    "is_current": True,
                    "industry": "SaaS",
                    "company_size": "201-500",
                    "description": description
                    or "Shipped production vector search and recommendation ranking systems using Python, FAISS, embeddings, NDCG, MRR, and online evaluation.",
                }
            ],
            "education": [],
            "skills": [
                {"name": "Python", "proficiency": "expert", "endorsements": 15, "duration_months": 72},
                {"name": "FAISS", "proficiency": "advanced", "endorsements": 8, "duration_months": 36},
                {"name": "Embeddings", "proficiency": "advanced", "endorsements": 8, "duration_months": 36},
            ],
            "certifications": [],
            "languages": [],
            "redrob_signals": {
                "profile_completeness_score": 95,
                "signup_date": "2025-01-01",
                "last_active_date": "2026-06-01",
                "open_to_work_flag": True,
                "profile_views_received_30d": 10,
                "applications_submitted_30d": 1,
                "recruiter_response_rate": 0.8,
                "avg_response_time_hours": 6,
                "skill_assessment_scores": {},
                "connection_count": 50,
                "endorsements_received": 20,
                "notice_period_days": 30,
                "expected_salary_range_inr_lpa": {"min": 30, "max": 45},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": 70,
                "search_appearance_30d": 20,
                "saved_by_recruiters_30d": 4,
                "interview_completion_rate": 0.9,
                "offer_acceptance_rate": 0.7,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True,
            },
        }
        cleaned, notes = clean_candidate(raw)
        self.assertIsNotNone(cleaned, notes)
        cleaned["_risk"] = assess_candidate_risk(cleaned)
        return cleaned

    def test_strong_retrieval_profile_extracts_tier_five_features(self):
        features = extract_candidate_features(self.make_candidate())

        self.assertEqual(features["estimated_tier"], 5)
        self.assertGreaterEqual(features["feature_scores"]["career_evidence"], 0.75)
        self.assertGreater(features["feature_scores"]["semantic_alignment"], 0.0)
        self.assertEqual(features["semantic"]["backend"], "hashed_bi_encoder")
        self.assertIn("faiss", features["matches"]["vector_search"])
        self.assertTrue(features["evidence_snippets"])

    def test_nontechnical_title_limits_tier_when_career_evidence_is_weak(self):
        candidate = self.make_candidate(
            title="Marketing Manager",
            description="Owned campaign reporting, content operations, and brand analytics dashboards.",
        )
        candidate["profile"]["summary"] = "Marketing leader with AI interests."
        features = extract_candidate_features(candidate)

        self.assertLessEqual(features["estimated_tier"], 2)
        self.assertEqual(features["feature_scores"]["title_alignment"], 0.0)

    def test_extract_file_reuses_valid_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "risk.jsonl"
            input_path.write_text(json.dumps(self.make_candidate()) + "\n", encoding="utf-8")
            out_dir = root / "features"

            first = extract_file(input_path, out_dir)
            second = extract_file(input_path, out_dir)

            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(second["total_records"], 1)
            self.assertTrue((out_dir / "feature_cache_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
