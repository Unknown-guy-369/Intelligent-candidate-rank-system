import copy
import json
import tempfile
import unittest
from pathlib import Path

from app.preprocess import clean_candidate, normalize_text, preprocess_file


class PreprocessTests(unittest.TestCase):
    def make_candidate(self):
        return {
            "candidate_id": "CAND_0000001",
            "profile": {
                "anonymized_name": " Test User ",
                "headline": " Senior ML Engineer ",
                "summary": " Built search   systems. ",
                "location": " Pune ",
                "country": " India ",
                "years_of_experience": 7.2,
                "current_title": " Senior ML Engineer ",
                "current_company": " ProductCo ",
                "current_company_size": "201-500",
                "current_industry": " SaaS ",
            },
            "career_history": [
                {
                    "company": "OldCo",
                    "title": "ML Engineer",
                    "start_date": "2019-01-01",
                    "end_date": "2021-01-01",
                    "duration_months": 24,
                    "is_current": False,
                    "industry": "SaaS",
                    "company_size": "51-200",
                    "description": "Built ranking models.",
                },
                {
                    "company": "ProductCo",
                    "title": "Senior ML Engineer",
                    "start_date": "2021-02-01",
                    "end_date": None,
                    "duration_months": 40,
                    "is_current": True,
                    "industry": "SaaS",
                    "company_size": "201-500",
                    "description": "Owns retrieval systems.",
                },
            ],
            "education": [
                {
                    "institution": "College",
                    "degree": "B.Tech",
                    "field_of_study": "Computer Science",
                    "start_year": 2013,
                    "end_year": 2017,
                    "grade": None,
                    "tier": "tier_2",
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "advanced", "endorsements": 10, "duration_months": 48},
                {"name": " python ", "proficiency": "expert", "endorsements": 3, "duration_months": 36},
                {"name": "FAISS", "proficiency": "intermediate", "endorsements": 4, "duration_months": 18},
            ],
            "certifications": [],
            "languages": [{"language": "English", "proficiency": "professional"}],
            "redrob_signals": {
                "profile_completeness_score": 91,
                "signup_date": "2025-01-01",
                "last_active_date": "2026-06-01",
                "open_to_work_flag": True,
                "profile_views_received_30d": 8,
                "applications_submitted_30d": 2,
                "recruiter_response_rate": 0.8,
                "avg_response_time_hours": 4.5,
                "skill_assessment_scores": {"Python": 88},
                "connection_count": 100,
                "endorsements_received": 22,
                "notice_period_days": 30,
                "expected_salary_range_inr_lpa": {"min": 30, "max": 45},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": 61,
                "search_appearance_30d": 12,
                "saved_by_recruiters_30d": 3,
                "interview_completion_rate": 0.9,
                "offer_acceptance_rate": 0.7,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True,
            },
        }

    def test_normalize_text_collapses_whitespace(self):
        self.assertEqual(normalize_text("  Built   retrieval\nsystems  "), "Built retrieval systems")

    def test_clean_candidate_merges_duplicate_skills(self):
        cleaned, notes = clean_candidate(self.make_candidate())
        self.assertIsNotNone(cleaned)
        self.assertIn("duplicate_skill_merged", notes)
        python_skill = next(skill for skill in cleaned["skills"] if skill["name_normalized"] == "python")
        self.assertEqual(python_skill["proficiency"], "expert")
        self.assertEqual(python_skill["endorsements"], 13)
        self.assertEqual(python_skill["duration_months"], 48)

    def test_clean_candidate_sorts_current_career_first(self):
        cleaned, _ = clean_candidate(self.make_candidate())
        self.assertEqual(cleaned["career_history"][0]["company"], "ProductCo")
        self.assertTrue(cleaned["career_history"][0]["is_current"])

    def test_invalid_candidate_id_rejected(self):
        candidate = copy.deepcopy(self.make_candidate())
        candidate["candidate_id"] = "bad-id"
        cleaned, notes = clean_candidate(candidate)
        self.assertIsNone(cleaned)
        self.assertEqual(notes, ["invalid_candidate_id"])

    def test_preprocess_file_reuses_valid_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "candidates.jsonl"
            input_path.write_text(json.dumps(self.make_candidate()) + "\n", encoding="utf-8")
            out_dir = root / "processed"

            first = preprocess_file(input_path, out_dir)
            second = preprocess_file(input_path, out_dir)

            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(second["cleaned_records"], 1)
            self.assertTrue((out_dir / "cache_metadata.json").exists())

    def test_preprocess_file_force_rebuilds_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "candidates.jsonl"
            input_path.write_text(json.dumps(self.make_candidate()) + "\n", encoding="utf-8")
            out_dir = root / "processed"

            preprocess_file(input_path, out_dir)
            forced = preprocess_file(input_path, out_dir, force=True)

            self.assertFalse(forced["cache_hit"])
            self.assertEqual(forced["cleaned_records"], 1)


if __name__ == "__main__":
    unittest.main()
