import copy
import json
import tempfile
import unittest
from pathlib import Path

from app.preprocess import clean_candidate
from app.risk import assess_candidate_risk, assess_file, term_in_text


class RiskTests(unittest.TestCase):
    def make_candidate(self):
        raw = {
            "candidate_id": "CAND_0000001",
            "profile": {
                "anonymized_name": "Test User",
                "headline": "Senior ML Engineer",
                "summary": "Built production retrieval and ranking systems for real users.",
                "location": "Pune",
                "country": "India",
                "years_of_experience": 7.0,
                "current_title": "Senior ML Engineer",
                "current_company": "ProductCo",
                "current_company_size": "201-500",
                "current_industry": "SaaS",
            },
            "career_history": [
                {
                    "company": "ProductCo",
                    "title": "Senior ML Engineer",
                    "start_date": "2020-01-01",
                    "end_date": None,
                    "duration_months": 72,
                    "is_current": True,
                    "industry": "SaaS",
                    "company_size": "201-500",
                    "description": "Shipped production vector search, ranking evaluation, NDCG tracking, and retrieval systems.",
                }
            ],
            "education": [],
            "skills": [
                {"name": "Python", "proficiency": "expert", "endorsements": 15, "duration_months": 72},
                {"name": "FAISS", "proficiency": "advanced", "endorsements": 8, "duration_months": 36},
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
        return cleaned

    def test_strong_profile_has_no_reject_recommendation(self):
        risk = assess_candidate_risk(self.make_candidate())
        self.assertIn(risk["recommendation"], {"pass", "light_penalty"})
        self.assertFalse(any(flag["severity"] == "severe" for flag in risk["flags"]))

    def test_term_matching_does_not_match_inside_words(self):
        self.assertFalse(term_in_text("tailwind", "ai"))
        self.assertFalse(term_in_text("collaboration", "lab"))
        self.assertTrue(term_in_text("built ai retrieval systems", "ai"))

    def test_nontechnical_ai_keyword_stack_is_high_risk(self):
        candidate = self.make_candidate()
        candidate["profile"]["current_title"] = "Marketing Manager"
        candidate["profile"]["headline"] = "Marketing Manager"
        candidate["profile"]["summary"] = "Owned campaign reporting and brand content."
        candidate["career_history"][0]["title"] = "Marketing Manager"
        candidate["career_history"][0]["description"] = "Owned campaign reporting and brand content."
        candidate["skills"] = [
            {"name": "RAG", "name_normalized": "rag", "proficiency": "advanced", "proficiency_score": 3, "endorsements": 10, "duration_months": 12},
            {"name": "Pinecone", "name_normalized": "pinecone", "proficiency": "advanced", "proficiency_score": 3, "endorsements": 10, "duration_months": 12},
            {"name": "LLM", "name_normalized": "llm", "proficiency": "advanced", "proficiency_score": 3, "endorsements": 10, "duration_months": 12},
            {"name": "LangChain", "name_normalized": "langchain", "proficiency": "advanced", "proficiency_score": 3, "endorsements": 10, "duration_months": 12},
        ]

        risk = assess_candidate_risk(candidate)
        flag_names = {flag["name"] for flag in risk["flags"]}

        self.assertIn("nontechnical_title_with_ai_skill_stack", flag_names)
        self.assertIn(risk["recommendation"], {"heavy_penalty", "moderate_penalty", "reject_or_near_zero"})

    def test_many_zero_duration_expert_skills_are_flagged(self):
        candidate = self.make_candidate()
        candidate["skills"] = [
            {
                "name": f"Skill {idx}",
                "name_normalized": f"skill {idx}",
                "proficiency": "expert",
                "proficiency_score": 4,
                "endorsements": 1,
                "duration_months": 0,
            }
            for idx in range(5)
        ]

        risk = assess_candidate_risk(candidate)
        self.assertIn("many_expert_zero_duration_skills", {flag["name"] for flag in risk["flags"]})

    def test_genpact_current_context_is_flagged_but_not_services_only(self):
        candidate = self.make_candidate()
        candidate["profile"]["current_company"] = "Genpact AI"
        candidate["profile"]["current_industry"] = "AI Services"
        candidate["career_history"].insert(
            0,
            {
                "company": "Genpact AI",
                "title": "Senior ML Engineer",
                "start_date": "2024-01-01",
                "end_date": None,
                "duration_months": 24,
                "is_current": True,
                "industry": "AI Services",
                "company_size": "10001+",
                "description": "Built retrieval systems.",
            },
        )
        candidate["career_history"][1]["is_current"] = False

        risk = assess_candidate_risk(candidate)
        flag_names = {flag["name"] for flag in risk["flags"]}

        self.assertIn("current_services_or_consulting_context", flag_names)
        self.assertNotIn("services_or_consulting_only_history", flag_names)

    def test_services_only_history_is_high_risk(self):
        candidate = self.make_candidate()
        candidate["profile"]["current_company"] = "Genpact AI"
        candidate["profile"]["current_industry"] = "AI Services"
        candidate["career_history"][0]["company"] = "Genpact AI"
        candidate["career_history"][0]["industry"] = "AI Services"

        risk = assess_candidate_risk(candidate)
        flags = {flag["name"]: flag for flag in risk["flags"]}

        self.assertEqual(flags["services_or_consulting_only_history"]["severity"], "high")

    def test_assess_file_reuses_valid_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "cleaned.jsonl"
            input_path.write_text(json.dumps(self.make_candidate()) + "\n", encoding="utf-8")
            out_dir = root / "risk"

            first = assess_file(input_path, out_dir)
            second = assess_file(input_path, out_dir)

            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(second["total_records"], 1)
            self.assertTrue((out_dir / "risk_cache_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
