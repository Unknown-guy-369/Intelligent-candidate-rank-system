import unittest

from app.semantic import semantic_alignment_score


def make_candidate(title, summary, description, skills=None):
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "headline": title,
            "summary": summary,
            "current_title": title,
            "current_company": "ProductCo",
            "current_industry": "SaaS",
        },
        "career_history": [
            {
                "title": title,
                "company": "ProductCo",
                "industry": "SaaS",
                "description": description,
            }
        ],
        "skills": [{"name": skill, "duration_months": 24} for skill in (skills or [])],
    }


class SemanticTests(unittest.TestCase):
    def test_retrieval_career_text_scores_above_generic_hr_profile(self):
        strong = make_candidate(
            "Senior Machine Learning Engineer",
            "Builds production candidate retrieval and job matching systems.",
            "Shipped hybrid search, ranking evaluation, embeddings, and feedback-loop improvements for real users.",
            ["Python", "FAISS"],
        )
        weak = make_candidate(
            "HR Manager",
            "Experienced HR generalist interested in AI hiring tools.",
            "Managed onboarding, employee engagement, payroll coordination, and recruitment operations.",
            ["AI", "ChatGPT", "Recruitment"],
        )

        self.assertGreater(
            semantic_alignment_score(strong)["score"],
            semantic_alignment_score(weak)["score"],
        )

    def test_semantic_score_is_bounded(self):
        score = semantic_alignment_score(make_candidate("Engineer", "", ""))["score"]

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_transformer_backend_falls_back_when_model_is_unavailable(self):
        result = semantic_alignment_score(
            make_candidate("Engineer", "", ""),
            backend="transformer",
            model_name="/definitely/not/a/local/model",
        )

        self.assertEqual(result["backend"], "hashed_bi_encoder_fallback")
        self.assertEqual(result["requested_backend"], "transformer")

    def test_transformer_backend_can_fail_loudly(self):
        with self.assertRaises(RuntimeError):
            semantic_alignment_score(
                make_candidate("Engineer", "", ""),
                backend="transformer",
                model_name="/definitely/not/a/local/model",
                allow_fallback=False,
            )


if __name__ == "__main__":
    unittest.main()
