"""Risk and trap detection for cleaned candidate records.

This stage annotates candidates with risk flags. It does not rank candidates.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.io import iter_candidate_records, write_json
from app.preprocess import input_fingerprint, normalize_label, normalize_text


RISK_VERSION = "risk_v5"
DEFAULT_REFERENCE_DATE = date(2026, 6, 29)

SEVERITY_POINTS = {
    "low": 5,
    "medium": 15,
    "high": 30,
    "severe": 50,
}

AI_SKILL_TERMS = {
    "ai",
    "artificial intelligence",
    "machine learning",
    "ml",
    "deep learning",
    "llm",
    "large language model",
    "rag",
    "nlp",
    "natural language processing",
    "embeddings",
    "sentence-transformers",
    "bge",
    "e5",
    "vector search",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "faiss",
    "elasticsearch",
    "opensearch",
    "ranking",
    "recommendation",
    "recommender",
    "learning to rank",
    "xgboost",
    "lora",
    "qlora",
    "peft",
    "fine-tuning",
    "fine tuning",
    "langchain",
    "openai",
}

RETRIEVAL_RANKING_TERMS = {
    "retrieval",
    "ranking",
    "ranker",
    "search",
    "recommendation",
    "recommender",
    "matching",
    "embeddings",
    "vector",
    "faiss",
    "milvus",
    "pinecone",
    "qdrant",
    "weaviate",
    "opensearch",
    "elasticsearch",
    "bm25",
    "hybrid search",
}

PRODUCTION_TERMS = {
    "production",
    "deployed",
    "shipped",
    "users",
    "scale",
    "latency",
    "monitoring",
    "a/b",
    "ab test",
    "offline",
    "online",
    "ndcg",
    "mrr",
    "map",
    "regression",
    "index refresh",
}

NONTECH_ROLE_TERMS = {
    "marketing",
    "sales",
    "hr",
    "human resources",
    "accountant",
    "finance",
    "graphic designer",
    "content writer",
    "customer support",
    "support executive",
    "operations manager",
}

CONSULTING_COMPANIES = {
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "tech mahindra",
    "mindtree",
    "ltimindtree",
    "genpact",
    "genpact ai",
}

CONSULTING_INDUSTRY_TERMS = {"it services", "consulting", "professional services", "ai services", "bpo", "outsourcing"}

CV_SPEECH_ROBOTICS_TERMS = {
    "computer vision",
    "image classification",
    "object detection",
    "opencv",
    "speech recognition",
    "tts",
    "text to speech",
    "robotics",
    "slam",
}

NLP_IR_TERMS = {"nlp", "natural language", "retrieval", "ranking", "search", "recommendation", "matching", "ir"}


@lru_cache(maxsize=512)
def term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term).replace("\\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def term_in_text(text: str, term: str) -> bool:
    return term_pattern(term).search(text) is not None


def contains_any(text: str, terms: set[str]) -> bool:
    return any(term_in_text(text, term) for term in terms)


def count_terms(text: str, terms: set[str]) -> int:
    return sum(1 for term in terms if term_in_text(text, term))


def parse_date(value: Any) -> date | None:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def skill_names(candidate: dict[str, Any]) -> list[str]:
    return [normalize_label(skill.get("name_normalized") or skill.get("name")) for skill in candidate.get("skills", [])]


def career_text(candidate: dict[str, Any]) -> str:
    parts = []
    profile = candidate.get("profile", {})
    parts.extend(
        [
            profile.get("headline", ""),
            profile.get("summary", ""),
            profile.get("current_title", ""),
            profile.get("current_industry", ""),
        ]
    )
    for job in candidate.get("career_history", []):
        parts.extend([job.get("title", ""), job.get("industry", ""), job.get("description", "")])
    return normalize_label(" ".join(parts))


def add_flag(flags: list[dict[str, Any]], name: str, severity: str, detail: str, evidence: dict[str, Any] | None = None) -> None:
    flags.append(
        {
            "name": name,
            "severity": severity,
            "points": SEVERITY_POINTS[severity],
            "detail": detail,
            "evidence": evidence or {},
        }
    )


def assess_candidate_risk(candidate: dict[str, Any], reference_date: date = DEFAULT_REFERENCE_DATE) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    cleaning_notes = set(candidate.get("_cleaning", {}).get("notes", []))
    skills = candidate.get("skills", [])
    jobs = candidate.get("career_history", [])
    text = career_text(candidate)
    names = skill_names(candidate)
    current_title = normalize_label(profile.get("current_title"))

    flag_cleaning_contradictions(flags, cleaning_notes)
    flag_experience_mismatch(flags, profile, jobs)
    flag_current_role_mismatch(flags, profile, jobs)
    flag_impossible_skill_claims(flags, profile, skills)
    flag_keyword_traps(flags, names, text, current_title)
    flag_jd_disqualifier_patterns(flags, candidate, names, text)
    flag_behavioral_risk(flags, signals, cleaning_notes, reference_date)

    risk_score = min(100, sum(flag["points"] for flag in flags))
    severity_counts = Counter(flag["severity"] for flag in flags)
    recommendation = recommendation_for(flags, risk_score)

    return {
        "risk_score": risk_score,
        "recommendation": recommendation,
        "flags": flags,
        "flag_count": len(flags),
        "severity_counts": dict(sorted(severity_counts.items())),
    }


def flag_cleaning_contradictions(flags: list[dict[str, Any]], cleaning_notes: set[str]) -> None:
    severe_notes = {
        "career_end_before_start",
        "invalid_career_start_date",
        "invalid_signup_date",
        "invalid_last_active_date",
    }
    high_notes = {"current_role_has_end_date"}
    for note in sorted(severe_notes & cleaning_notes):
        add_flag(flags, note, "severe", "Core date or career structure is contradictory.", {"cleaning_note": note})
    for note in sorted(high_notes & cleaning_notes):
        add_flag(flags, note, "high", "Current role metadata is internally inconsistent.", {"cleaning_note": note})
    if "last_active_before_signup" in cleaning_notes:
        add_flag(
            flags,
            "last_active_before_signup",
            "medium",
            "Activity date is earlier than signup date; this is suspicious but not enough to reject alone.",
        )


def flag_experience_mismatch(flags: list[dict[str, Any]], profile: dict[str, Any], jobs: list[dict[str, Any]]) -> None:
    profile_years = float(profile.get("years_of_experience") or 0.0)
    career_months = sum(max(0, int(job.get("duration_months") or 0)) for job in jobs)
    career_years = career_months / 12.0
    if career_years <= 0:
        add_flag(flags, "missing_career_duration", "high", "Career history has no usable duration.")
        return

    diff = abs(profile_years - career_years)
    evidence = {"profile_years": round(profile_years, 2), "career_years_from_roles": round(career_years, 2)}
    if profile_years >= 5 and career_years < 1:
        add_flag(flags, "severe_experience_mismatch", "severe", "Profile years conflict strongly with career durations.", evidence)
    elif diff >= 5:
        add_flag(flags, "large_experience_mismatch", "high", "Profile years differ materially from summed role durations.", evidence)
    elif diff >= 3:
        add_flag(flags, "moderate_experience_mismatch", "medium", "Profile years differ from summed role durations.", evidence)


def flag_current_role_mismatch(flags: list[dict[str, Any]], profile: dict[str, Any], jobs: list[dict[str, Any]]) -> None:
    current_jobs = [job for job in jobs if job.get("is_current")]
    if not current_jobs:
        add_flag(flags, "missing_current_role", "medium", "No current role appears in career history.")
        return
    current = current_jobs[0]
    profile_title = normalize_label(profile.get("current_title"))
    profile_company = normalize_label(profile.get("current_company"))
    job_title = normalize_label(current.get("title"))
    job_company = normalize_label(current.get("company"))
    title_match = profile_title and job_title and (profile_title in job_title or job_title in profile_title)
    company_match = profile_company and job_company and (profile_company in job_company or job_company in profile_company)
    if not title_match and not company_match:
        add_flag(
            flags,
            "current_profile_career_mismatch",
            "medium",
            "Profile current role does not match the current career-history role.",
            {
                "profile_title": profile.get("current_title"),
                "profile_company": profile.get("current_company"),
                "career_title": current.get("title"),
                "career_company": current.get("company"),
            },
        )


def flag_impossible_skill_claims(flags: list[dict[str, Any]], profile: dict[str, Any], skills: list[dict[str, Any]]) -> None:
    if not skills:
        return
    profile_months = int(float(profile.get("years_of_experience") or 0.0) * 12)
    zero_expert = [
        skill.get("name")
        for skill in skills
        if skill.get("proficiency") == "expert" and int(skill.get("duration_months") or 0) <= 1
    ]
    if len(zero_expert) >= 5 or (len(zero_expert) >= 3 and len(zero_expert) / len(skills) >= 0.35):
        add_flag(
            flags,
            "many_expert_zero_duration_skills",
            "high",
            "Several skills are marked expert with near-zero duration.",
            {"skills": zero_expert[:10], "count": len(zero_expert)},
        )

    overlong = [
        skill.get("name")
        for skill in skills
        if profile_months > 0 and int(skill.get("duration_months") or 0) > profile_months + 24
    ]
    if len(overlong) >= 3:
        add_flag(
            flags,
            "skill_duration_exceeds_experience",
            "medium",
            "Multiple skills have durations longer than the candidate's total stated experience.",
            {"skills": overlong[:10], "count": len(overlong)},
        )


def flag_keyword_traps(flags: list[dict[str, Any]], names: list[str], text: str, current_title: str) -> None:
    ai_skill_hits = [name for name in names if contains_any(name, AI_SKILL_TERMS)]
    retrieval_hits = count_terms(text, RETRIEVAL_RANKING_TERMS)
    production_hits = count_terms(text, PRODUCTION_TERMS)
    evidence_hits = retrieval_hits + production_hits

    if len(ai_skill_hits) >= 6 and evidence_hits <= 1:
        add_flag(
            flags,
            "ai_keyword_stuffing_without_career_evidence",
            "high",
            "AI-heavy skill list is not supported by production/retrieval/ranking career evidence.",
            {"ai_skill_count": len(ai_skill_hits), "career_evidence_hits": evidence_hits},
        )
    elif len(ai_skill_hits) >= 4 and evidence_hits == 0:
        add_flag(
            flags,
            "possible_ai_keyword_stuffing",
            "medium",
            "Several AI skills appear without supporting career evidence.",
            {"ai_skill_count": len(ai_skill_hits)},
        )

    if contains_any(current_title, NONTECH_ROLE_TERMS) and len(ai_skill_hits) >= 4 and evidence_hits <= 1:
        add_flag(
            flags,
            "nontechnical_title_with_ai_skill_stack",
            "high",
            "Current title is weakly aligned to the JD despite many AI skills.",
            {"current_title": current_title, "ai_skill_count": len(ai_skill_hits)},
        )


def flag_jd_disqualifier_patterns(
    flags: list[dict[str, Any]], candidate: dict[str, Any], names: list[str], text: str
) -> None:
    jobs = candidate.get("career_history", [])
    companies = [normalize_label(job.get("company")) for job in jobs]
    industries = [normalize_label(job.get("industry")) for job in jobs]
    company_industry_text = " ".join(companies + industries)

    if ("langchain" in text or "langchain" in names or "openai" in names) and count_terms(text, RETRIEVAL_RANKING_TERMS) == 0:
        add_flag(
            flags,
            "recent_framework_wrapper_without_retrieval_depth",
            "medium",
            "LLM/framework signal appears without retrieval, ranking, or production depth.",
        )

    if contains_any(text, {"research", "academic", "lab", "publication"}) and not contains_any(text, PRODUCTION_TERMS):
        add_flag(
            flags,
            "research_heavy_without_production_signal",
            "medium",
            "Research-heavy profile lacks explicit production deployment evidence.",
        )

    consulting_matches = [company for company in companies if contains_any(company, CONSULTING_COMPANIES)]
    consulting_industries = [industry for industry in industries if contains_any(industry, CONSULTING_INDUSTRY_TERMS)]
    service_context_count = 0
    for company, industry in zip(companies, industries):
        if contains_any(company, CONSULTING_COMPANIES) or contains_any(industry, CONSULTING_INDUSTRY_TERMS):
            service_context_count += 1
    current_jobs = [job for job in jobs if job.get("is_current")]
    current_job = current_jobs[0] if current_jobs else (jobs[0] if jobs else {})
    current_company = normalize_label(current_job.get("company"))
    current_industry = normalize_label(current_job.get("industry"))
    current_is_services = contains_any(current_company, CONSULTING_COMPANIES) or contains_any(
        current_industry, CONSULTING_INDUSTRY_TERMS
    )

    if jobs and service_context_count == len(jobs):
        add_flag(
            flags,
            "services_or_consulting_only_history",
            "high",
            "Career history appears entirely services/consulting-oriented.",
            {"matched_companies": consulting_matches[:5], "matched_industries": consulting_industries[:5]},
        )
    elif current_is_services:
        add_flag(
            flags,
            "current_services_or_consulting_context",
            "medium",
            "Current role is in a services/consulting context, which the JD treats as a fit concern unless prior product experience is strong.",
            {"current_company": current_job.get("company"), "current_industry": current_job.get("industry")},
        )

    cv_speech_hits = [name for name in names if contains_any(name, CV_SPEECH_ROBOTICS_TERMS)]
    nlp_ir_hits = count_terms(text + " " + " ".join(names), NLP_IR_TERMS)
    if len(cv_speech_hits) >= 3 and nlp_ir_hits == 0:
        add_flag(
            flags,
            "cv_speech_robotics_without_nlp_ir_overlap",
            "medium",
            "Primary AI signals appear outside the JD's NLP/IR/retrieval focus.",
            {"skills": cv_speech_hits[:10]},
        )


def flag_behavioral_risk(
    flags: list[dict[str, Any]],
    signals: dict[str, Any],
    cleaning_notes: set[str],
    reference_date: date,
) -> None:
    last_active = parse_date(signals.get("last_active_date"))
    if last_active and "last_active_before_signup" not in cleaning_notes:
        inactive_days = (reference_date - last_active).days
        if inactive_days >= 180:
            add_flag(
                flags,
                "long_inactive_candidate",
                "medium",
                "Candidate has not been active recently enough for a high-conversion hiring flow.",
                {"inactive_days": inactive_days},
            )
        elif inactive_days >= 90:
            add_flag(flags, "moderate_inactivity", "low", "Candidate has moderate recent inactivity.", {"inactive_days": inactive_days})

    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    if response_rate <= 0.05:
        add_flag(flags, "very_low_recruiter_response_rate", "medium", "Recruiter response rate is very low.", {"rate": response_rate})
    elif response_rate <= 0.15:
        add_flag(flags, "low_recruiter_response_rate", "low", "Recruiter response rate is low.", {"rate": response_rate})

    if not bool(signals.get("open_to_work_flag")):
        add_flag(flags, "not_open_to_work", "low", "Candidate is not marked open to work.")

    notice = int(signals.get("notice_period_days") or 0)
    if notice >= 120:
        add_flag(flags, "very_long_notice_period", "high", "Notice period is very long for this startup role.", {"notice_period_days": notice})
    elif notice > 90:
        add_flag(flags, "very_long_notice_period", "medium", "Notice period is long for this startup role.", {"notice_period_days": notice})
    elif notice > 60:
        add_flag(flags, "long_notice_period", "low", "Notice period is above the JD preference.", {"notice_period_days": notice})


def recommendation_for(flags: list[dict[str, Any]], risk_score: int) -> str:
    severe_count = sum(1 for flag in flags if flag["severity"] == "severe")
    high_count = sum(1 for flag in flags if flag["severity"] == "high")
    if severe_count >= 1 or risk_score >= 80:
        return "reject_or_near_zero"
    if high_count >= 2 or risk_score >= 50:
        return "heavy_penalty"
    if risk_score >= 25:
        return "moderate_penalty"
    if risk_score > 0:
        return "light_penalty"
    return "pass"


def risk_fingerprint(input_path: str | Path) -> dict[str, Any]:
    return {
        "stage": "risk",
        "stage_version": RISK_VERSION,
        **input_fingerprint(input_path),
    }


def is_cache_valid(input_path: str | Path, output_dir: str | Path) -> bool:
    output = Path(output_dir)
    required = [output / "candidates_with_risk.jsonl", output / "risk_report.json", output / "risk_cache_metadata.json"]
    if not all(path.exists() for path in required):
        return False
    try:
        with (output / "risk_cache_metadata.json").open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    expected = risk_fingerprint(input_path)
    return all(metadata.get(key) == value for key, value in expected.items())


def load_cached_report(output_dir: str | Path) -> dict[str, Any]:
    with (Path(output_dir) / "risk_report.json").open("r", encoding="utf-8") as f:
        report = json.load(f)
    report["cache_hit"] = True
    return report


def assess_file(input_path: str | Path, output_dir: str | Path, force: bool = False, reference_date: date = DEFAULT_REFERENCE_DATE) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    risk_path = output / "candidates_with_risk.jsonl"
    report_path = output / "risk_report.json"
    cache_path = output / "risk_cache_metadata.json"

    if not force and is_cache_valid(input_path, output):
        return load_cached_report(output)

    total = 0
    risk_score_sum = 0
    flag_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    recommendation_counts: Counter[str] = Counter()

    with risk_path.open("w", encoding="utf-8") as f:
        for total, candidate in enumerate(iter_candidate_records(input_path), start=1):
            risk = assess_candidate_risk(candidate, reference_date=reference_date)
            candidate["_risk"] = risk
            f.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            risk_score_sum += risk["risk_score"]
            recommendation_counts[risk["recommendation"]] += 1
            for flag in risk["flags"]:
                flag_counts[flag["name"]] += 1
                severity_counts[flag["severity"]] += 1

    report = {
        "input_path": str(input_path),
        "total_records": total,
        "risk_output": str(risk_path),
        "average_risk_score": round(risk_score_sum / total, 4) if total else 0.0,
        "flag_counts": dict(sorted(flag_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "recommendation_counts": dict(sorted(recommendation_counts.items())),
        "cache_hit": False,
        "stage_version": RISK_VERSION,
        "reference_date": reference_date.isoformat(),
    }
    write_json(report_path, report)
    write_json(
        cache_path,
        {
            **risk_fingerprint(input_path),
            "risk_output": str(risk_path),
            "report_output": str(report_path),
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        },
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate cleaned candidate records with risk flags.")
    parser.add_argument("--input", required=True, help="Path to cleaned candidates JSONL")
    parser.add_argument("--out-dir", default="data/processed/risk", help="Directory for risk artifacts")
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even when the cache is valid")
    parser.add_argument("--reference-date", default=DEFAULT_REFERENCE_DATE.isoformat(), help="Date used for activity recency checks")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    reference_date = date.fromisoformat(args.reference_date)
    report = assess_file(args.input, args.out_dir, force=args.force, reference_date=reference_date)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
