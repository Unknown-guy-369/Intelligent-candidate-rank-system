"""JD evidence feature extraction for risk-annotated candidates.

This stage produces compact feature records for scoring. It does not create
the final ranking CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from app.io import iter_candidate_records, write_json
from app.preprocess import input_fingerprint, normalize_label, normalize_text
from app.risk import DEFAULT_REFERENCE_DATE, contains_any, count_terms, parse_date, term_in_text
from app.semantic import DEFAULT_TRANSFORMER_MODEL, semantic_alignment_score, semantic_alignment_scores


FEATURE_VERSION = "features_v6"

PYTHON_TERMS = {"python", "pyspark", "fastapi", "flask", "django"}
EMBEDDING_TERMS = {"embedding", "embeddings", "sentence-transformers", "sentence transformers", "bge", "e5"}
VECTOR_SEARCH_TERMS = {"faiss", "milvus", "pinecone", "qdrant", "weaviate", "vector search", "vector database"}
SEARCH_RANKING_TERMS = {
    "retrieval",
    "ranking",
    "ranker",
    "recommendation",
    "recommender",
    "matching",
    "bm25",
    "hybrid search",
    "elasticsearch",
    "opensearch",
}
SEARCH_SYSTEM_TERMS = {
    "search system",
    "search infrastructure",
    "search ranking",
    "candidate search",
    "semantic search",
    "hybrid search",
}
CORE_RETRIEVAL_TERMS = SEARCH_RANKING_TERMS | SEARCH_SYSTEM_TERMS | VECTOR_SEARCH_TERMS | EMBEDDING_TERMS
EVALUATION_TERMS = {
    "ndcg",
    "mrr",
    "map",
    "a/b",
    "ab test",
    "offline evaluation",
    "online evaluation",
    "evaluation framework",
    "recruiter feedback loop",
    "user feedback loop",
    "retrieval-quality regression",
    "regression",
}
PRODUCTION_TERMS = {
    "production system",
    "production deployment",
    "production code",
    "deployed",
    "shipped",
    "real users",
    "users",
    "latency",
    "monitoring",
    "on-call",
    "index refresh",
    "drift",
}
LLM_FINETUNE_TERMS = {"fine-tuning", "fine tuning", "lora", "qlora", "peft", "llm", "rag"}
BACKEND_DATA_TERMS = {
    "backend",
    "data engineering",
    "data pipeline",
    "pipeline",
    "spark",
    "airflow",
    "kafka",
    "warehouse",
    "ml infrastructure",
}
PRODUCT_CONTEXT_TERMS = {
    "saas",
    "marketplace",
    "product",
    "platform",
    "hr-tech",
    "recruiting",
    "talent",
    "search",
    "recommendation",
}
STRONG_TITLE_TERMS = {
    "ml engineer",
    "machine learning engineer",
    "ai engineer",
    "search engineer",
    "ranking engineer",
    "recommendation systems engineer",
    "applied scientist",
    "data scientist",
}
ADJACENT_TITLE_TERMS = {"backend engineer", "data engineer", "analytics engineer", "software engineer", "cloud engineer"}
NONTECH_TITLE_TERMS = {
    "marketing",
    "sales",
    "hr",
    "accountant",
    "graphic designer",
    "content writer",
    "customer support",
    "operations manager",
}
PREFERRED_LOCATION_TERMS = {
    "pune",
    "noida",
    "delhi",
    "ncr",
    "gurugram",
    "gurgaon",
    "mumbai",
    "hyderabad",
    "bangalore",
    "bengaluru",
}


def progress_log(message: str, enabled: bool = True) -> None:
    if enabled:
        print(f"[features] {message}", file=sys.stderr, flush=True)


def matched_terms(text: str, terms: set[str]) -> list[str]:
    return sorted(term for term in terms if term_in_text(text, term))


def skill_lookup(candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        normalize_label(skill.get("name_normalized") or skill.get("name")): skill
        for skill in candidate.get("skills", [])
        if normalize_text(skill.get("name"))
    }


def combined_text(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
    ]
    for job in candidate.get("career_history", []):
        parts.extend([job.get("title", ""), job.get("company", ""), job.get("industry", ""), job.get("description", "")])
    return normalize_label(" ".join(parts))


def career_only_text(candidate: dict[str, Any]) -> str:
    parts = []
    for job in candidate.get("career_history", []):
        parts.extend([job.get("title", ""), job.get("industry", ""), job.get("description", "")])
    return normalize_label(" ".join(parts))


def transformer_prefilter_record(candidate: dict[str, Any]) -> tuple[bool, str]:
    profile = candidate.get("profile", {})
    risk = candidate.get("_risk", {})
    current_title = normalize_label(profile.get("current_title"))
    text = combined_text(candidate)
    career_text = career_only_text(candidate)
    skills = skill_lookup(candidate)
    skill_text = " ".join(skills)

    risk_score = int(risk.get("risk_score") or 0)
    recommendation = risk.get("recommendation", "")
    if recommendation == "reject_or_near_zero" or risk_score >= 70:
        return False, "high risk or rejected before semantic stage"

    title_strong = contains_any(current_title, STRONG_TITLE_TERMS)
    title_adjacent = contains_any(current_title, ADJACENT_TITLE_TERMS)
    title_nontech = contains_any(current_title, NONTECH_TITLE_TERMS)
    career_retrieval = contains_any(career_text, CORE_RETRIEVAL_TERMS | EVALUATION_TERMS)
    career_production = contains_any(career_text, PRODUCTION_TERMS)
    skill_support = contains_any(skill_text, PYTHON_TERMS | CORE_RETRIEVAL_TERMS | LLM_FINETUNE_TERMS)
    text_retrieval = contains_any(text, CORE_RETRIEVAL_TERMS | EVALUATION_TERMS)

    if career_retrieval and (career_production or title_strong or title_adjacent):
        return True, "career retrieval/search/ranking evidence"
    if title_strong and (career_retrieval or skill_support):
        return True, "strong ML/AI/search title with support"
    if title_adjacent and career_retrieval and skill_support:
        return True, "adjacent engineering title with retrieval support"
    if not title_nontech and text_retrieval and career_production:
        return True, "non-nontech profile with production semantic evidence"

    return False, "no cheap JD evidence for transformer"


def bounded_score(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return round(max(0.0, min(1.0, value / maximum)), 4)


def days_since(value: Any, reference_date: date) -> int | None:
    parsed = parse_date(value)
    if not parsed:
        return None
    return (reference_date - parsed).days


def extract_candidate_features(candidate: dict[str, Any], reference_date: date = DEFAULT_REFERENCE_DATE) -> dict[str, Any]:
    return extract_candidate_features_with_semantic(candidate, reference_date=reference_date)


def extract_candidate_features_with_semantic(
    candidate: dict[str, Any],
    reference_date: date = DEFAULT_REFERENCE_DATE,
    semantic_backend: str = "hashed",
    semantic_model: str = DEFAULT_TRANSFORMER_MODEL,
    semantic_local_files_only: bool = True,
    semantic_allow_fallback: bool = True,
    semantic_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    risk = candidate.get("_risk", {})
    skills = skill_lookup(candidate)
    text = combined_text(candidate)
    career_text = career_only_text(candidate)
    current_title = normalize_label(profile.get("current_title"))
    current_industry = normalize_label(profile.get("current_industry"))

    text_matches = {
        "python": matched_terms(text + " " + " ".join(skills), PYTHON_TERMS),
        "embeddings": matched_terms(text + " " + " ".join(skills), EMBEDDING_TERMS),
        "vector_search": matched_terms(text + " " + " ".join(skills), VECTOR_SEARCH_TERMS),
        "search_ranking": matched_terms(text + " " + " ".join(skills), SEARCH_RANKING_TERMS | SEARCH_SYSTEM_TERMS),
        "evaluation": matched_terms(text, EVALUATION_TERMS),
        "production": matched_terms(text, PRODUCTION_TERMS),
        "llm_finetune": matched_terms(text + " " + " ".join(skills), LLM_FINETUNE_TERMS),
        "backend_data": matched_terms(text, BACKEND_DATA_TERMS),
        "product_context": matched_terms(text + " " + current_industry, PRODUCT_CONTEXT_TERMS),
    }

    top_relevant_skills = relevant_skills(candidate, skills)
    evidence_snippets = extract_evidence_snippets(candidate)
    semantic = semantic_override
    if semantic is None:
        semantic = semantic_alignment_score(
            candidate,
            backend=semantic_backend,
            model_name=semantic_model,
            local_files_only=semantic_local_files_only,
            allow_fallback=semantic_allow_fallback,
        )
    years = float(profile.get("years_of_experience") or 0.0)
    inactive_days = days_since(signals.get("last_active_date"), reference_date)

    feature_scores = {
        "experience_band": experience_band_score(years),
        "title_alignment": title_alignment_score(current_title),
        "python": bounded_score(len(text_matches["python"]), 2),
        "embeddings": bounded_score(len(text_matches["embeddings"]), 2),
        "vector_search": bounded_score(len(text_matches["vector_search"]), 2),
        "search_ranking": bounded_score(len(text_matches["search_ranking"]), 4),
        "evaluation": bounded_score(len(text_matches["evaluation"]), 3),
        "production": bounded_score(len(text_matches["production"]), 4),
        "llm_finetune": bounded_score(len(text_matches["llm_finetune"]), 3),
        "backend_data": bounded_score(len(text_matches["backend_data"]), 4),
        "product_context": bounded_score(len(text_matches["product_context"]), 3),
        "career_evidence": career_evidence_score(text_matches, career_text),
        "semantic_alignment": float(semantic["score"]),
        "behavioral_availability": behavioral_availability_score(signals, inactive_days),
        "location_logistics": location_logistics_score(profile, signals),
    }

    tier = estimate_tier(feature_scores, risk, current_title, text_matches)

    return {
        "candidate_id": candidate.get("candidate_id"),
        "profile": {
            "current_title": profile.get("current_title", ""),
            "current_company": profile.get("current_company", ""),
            "current_industry": profile.get("current_industry", ""),
            "years_of_experience": years,
            "location": profile.get("location", ""),
            "country": profile.get("country", ""),
        },
        "signals": {
            "open_to_work": bool(signals.get("open_to_work_flag")),
            "recruiter_response_rate": float(signals.get("recruiter_response_rate") or 0.0),
            "notice_period_days": int(signals.get("notice_period_days") or 0),
            "preferred_work_mode": signals.get("preferred_work_mode", ""),
            "willing_to_relocate": bool(signals.get("willing_to_relocate")),
            "last_active_date": signals.get("last_active_date"),
            "inactive_days": inactive_days,
            "github_activity_score": float(signals.get("github_activity_score") or -1.0),
            "saved_by_recruiters_30d": int(signals.get("saved_by_recruiters_30d") or 0),
        },
        "matches": text_matches,
        "semantic": semantic,
        "top_relevant_skills": top_relevant_skills,
        "evidence_snippets": evidence_snippets,
        "feature_scores": feature_scores,
        "estimated_tier": tier,
        "risk": {
            "risk_score": int(risk.get("risk_score") or 0),
            "recommendation": risk.get("recommendation", "unknown"),
            "flag_names": [flag.get("name") for flag in risk.get("flags", [])],
            "severity_counts": risk.get("severity_counts", {}),
        },
    }


def relevant_skills(candidate: dict[str, Any], skills: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    all_terms = (
        PYTHON_TERMS
        | EMBEDDING_TERMS
        | VECTOR_SEARCH_TERMS
        | SEARCH_RANKING_TERMS
        | EVALUATION_TERMS
        | LLM_FINETUNE_TERMS
        | BACKEND_DATA_TERMS
    )
    selected = []
    for key, skill in skills.items():
        if any(term_in_text(key, term) for term in all_terms):
            selected.append(
                {
                    "name": skill.get("name"),
                    "proficiency": skill.get("proficiency"),
                    "proficiency_score": int(skill.get("proficiency_score") or 0),
                    "duration_months": int(skill.get("duration_months") or 0),
                    "endorsements": int(skill.get("endorsements") or 0),
                }
            )
    selected.sort(key=lambda row: (row["proficiency_score"], row["duration_months"], row["endorsements"]), reverse=True)
    return selected[:12]


def extract_evidence_snippets(candidate: dict[str, Any]) -> list[dict[str, str]]:
    evidence_terms = CORE_RETRIEVAL_TERMS | EVALUATION_TERMS | PRODUCTION_TERMS
    snippets = []
    for job in candidate.get("career_history", []):
        description = normalize_text(job.get("description"))
        if not description:
            continue
        lower_description = normalize_label(description)
        hits = matched_terms(lower_description, evidence_terms)
        if hits:
            snippets.append(
                {
                    "title": normalize_text(job.get("title")),
                    "company": normalize_text(job.get("company")),
                    "matched_terms": ", ".join(hits[:8]),
                    "snippet": description[:260],
                }
            )
        if len(snippets) >= 3:
            break
    return snippets


def experience_band_score(years: float) -> float:
    if 6 <= years <= 8:
        return 1.0
    if 5 <= years <= 9:
        return 0.9
    if 4 <= years < 5 or 9 < years <= 11:
        return 0.65
    if 3 <= years < 4 or 11 < years <= 14:
        return 0.35
    return 0.1


def title_alignment_score(title: str) -> float:
    if contains_any(title, STRONG_TITLE_TERMS):
        return 1.0
    if contains_any(title, ADJACENT_TITLE_TERMS):
        return 0.55
    if contains_any(title, NONTECH_TITLE_TERMS):
        return 0.0
    return 0.25


def career_evidence_score(matches: dict[str, list[str]], career_text: str) -> float:
    retrieval = count_terms(career_text, CORE_RETRIEVAL_TERMS)
    production = count_terms(career_text, PRODUCTION_TERMS)
    evaluation = count_terms(career_text, EVALUATION_TERMS)
    value = min(1.0, retrieval * 0.18 + production * 0.12 + evaluation * 0.2)
    if matches["search_ranking"] and matches["production"]:
        value += 0.15
    if matches["evaluation"]:
        value += 0.15
    return round(min(1.0, value), 4)


def behavioral_availability_score(signals: dict[str, Any], inactive_days: int | None) -> float:
    score = 0.0
    if bool(signals.get("open_to_work_flag")):
        score += 0.25
    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    score += min(0.3, response_rate * 0.3)
    notice = int(signals.get("notice_period_days") or 0)
    if notice <= 30:
        score += 0.25
    elif notice <= 60:
        score += 0.15
    elif notice <= 90:
        score += 0.05
    if inactive_days is None:
        score += 0.05
    elif inactive_days <= 30:
        score += 0.2
    elif inactive_days <= 90:
        score += 0.12
    elif inactive_days <= 180:
        score += 0.04
    return round(min(1.0, score), 4)


def location_logistics_score(profile: dict[str, Any], signals: dict[str, Any]) -> float:
    location = normalize_label(" ".join([profile.get("location", ""), profile.get("country", "")]))
    if contains_any(location, {"pune", "noida"}):
        return 1.0
    if contains_any(location, PREFERRED_LOCATION_TERMS):
        return 0.8
    if normalize_label(profile.get("country")) == "india" and bool(signals.get("willing_to_relocate")):
        return 0.7
    if bool(signals.get("willing_to_relocate")):
        return 0.45
    return 0.2


def estimate_tier(
    scores: dict[str, float],
    risk: dict[str, Any],
    current_title: str,
    matches: dict[str, list[str]],
) -> int:
    recommendation = risk.get("recommendation", "")
    risk_score = int(risk.get("risk_score") or 0)
    if recommendation == "reject_or_near_zero":
        return 0
    if contains_any(current_title, NONTECH_TITLE_TERMS) and scores["career_evidence"] < 0.35:
        return 1
    if risk_score >= 70:
        return 1
    if (
        scores["experience_band"] >= 0.9
        and scores["career_evidence"] >= 0.75
        and scores["title_alignment"] >= 0.55
        and scores["python"] >= 0.5
        and (scores["vector_search"] >= 0.5 or scores["embeddings"] >= 0.5)
        and scores["evaluation"] >= 0.3
        and risk_score < 40
    ):
        return 5
    if (
        scores["career_evidence"] >= 0.55
        and scores["search_ranking"] >= 0.5
        and scores["title_alignment"] >= 0.55
        and risk_score < 55
    ):
        return 4
    if (
        scores["backend_data"] >= 0.35
        or (scores["python"] >= 0.5 and scores["title_alignment"] >= 0.55)
        or (matches["llm_finetune"] and scores["title_alignment"] >= 0.55)
    ):
        return 3 if risk_score < 60 else 2
    if matches["search_ranking"] or matches["embeddings"] or matches["vector_search"]:
        return 2
    return 1


def feature_fingerprint(
    input_path: str | Path,
    semantic_backend: str = "hashed",
    semantic_model: str = DEFAULT_TRANSFORMER_MODEL,
    semantic_local_files_only: bool = True,
    semantic_batch_size: int = 32,
    semantic_prefilter: bool = True,
) -> dict[str, Any]:
    return {
        "stage": "features",
        "stage_version": FEATURE_VERSION,
        "semantic_backend": semantic_backend,
        "semantic_model": semantic_model if semantic_backend == "transformer" else None,
        "semantic_local_files_only": semantic_local_files_only,
        "semantic_batch_size": semantic_batch_size,
        "semantic_prefilter": semantic_prefilter,
        **input_fingerprint(input_path),
    }


def is_cache_valid(
    input_path: str | Path,
    output_dir: str | Path,
    semantic_backend: str = "hashed",
    semantic_model: str = DEFAULT_TRANSFORMER_MODEL,
    semantic_local_files_only: bool = True,
    semantic_batch_size: int = 32,
    semantic_prefilter: bool = True,
) -> bool:
    output = Path(output_dir)
    required = [output / "candidate_features.jsonl", output / "feature_report.json", output / "feature_cache_metadata.json"]
    if not all(path.exists() for path in required):
        return False
    try:
        with (output / "feature_cache_metadata.json").open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    expected = feature_fingerprint(
        input_path,
        semantic_backend=semantic_backend,
        semantic_model=semantic_model,
        semantic_local_files_only=semantic_local_files_only,
        semantic_batch_size=semantic_batch_size,
        semantic_prefilter=semantic_prefilter,
    )
    return all(metadata.get(key) == value for key, value in expected.items())


def load_cached_report(output_dir: str | Path) -> dict[str, Any]:
    with (Path(output_dir) / "feature_report.json").open("r", encoding="utf-8") as f:
        report = json.load(f)
    report["cache_hit"] = True
    return report


def extract_file(
    input_path: str | Path,
    output_dir: str | Path,
    force: bool = False,
    reference_date: date = DEFAULT_REFERENCE_DATE,
    semantic_backend: str = "hashed",
    semantic_model: str = DEFAULT_TRANSFORMER_MODEL,
    semantic_local_files_only: bool = True,
    semantic_allow_fallback: bool = True,
    semantic_batch_size: int = 32,
    semantic_prefilter: bool = True,
    progress_every: int = 5000,
    progress: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features_path = output / "candidate_features.jsonl"
    report_path = output / "feature_report.json"
    cache_path = output / "feature_cache_metadata.json"

    started_at = time.monotonic()
    progress_log(
        (
            f"start input={input_path} out_dir={output_dir} semantic={semantic_backend} "
            f"model={semantic_model if semantic_backend == 'transformer' else '-'} "
            f"batch_size={semantic_batch_size} prefilter={semantic_prefilter}"
        ),
        enabled=progress,
    )

    if not force and is_cache_valid(
        input_path,
        output,
        semantic_backend=semantic_backend,
        semantic_model=semantic_model,
        semantic_local_files_only=semantic_local_files_only,
        semantic_batch_size=semantic_batch_size,
        semantic_prefilter=semantic_prefilter,
    ):
        progress_log("cache hit; reusing existing feature artifacts", enabled=progress)
        return load_cached_report(output)

    total = 0
    tier_counts: Counter[str] = Counter()
    title_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    score_sums: Counter[str] = Counter()
    semantic_backend_counts: Counter[str] = Counter()
    semantic_prefilter_counts: Counter[str] = Counter()

    def write_feature_batch(batch: list[dict[str, Any]], output_file) -> None:
        semantics: list[dict[str, Any] | None] = [None] * len(batch)
        transformer_indexes: list[int] = []

        if semantic_backend == "transformer" and semantic_prefilter:
            transformer_batch = []
            for index, candidate in enumerate(batch):
                allowed, reason = transformer_prefilter_record(candidate)
                if allowed:
                    transformer_indexes.append(index)
                    transformer_batch.append(candidate)
                    semantic_prefilter_counts["encoded"] += 1
                else:
                    semantic_prefilter_counts["skipped"] += 1
                    semantics[index] = {
                        "backend": "transformer_skipped_by_prefilter",
                        "model_name": semantic_model,
                        "stage_version": "semantic_v2",
                        "prefilter_reason": reason,
                        "score": 0.0,
                    }
            encoded_semantics = semantic_alignment_scores(
                transformer_batch,
                backend=semantic_backend,
                model_name=semantic_model,
                local_files_only=semantic_local_files_only,
                allow_fallback=semantic_allow_fallback,
                batch_size=semantic_batch_size,
            )
            for index, semantic in zip(transformer_indexes, encoded_semantics):
                semantics[index] = semantic
            if transformer_batch:
                progress_log(
                    (
                        f"encoded transformer batch size={len(transformer_batch)} "
                        f"skipped_in_batch={len(batch) - len(transformer_batch)}"
                    ),
                    enabled=progress,
                )
        else:
            semantic_prefilter_counts["encoded"] += len(batch)
            semantics = semantic_alignment_scores(
                batch,
                backend=semantic_backend,
                model_name=semantic_model,
                local_files_only=semantic_local_files_only,
                allow_fallback=semantic_allow_fallback,
                batch_size=semantic_batch_size,
            )

        for candidate, semantic in zip(batch, semantics):
            assert semantic is not None
            features = extract_candidate_features_with_semantic(
                candidate,
                reference_date=reference_date,
                semantic_backend=semantic_backend,
                semantic_model=semantic_model,
                semantic_local_files_only=semantic_local_files_only,
                semantic_allow_fallback=semantic_allow_fallback,
                semantic_override=semantic,
            )
            output_file.write(json.dumps(features, ensure_ascii=False, sort_keys=True))
            output_file.write("\n")
            tier_counts[str(features["estimated_tier"])] += 1
            title_counts[features["profile"]["current_title"]] += 1
            risk_counts[features["risk"]["recommendation"]] += 1
            semantic_backend_counts[semantic.get("backend", "unknown")] += 1
            for key, value in features["feature_scores"].items():
                score_sums[key] += float(value)

    with features_path.open("w", encoding="utf-8") as f:
        batch: list[dict[str, Any]] = []
        for total, candidate in enumerate(iter_candidate_records(input_path), start=1):
            batch.append(candidate)
            if len(batch) >= semantic_batch_size:
                write_feature_batch(batch, f)
                batch = []
            if progress_every > 0 and total % progress_every == 0:
                elapsed = max(0.001, time.monotonic() - started_at)
                progress_log(
                    (
                        f"processed={total} elapsed={elapsed:.1f}s rate={total / elapsed:.1f}/s "
                        f"encoded={semantic_prefilter_counts.get('encoded', 0)} "
                        f"skipped={semantic_prefilter_counts.get('skipped', 0)}"
                    ),
                    enabled=progress,
                )
        if batch:
            write_feature_batch(batch, f)
            if progress_every > 0:
                elapsed = max(0.001, time.monotonic() - started_at)
                progress_log(
                    (
                        f"processed={total} elapsed={elapsed:.1f}s rate={total / elapsed:.1f}/s "
                        f"encoded={semantic_prefilter_counts.get('encoded', 0)} "
                        f"skipped={semantic_prefilter_counts.get('skipped', 0)}"
                    ),
                    enabled=progress,
                )

    average_scores = {key: round(value / total, 4) for key, value in score_sums.items()} if total else {}
    report = {
        "input_path": str(input_path),
        "total_records": total,
        "features_output": str(features_path),
        "tier_counts": dict(sorted(tier_counts.items())),
        "risk_recommendation_counts": dict(sorted(risk_counts.items())),
        "top_current_titles": dict(title_counts.most_common(20)),
        "average_feature_scores": average_scores,
        "cache_hit": False,
        "stage_version": FEATURE_VERSION,
        "reference_date": reference_date.isoformat(),
        "semantic_backend": semantic_backend,
        "semantic_effective_backend_counts": dict(sorted(semantic_backend_counts.items())),
        "semantic_model": semantic_model if semantic_backend == "transformer" else None,
        "semantic_local_files_only": semantic_local_files_only,
        "semantic_batch_size": semantic_batch_size,
        "semantic_prefilter": semantic_prefilter,
        "semantic_prefilter_counts": dict(sorted(semantic_prefilter_counts.items())),
    }
    write_json(report_path, report)
    write_json(
        cache_path,
        {
            **feature_fingerprint(
                input_path,
                semantic_backend=semantic_backend,
                semantic_model=semantic_model,
                semantic_local_files_only=semantic_local_files_only,
                semantic_batch_size=semantic_batch_size,
                semantic_prefilter=semantic_prefilter,
            ),
            "features_output": str(features_path),
            "report_output": str(report_path),
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "semantic_backend": semantic_backend,
            "semantic_model": semantic_model if semantic_backend == "transformer" else None,
            "semantic_local_files_only": semantic_local_files_only,
            "semantic_batch_size": semantic_batch_size,
            "semantic_prefilter": semantic_prefilter,
        },
    )
    elapsed = max(0.001, time.monotonic() - started_at)
    progress_log(
        (
            f"done total={total} elapsed={elapsed:.1f}s rate={total / elapsed:.1f}/s "
            f"semantic_backends={dict(sorted(semantic_backend_counts.items()))}"
        ),
        enabled=progress,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract JD evidence features from risk-annotated candidates.")
    parser.add_argument("--input", required=True, help="Path to candidates_with_risk.jsonl")
    parser.add_argument("--out-dir", default="data/processed/features", help="Directory for feature artifacts")
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even when the cache is valid")
    parser.add_argument("--reference-date", default=DEFAULT_REFERENCE_DATE.isoformat(), help="Date used for activity recency checks")
    parser.add_argument("--semantic-backend", choices=["hashed", "transformer"], default="hashed", help="Semantic backend")
    parser.add_argument("--semantic-model", default=DEFAULT_TRANSFORMER_MODEL, help="Local transformer model path or HF model id")
    parser.add_argument("--semantic-allow-download", action="store_true", help="Allow transformer loading to use network/cache misses")
    parser.add_argument("--semantic-no-fallback", action="store_true", help="Fail instead of falling back when transformer backend is unavailable")
    parser.add_argument("--semantic-batch-size", type=int, default=32, help="Semantic encoder batch size")
    parser.add_argument("--semantic-no-prefilter", action="store_true", help="Encode every candidate with transformer instead of only plausible JD matches")
    parser.add_argument("--progress-every", type=int, default=5000, help="Print progress every N records; 0 disables periodic logs")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logs")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    reference_date = date.fromisoformat(args.reference_date)
    report = extract_file(
        args.input,
        args.out_dir,
        force=args.force,
        reference_date=reference_date,
        semantic_backend=args.semantic_backend,
        semantic_model=args.semantic_model,
        semantic_local_files_only=not args.semantic_allow_download,
        semantic_allow_fallback=not args.semantic_no_fallback,
        semantic_batch_size=args.semantic_batch_size,
        semantic_prefilter=not args.semantic_no_prefilter,
        progress_every=args.progress_every,
        progress=not args.quiet,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
