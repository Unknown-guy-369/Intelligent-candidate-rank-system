"""Data cleaning for candidate records.

This module performs structural cleaning only. It does not assign JD fit,
trap risk, or ranking scores.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from app.io import iter_candidate_records, write_json, write_jsonl


PREPROCESS_VERSION = "preprocess_v1"
CANDIDATE_ID_RE = re.compile(r"^CAND_[0-9]{7}$")
WHITESPACE_RE = re.compile(r"\s+")

PROFICIENCY_SCORE = {
    "beginner": 1,
    "intermediate": 2,
    "advanced": 3,
    "expert": 4,
}

VALID_COMPANY_SIZES = {
    "1-10",
    "11-50",
    "51-200",
    "201-500",
    "501-1000",
    "1001-5000",
    "5001-10000",
    "10001+",
}

VALID_WORK_MODES = {"remote", "hybrid", "onsite", "flexible"}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return WHITESPACE_RE.sub(" ", text)


def normalize_label(value: Any) -> str:
    return normalize_text(value).lower()


def parse_iso_date(value: Any) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def to_float(value: Any, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def to_int(value: Any, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    number = int(round(to_float(value, float(default), minimum, maximum)))
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def clean_candidate(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """Return a cleaned candidate and cleaning notes.

    A candidate is rejected only when core structure is unusable.
    """
    notes: list[str] = []
    candidate_id = normalize_text(raw.get("candidate_id"))
    if not CANDIDATE_ID_RE.match(candidate_id):
        return None, ["invalid_candidate_id"]

    required_objects = ["profile", "redrob_signals"]
    for field in required_objects:
        if not isinstance(raw.get(field), dict):
            return None, [f"missing_or_invalid_{field}"]

    for field in ["career_history", "education", "skills"]:
        if not isinstance(raw.get(field), list):
            return None, [f"missing_or_invalid_{field}"]

    cleaned = copy.deepcopy(raw)
    cleaned["candidate_id"] = candidate_id
    cleaned["profile"] = clean_profile(raw["profile"], notes)
    cleaned["career_history"] = clean_career_history(raw["career_history"], notes)
    if not cleaned["career_history"]:
        return None, notes + ["empty_career_history_after_cleaning"]
    cleaned["education"] = clean_education(raw.get("education", []), notes)
    cleaned["skills"] = clean_skills(raw.get("skills", []), notes)
    cleaned["certifications"] = clean_certifications(raw.get("certifications", []), notes)
    cleaned["languages"] = clean_languages(raw.get("languages", []), notes)
    cleaned["redrob_signals"] = clean_redrob_signals(raw["redrob_signals"], notes)
    cleaned["_cleaning"] = {
        "notes": sorted(set(notes)),
        "note_count": len(set(notes)),
        "cleaned_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    return cleaned, notes


def clean_profile(profile: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    cleaned = {
        "anonymized_name": normalize_text(profile.get("anonymized_name")),
        "headline": normalize_text(profile.get("headline")),
        "summary": normalize_text(profile.get("summary")),
        "location": normalize_text(profile.get("location")),
        "country": normalize_text(profile.get("country")),
        "years_of_experience": round(to_float(profile.get("years_of_experience"), 0.0, 0.0, 50.0), 2),
        "current_title": normalize_text(profile.get("current_title")),
        "current_company": normalize_text(profile.get("current_company")),
        "current_company_size": normalize_company_size(profile.get("current_company_size"), notes),
        "current_industry": normalize_text(profile.get("current_industry")),
    }
    for key in ["headline", "summary", "current_title", "current_company"]:
        if not cleaned[key]:
            notes.append(f"empty_profile_{key}")
    return cleaned


def normalize_company_size(value: Any, notes: list[str]) -> str:
    text = normalize_text(value)
    if text in VALID_COMPANY_SIZES:
        return text
    notes.append("invalid_company_size")
    return "unknown"


def clean_career_history(items: list[Any], notes: list[str]) -> list[dict[str, Any]]:
    cleaned_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            notes.append("invalid_career_item")
            continue
        start_date = parse_iso_date(item.get("start_date"))
        end_date = parse_iso_date(item.get("end_date")) if item.get("end_date") is not None else None
        is_current = bool(item.get("is_current"))
        if not start_date:
            notes.append("invalid_career_start_date")
        if item.get("end_date") is not None and not end_date:
            notes.append("invalid_career_end_date")
        if start_date and end_date and end_date < start_date:
            notes.append("career_end_before_start")
        if is_current and end_date is not None:
            notes.append("current_role_has_end_date")

        cleaned_items.append(
            {
                "company": normalize_text(item.get("company")),
                "title": normalize_text(item.get("title")),
                "start_date": start_date,
                "end_date": end_date,
                "duration_months": to_int(item.get("duration_months"), 0, 0),
                "is_current": is_current,
                "industry": normalize_text(item.get("industry")),
                "company_size": normalize_company_size(item.get("company_size"), notes),
                "description": normalize_text(item.get("description")),
                "_source_index": index,
            }
        )

    cleaned_items.sort(
        key=lambda row: (
            1 if row["is_current"] else 0,
            row["start_date"] or "",
            row["duration_months"],
        ),
        reverse=True,
    )
    for new_index, item in enumerate(cleaned_items):
        item["_career_order"] = new_index
    return cleaned_items


def clean_education(items: list[Any], notes: list[str]) -> list[dict[str, Any]]:
    cleaned_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            notes.append("invalid_education_item")
            continue
        start_year = to_int(item.get("start_year"), 0)
        end_year = to_int(item.get("end_year"), 0)
        if start_year and end_year and end_year < start_year:
            notes.append("education_end_before_start")
        cleaned_items.append(
            {
                "institution": normalize_text(item.get("institution")),
                "degree": normalize_text(item.get("degree")),
                "field_of_study": normalize_text(item.get("field_of_study")),
                "start_year": start_year,
                "end_year": end_year,
                "grade": normalize_text(item.get("grade")) or None,
                "tier": normalize_label(item.get("tier")) or "unknown",
            }
        )
    cleaned_items.sort(key=lambda row: row["end_year"], reverse=True)
    return cleaned_items


def clean_skills(items: list[Any], notes: list[str]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            notes.append("invalid_skill_item")
            continue
        name = normalize_text(item.get("name"))
        key = normalize_label(name)
        if not key:
            notes.append("empty_skill_name")
            continue
        proficiency = normalize_label(item.get("proficiency"))
        if proficiency not in PROFICIENCY_SCORE:
            notes.append("invalid_skill_proficiency")
            proficiency = "beginner"
        skill = {
            "name": name,
            "name_normalized": key,
            "proficiency": proficiency,
            "proficiency_score": PROFICIENCY_SCORE[proficiency],
            "endorsements": to_int(item.get("endorsements"), 0, 0),
            "duration_months": to_int(item.get("duration_months"), 0, 0),
        }
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = skill
            continue
        notes.append("duplicate_skill_merged")
        if skill["proficiency_score"] > existing["proficiency_score"]:
            existing["proficiency"] = skill["proficiency"]
            existing["proficiency_score"] = skill["proficiency_score"]
        existing["endorsements"] += skill["endorsements"]
        existing["duration_months"] = max(existing["duration_months"], skill["duration_months"])

    return sorted(
        by_key.values(),
        key=lambda row: (row["proficiency_score"], row["duration_months"], row["endorsements"], row["name_normalized"]),
        reverse=True,
    )


def clean_certifications(items: Any, notes: list[str]) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        notes.append("invalid_certifications")
        return []
    cleaned_items = []
    for item in items:
        if not isinstance(item, dict):
            notes.append("invalid_certification_item")
            continue
        cleaned_items.append(
            {
                "name": normalize_text(item.get("name")),
                "issuer": normalize_text(item.get("issuer")),
                "year": to_int(item.get("year"), 0),
            }
        )
    return sorted(cleaned_items, key=lambda row: row["year"], reverse=True)


def clean_languages(items: Any, notes: list[str]) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        notes.append("invalid_languages")
        return []
    cleaned_items = []
    for item in items:
        if not isinstance(item, dict):
            notes.append("invalid_language_item")
            continue
        cleaned_items.append(
            {
                "language": normalize_text(item.get("language")),
                "proficiency": normalize_label(item.get("proficiency")),
            }
        )
    return cleaned_items


def clean_redrob_signals(signals: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    expected_salary = signals.get("expected_salary_range_inr_lpa")
    if not isinstance(expected_salary, dict):
        notes.append("invalid_expected_salary")
        expected_salary = {}

    assessments = signals.get("skill_assessment_scores")
    if not isinstance(assessments, dict):
        notes.append("invalid_skill_assessments")
        assessments = {}

    signup_date = parse_iso_date(signals.get("signup_date"))
    last_active_date = parse_iso_date(signals.get("last_active_date"))
    if not signup_date:
        notes.append("invalid_signup_date")
    if not last_active_date:
        notes.append("invalid_last_active_date")
    if signup_date and last_active_date and last_active_date < signup_date:
        notes.append("last_active_before_signup")

    work_mode = normalize_label(signals.get("preferred_work_mode"))
    if work_mode not in VALID_WORK_MODES:
        notes.append("invalid_work_mode")
        work_mode = "flexible"

    return {
        "profile_completeness_score": round(to_float(signals.get("profile_completeness_score"), 0.0, 0.0, 100.0), 2),
        "signup_date": signup_date,
        "last_active_date": last_active_date,
        "open_to_work_flag": bool(signals.get("open_to_work_flag")),
        "profile_views_received_30d": to_int(signals.get("profile_views_received_30d"), 0, 0),
        "applications_submitted_30d": to_int(signals.get("applications_submitted_30d"), 0, 0),
        "recruiter_response_rate": round(to_float(signals.get("recruiter_response_rate"), 0.0, 0.0, 1.0), 4),
        "avg_response_time_hours": round(to_float(signals.get("avg_response_time_hours"), 0.0, 0.0), 2),
        "skill_assessment_scores": {
            normalize_text(key): round(to_float(value, 0.0, 0.0, 100.0), 2)
            for key, value in assessments.items()
            if normalize_text(key)
        },
        "connection_count": to_int(signals.get("connection_count"), 0, 0),
        "endorsements_received": to_int(signals.get("endorsements_received"), 0, 0),
        "notice_period_days": to_int(signals.get("notice_period_days"), 0, 0, 180),
        "expected_salary_range_inr_lpa": {
            "min": round(to_float(expected_salary.get("min"), 0.0, 0.0), 2),
            "max": round(to_float(expected_salary.get("max"), 0.0, 0.0), 2),
        },
        "preferred_work_mode": work_mode,
        "willing_to_relocate": bool(signals.get("willing_to_relocate")),
        "github_activity_score": round(to_float(signals.get("github_activity_score"), -1.0, -1.0, 100.0), 2),
        "search_appearance_30d": to_int(signals.get("search_appearance_30d"), 0, 0),
        "saved_by_recruiters_30d": to_int(signals.get("saved_by_recruiters_30d"), 0, 0),
        "interview_completion_rate": round(to_float(signals.get("interview_completion_rate"), 0.0, 0.0, 1.0), 4),
        "offer_acceptance_rate": round(to_float(signals.get("offer_acceptance_rate"), -1.0, -1.0, 1.0), 4),
        "verified_email": bool(signals.get("verified_email")),
        "verified_phone": bool(signals.get("verified_phone")),
        "linkedin_connected": bool(signals.get("linkedin_connected")),
    }


def input_fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    return {
        "input_path": str(source),
        "input_size_bytes": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
    }


def is_cache_valid(input_path: str | Path, output_dir: str | Path) -> bool:
    output = Path(output_dir)
    cleaned_path = output / "candidates_clean.jsonl"
    rejected_path = output / "rejected_candidates.jsonl"
    report_path = output / "preprocessing_report.json"
    cache_path = output / "cache_metadata.json"
    if not all(path.exists() for path in [cleaned_path, rejected_path, report_path, cache_path]):
        return False

    try:
        with cache_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    expected = {
        "stage": "preprocess",
        "stage_version": PREPROCESS_VERSION,
        **input_fingerprint(input_path),
    }
    return all(metadata.get(key) == value for key, value in expected.items())


def load_cached_report(output_dir: str | Path) -> dict[str, Any]:
    report_path = Path(output_dir) / "preprocessing_report.json"
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    report["cache_hit"] = True
    return report


def preprocess_file(input_path: str | Path, output_dir: str | Path, force: bool = False) -> dict[str, Any]:
    output = Path(output_dir)
    cleaned_path = output / "candidates_clean.jsonl"
    rejected_path = output / "rejected_candidates.jsonl"
    report_path = output / "preprocessing_report.json"
    cache_path = output / "cache_metadata.json"

    if not force and is_cache_valid(input_path, output):
        return load_cached_report(output)

    cleaned_records = []
    rejected_records = []
    note_counts: Counter[str] = Counter()
    total = 0

    for total, raw in enumerate(iter_candidate_records(input_path), start=1):
        cleaned, notes = clean_candidate(raw)
        note_counts.update(notes)
        if cleaned is None:
            rejected_records.append(
                {
                    "candidate_id": normalize_text(raw.get("candidate_id")) if isinstance(raw, dict) else "",
                    "reasons": sorted(set(notes)),
                }
            )
        else:
            cleaned_records.append(cleaned)

    write_jsonl(cleaned_path, cleaned_records)
    write_jsonl(rejected_path, rejected_records)

    report = {
        "input_path": str(input_path),
        "total_records": total,
        "cleaned_records": len(cleaned_records),
        "rejected_records": len(rejected_records),
        "cleaned_output": str(cleaned_path),
        "rejected_output": str(rejected_path),
        "note_counts": dict(sorted(note_counts.items())),
        "cache_hit": False,
        "stage_version": PREPROCESS_VERSION,
    }
    write_json(report_path, report)

    cache_metadata = {
        "stage": "preprocess",
        "stage_version": PREPROCESS_VERSION,
        **input_fingerprint(input_path),
        "cleaned_output": str(cleaned_path),
        "rejected_output": str(rejected_path),
        "report_output": str(report_path),
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    write_json(cache_path, cache_metadata)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean and normalize Redrob candidate data.")
    parser.add_argument("--input", required=True, help="Path to candidates JSON, JSONL, or JSONL.GZ")
    parser.add_argument("--out-dir", default="data/processed", help="Directory for cleaned artifacts")
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even when the cache is valid")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = preprocess_file(args.input, args.out_dir, force=args.force)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
