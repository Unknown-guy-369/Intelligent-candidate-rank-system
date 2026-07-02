"""Final scoring and CSV ranking for candidate features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from app.io import iter_candidate_records, write_json, write_jsonl


SCORING_VERSION = "scoring_v3"

TIER_PRIOR = {
    5: 0.95,
    4: 0.78,
    3: 0.5,
    2: 0.3,
    1: 0.08,
    0: 0.0,
}

WEIGHTS = {
    "tier_prior": 0.22,
    "career_evidence": 0.18,
    "semantic_alignment": 0.08,
    "search_ranking": 0.09,
    "vector_search": 0.07,
    "embeddings": 0.06,
    "evaluation": 0.1,
    "python": 0.08,
    "production": 0.07,
    "title_alignment": 0.06,
    "behavioral_availability": 0.06,
    "experience_band": 0.05,
    "product_context": 0.04,
    "location_logistics": 0.03,
    "backend_data": 0.03,
    "llm_finetune": 0.02,
}

RISK_RECOMMENDATION_PENALTY = {
    "pass": 0.0,
    "light_penalty": 0.04,
    "moderate_penalty": 0.12,
    "heavy_penalty": 0.24,
    "reject_or_near_zero": 1.0,
    "unknown": 0.08,
}


def weighted_average(values: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    return sum(values.get(key, 0.0) * weight for key, weight in weights.items()) / total_weight


def score_candidate(features: dict[str, Any]) -> dict[str, Any]:
    feature_scores = features.get("feature_scores", {})
    tier = int(features.get("estimated_tier") or 0)
    risk = features.get("risk", {})
    recommendation = risk.get("recommendation", "unknown")
    risk_score = int(risk.get("risk_score") or 0)

    if recommendation == "reject_or_near_zero" or tier == 0:
        final_score = 0.0
    else:
        values = {
            **{key: float(value) for key, value in feature_scores.items()},
            "tier_prior": TIER_PRIOR.get(tier, 0.0),
        }
        evidence_score = weighted_average(values, WEIGHTS)
        risk_penalty = RISK_RECOMMENDATION_PENALTY.get(recommendation, 0.08)
        risk_penalty += min(0.18, risk_score / 100.0 * 0.18)
        business_penalty, score_cap = business_penalty_and_cap(features)
        final_score = max(0.0, min(score_cap, evidence_score - risk_penalty - business_penalty))

    return {
        "candidate_id": features["candidate_id"],
        "score": round(final_score, 6),
        "score_components": {
            "tier": tier,
            "tier_prior": TIER_PRIOR.get(tier, 0.0),
            "risk_score": risk_score,
            "risk_recommendation": recommendation,
            "feature_scores": feature_scores,
            "business_penalty": business_penalty_and_cap(features)[0],
            "score_cap": business_penalty_and_cap(features)[1],
        },
    }


def business_penalty_and_cap(features: dict[str, Any]) -> tuple[float, float]:
    signals = features.get("signals", {})
    profile = features.get("profile", {})
    risk = features.get("risk", {})
    flag_names = set(risk.get("flag_names", []))
    penalty = 0.0
    cap = 1.0

    notice = int(signals.get("notice_period_days") or 0)
    if notice >= 120:
        penalty += 0.18
        cap = min(cap, 0.7)
    elif notice > 90:
        penalty += 0.14
        cap = min(cap, 0.74)
    elif notice > 60:
        penalty += 0.07

    if "services_or_consulting_only_history" in flag_names:
        penalty += 0.18
        cap = min(cap, 0.62)
    elif "current_services_or_consulting_context" in flag_names:
        penalty += 0.08

    years = float(profile.get("years_of_experience") or 0.0)
    if years > 14:
        penalty += 0.08
        cap = min(cap, 0.68)
    if "large_experience_mismatch" in flag_names:
        penalty += 0.08

    return round(penalty, 6), cap


def first_sentence(text: str, max_chars: int = 180) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    stop_positions = [idx for idx in [cleaned.find(". "), cleaned.find("; ")] if idx != -1]
    if stop_positions:
        cleaned = cleaned[: min(stop_positions) + 1]
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def evidence_label(features: dict[str, Any]) -> str:
    scores = features.get("feature_scores", {})
    labels = []
    if scores.get("career_evidence", 0) >= 0.75:
        labels.append("strong production retrieval/ranking evidence")
    elif scores.get("career_evidence", 0) >= 0.45:
        labels.append("solid search or recommendation evidence")
    if scores.get("evaluation", 0) >= 0.3:
        labels.append("ranking evaluation exposure")
    if scores.get("vector_search", 0) >= 0.5 or scores.get("embeddings", 0) >= 0.5:
        labels.append("embedding/vector-search depth")
    if scores.get("python", 0) >= 0.5:
        labels.append("Python strength")
    return ", ".join(labels[:3]) or "relevant applied-ML evidence"


def build_reasoning(features: dict[str, Any], score_record: dict[str, Any]) -> str:
    profile = features.get("profile", {})
    signals = features.get("signals", {})
    risk = features.get("risk", {})
    snippets = features.get("evidence_snippets", [])
    skills = features.get("top_relevant_skills", [])

    title = profile.get("current_title") or "Candidate"
    years = profile.get("years_of_experience")
    years_text = f"{years:.1f} yrs" if isinstance(years, (int, float)) else "experience"

    evidence_parts = []
    if snippets:
        first = snippets[0]
        company = first.get("company") or "recent role"
        snippet = first_sentence(first.get("snippet", ""))
        if snippet:
            evidence_parts.append(f"{title} with {years_text}; at {company}, {snippet}")
        else:
            evidence_parts.append(f"{title} with {years_text}; {company} role shows {evidence_label(features)}.")
    else:
        evidence_parts.append(f"{title} with {years_text}; profile shows {evidence_label(features)}.")

    skill_names = [skill.get("name") for skill in skills[:2] if skill.get("name")]
    if skill_names and not snippets:
        evidence_parts.append(f"Relevant supporting skills include {', '.join(skill_names)}.")
    elif skill_names:
        evidence_parts.append(f"Supporting skills: {', '.join(skill_names)}.")

    concerns = []
    recommendation = risk.get("recommendation", "pass")
    if recommendation in {"moderate_penalty", "heavy_penalty"}:
        flag_names = risk.get("flag_names", [])[:2]
        if flag_names:
            concerns.append("concerns: " + ", ".join(flag_names).replace("_", " "))
    notice = int(signals.get("notice_period_days") or 0)
    if notice > 60:
        concerns.append(f"{notice}-day notice")
    response = float(signals.get("recruiter_response_rate") or 0.0)
    if response < 0.2:
        concerns.append(f"response rate {response:.2f}")
    if concerns:
        evidence_parts.append("Tradeoff: " + "; ".join(concerns) + ".")

    reasoning = " ".join(evidence_parts)
    return reasoning[:700]


def rank_features(input_path: str | Path, output_csv: str | Path, audit_path: str | Path | None = None, top_n: int = 100) -> dict[str, Any]:
    scored = []
    total = 0
    skipped_near_zero = 0

    for total, features in enumerate(iter_candidate_records(input_path), start=1):
        score_record = score_candidate(features)
        if score_record["score"] <= 0:
            skipped_near_zero += 1
        scored.append({**score_record, "features": features})

    scored.sort(key=lambda row: (-row["score"], row["candidate_id"]))
    selected = scored[:top_n]

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, row in enumerate(selected, start=1):
            writer.writerow(
                [
                    row["candidate_id"],
                    rank,
                    f"{row['score']:.6f}",
                    build_reasoning(row["features"], row),
                ]
            )

    audit_records = []
    for rank, row in enumerate(selected, start=1):
        features = row["features"]
        audit_records.append(
            {
                "candidate_id": row["candidate_id"],
                "rank": rank,
                "score": row["score"],
                "estimated_tier": features.get("estimated_tier"),
                "profile": features.get("profile", {}),
                "risk": features.get("risk", {}),
                "score_components": row["score_components"],
                "reasoning": build_reasoning(features, row),
            }
        )
    if audit_path is not None:
        write_jsonl(audit_path, audit_records)

    report = {
        "input_path": str(input_path),
        "output_csv": str(output_csv),
        "audit_output": str(audit_path) if audit_path else None,
        "total_records": total,
        "top_n": top_n,
        "selected_records": len(selected),
        "zero_score_records": skipped_near_zero,
        "stage_version": SCORING_VERSION,
        "top_score": selected[0]["score"] if selected else 0.0,
        "bottom_selected_score": selected[-1]["score"] if selected else 0.0,
    }
    report_path = output.parent / "ranking_report.json"
    write_json(report_path, report)
    report["report_output"] = str(report_path)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank candidate feature records and write submission CSV.")
    parser.add_argument("--features", required=True, help="Path to candidate_features.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--audit-out", help="Optional JSONL audit path for selected candidates")
    parser.add_argument("--top-n", type=int, default=100, help="Number of candidates to output")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = rank_features(args.features, args.out, audit_path=args.audit_out, top_n=args.top_n)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
