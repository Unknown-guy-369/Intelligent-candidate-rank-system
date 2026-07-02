#!/usr/bin/env python3
"""Submission reproduction entrypoint.

Two supported modes:

1. Fast ranking from precomputed feature records:
   python rank.py --features data/processed/full_features_transformer/candidate_features.jsonl --out submission.csv

2. Full local pipeline from candidates:
   python rank.py --candidates data/candidates.jsonl --out submission.csv --work-dir data/processed/repro
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.features import extract_file
from app.preprocess import preprocess_file
from app.risk import DEFAULT_REFERENCE_DATE, assess_file
from app.scoring import rank_features
from app.semantic import DEFAULT_TRANSFORMER_MODEL


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Redrob top-100 submission CSV.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--features", help="Precomputed candidate_features.jsonl path for fast ranking")
    source.add_argument("--candidates", help="Raw candidates JSON/JSONL/JSONL.GZ path for full local pipeline")
    parser.add_argument("--out", required=True, help="Output submission CSV path")
    parser.add_argument("--audit-out", help="Optional top-100 audit JSONL path")
    parser.add_argument("--top-n", type=int, default=100, help="Number of ranked candidates to output")
    parser.add_argument("--work-dir", default="data/processed/repro", help="Work directory for full-pipeline mode")
    parser.add_argument("--force", action="store_true", help="Rebuild cached artifacts in full-pipeline mode")

    parser.add_argument("--semantic-backend", choices=["hashed", "transformer"], default="hashed", help="Semantic backend for full-pipeline mode")
    parser.add_argument("--semantic-model", default=DEFAULT_TRANSFORMER_MODEL, help="Transformer model id or local path")
    parser.add_argument("--semantic-batch-size", type=int, default=64, help="Semantic encoder batch size")
    parser.add_argument("--semantic-allow-download", action="store_true", help="Allow transformer model download/cache miss in full-pipeline mode")
    parser.add_argument("--semantic-no-fallback", action="store_true", help="Fail instead of falling back when transformer is unavailable")
    parser.add_argument("--semantic-no-prefilter", action="store_true", help="Encode every candidate with transformer instead of plausible JD matches only")
    parser.add_argument("--progress-every", type=int, default=5000, help="Feature progress log interval in full-pipeline mode")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logs in full-pipeline mode")
    return parser


def run_from_candidates(args: argparse.Namespace) -> dict:
    work_dir = Path(args.work_dir)
    clean_dir = work_dir / "clean"
    risk_dir = work_dir / "risk"
    feature_dir = work_dir / "features"
    audit_out = args.audit_out or str(Path(args.out).with_suffix(".audit.jsonl"))

    preprocess_report = preprocess_file(args.candidates, clean_dir, force=args.force)
    risk_report = assess_file(
        clean_dir / "candidates_clean.jsonl",
        risk_dir,
        force=args.force,
        reference_date=DEFAULT_REFERENCE_DATE,
    )
    feature_report = extract_file(
        risk_dir / "candidates_with_risk.jsonl",
        feature_dir,
        force=args.force,
        reference_date=DEFAULT_REFERENCE_DATE,
        semantic_backend=args.semantic_backend,
        semantic_model=args.semantic_model,
        semantic_local_files_only=not args.semantic_allow_download,
        semantic_allow_fallback=not args.semantic_no_fallback,
        semantic_batch_size=args.semantic_batch_size,
        semantic_prefilter=not args.semantic_no_prefilter,
        progress_every=args.progress_every,
        progress=not args.quiet,
    )
    ranking_report = rank_features(
        feature_dir / "candidate_features.jsonl",
        args.out,
        audit_path=audit_out,
        top_n=args.top_n,
    )
    return {
        "mode": "full_pipeline",
        "preprocess": preprocess_report,
        "risk": risk_report,
        "features": feature_report,
        "ranking": ranking_report,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.features:
        audit_out = args.audit_out or str(Path(args.out).with_suffix(".audit.jsonl"))
        report = {
            "mode": "features_only",
            "ranking": rank_features(args.features, args.out, audit_path=audit_out, top_n=args.top_n),
        }
    else:
        report = run_from_candidates(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
