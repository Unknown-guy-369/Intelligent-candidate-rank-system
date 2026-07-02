"""Small-sample sandbox demo for the Redrob ranking pipeline.

Run locally:
  python sandbox_app.py

This app is intended for hosted sandbox/demo checks. It accepts a small
candidate JSON/JSONL upload, runs the offline pipeline with the fast hashed
semantic backend, and returns a ranked CSV.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr

from app.features import extract_file
from app.preprocess import preprocess_file
from app.risk import DEFAULT_REFERENCE_DATE, assess_file
from app.scoring import rank_features


def run_sandbox(candidate_file: str, top_n: int) -> tuple[str, list[list[Any]], str]:
    if not candidate_file:
        raise gr.Error("Upload a candidate JSON, JSONL, or JSONL.GZ file.")

    top_n = max(1, min(100, int(top_n)))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_path = Path(candidate_file)
        clean_dir = root / "clean"
        risk_dir = root / "risk"
        features_dir = root / "features"
        output_csv = root / "submission.csv"
        audit_path = root / "audit.jsonl"

        preprocess_report = preprocess_file(input_path, clean_dir, force=True)
        risk_report = assess_file(
            clean_dir / "candidates_clean.jsonl",
            risk_dir,
            force=True,
            reference_date=DEFAULT_REFERENCE_DATE,
        )
        feature_report = extract_file(
            risk_dir / "candidates_with_risk.jsonl",
            features_dir,
            force=True,
            reference_date=DEFAULT_REFERENCE_DATE,
            semantic_backend="hashed",
            progress=False,
        )
        ranking_report = rank_features(
            features_dir / "candidate_features.jsonl",
            output_csv,
            audit_path=audit_path,
            top_n=top_n,
        )

        preview = []
        with output_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                preview.append([row["rank"], row["candidate_id"], row["score"], row["reasoning"]])

        summary = (
            f"Processed {preprocess_report['total_records']} records; "
            f"cleaned {preprocess_report['cleaned_records']}; "
            f"risk pass {risk_report['recommendation_counts'].get('pass', 0)}; "
            f"ranked {ranking_report['selected_records']} candidates; "
            f"semantic backend {feature_report['semantic_backend']}."
        )

        durable_output = Path(tempfile.gettempdir()) / "redrob_sandbox_submission.csv"
        durable_output.write_text(output_csv.read_text(encoding="utf-8"), encoding="utf-8")
        return str(durable_output), preview, summary


with gr.Blocks(title="Redrob Candidate Ranker Sandbox") as demo:
    gr.Markdown("# Redrob Candidate Ranker Sandbox")
    gr.Markdown("Upload a small candidate JSON/JSONL file and generate a ranked CSV using the offline pipeline.")
    with gr.Row():
        candidate_file = gr.File(label="Candidate file", file_types=[".json", ".jsonl", ".gz"], type="filepath")
        top_n = gr.Slider(1, 100, value=10, step=1, label="Top N")
    run_button = gr.Button("Rank Candidates", variant="primary")
    summary = gr.Textbox(label="Run summary", lines=3)
    preview = gr.Dataframe(headers=["rank", "candidate_id", "score", "reasoning"], label="Preview")
    output_file = gr.File(label="Download CSV")

    run_button.click(run_sandbox, inputs=[candidate_file, top_n], outputs=[output_file, preview, summary])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
