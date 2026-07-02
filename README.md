---
title: Redrob Candidate Ranker Sandbox
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Redrob Intelligent Candidate Discovery & Ranking

Offline hybrid candidate ranking system for the Redrob Intelligent Candidate Discovery & Ranking Hackathon.

The system ranks the top 100 candidates for the Senior AI Engineer - Founding Team JD using full profile evidence, not skill-keyword counts. It combines deterministic cleaning/risk checks, JD evidence extraction, optional local transformer semantic scoring, business/logistics penalties, and grounded reasoning.

## Final Submission File

The current submit-ready CSV is:

```bash
data/processed/full_rank_transformer/submission.csv
```

Validate it with:

```bash
python3 validator/validate_submission.py data/processed/full_rank_transformer/submission.csv
```

Expected result:

```text
Submission is valid.
```

Before uploading, copy it to the registered participant/team filename required by the portal:

```bash
cp data/processed/full_rank_transformer/submission.csv YOUR_TEAM_ID.csv
python3 validator/validate_submission.py YOUR_TEAM_ID.csv
```

## Why This Is Not Keyword Matching

The hackathon sample submission ranks HR Managers and other non-technical profiles highly because they list many AI skills. This project treats that as a trap.

The ranker checks:

- career-history prose, not only `skills[]`
- production search/ranking/retrieval/recommendation evidence
- evaluation evidence such as A/B testing, NDCG, MRR, and offline-online relevance work
- title and career consistency
- product-company/startup fit
- availability signals such as notice period, activity, response rate, and relocation
- honeypot and contradiction risk
- local transformer semantic fit only after a cheap JD-evidence prefilter

## Architecture

```text
candidates.jsonl
  -> preprocessing / normalization
  -> risk and honeypot detection
  -> JD evidence feature extraction
  -> optional local transformer semantic scoring
  -> hybrid scoring and business penalties
  -> top-100 CSV with grounded reasoning
```

Main modules:

```text
app/io.py          streaming JSON/JSONL/GZ input helpers
app/preprocess.py cleaning and normalization
app/risk.py       honeypot, trap, contradiction, and business-risk flags
app/features.py   JD evidence and semantic feature extraction
app/semantic.py   hashed fallback + optional local transformer backend
app/scoring.py    final score, ranking, reasoning, audit output
rank.py           one-command reproduction entrypoint
```

## Trap And Honeypot Handling

The pipeline does not blindly delete unusual candidates. It rejects only severe contradictions and penalizes uncertain risk.

Examples of detected risks:

- impossible career dates or current-role metadata
- profile years inconsistent with summed career history
- expert AI skills with near-zero duration
- dense AI skill stuffing with no career evidence
- non-technical current role with AI buzzwords
- CV/speech/robotics-only profile without NLP/IR/retrieval overlap
- services/consulting-only career history
- inactive or low-response candidates
- very long notice period

Severe candidates are scored near zero. Medium-risk candidates remain eligible but receive penalties and score caps.

## Semantic Scoring

The default code path is dependency-free and uses a hashed bi-encoder fallback. The stronger path uses a real local transformer model:

```text
BAAI/bge-small-en-v1.5
```

Transformer scoring is protected by a prefilter:

- candidates with cheap JD evidence are transformer-encoded
- candidates with no cheap search/ranking/retrieval/product-ML evidence get no semantic boost

Observed full run:

```text
total candidates: 100000
sentence-transformer encoded: 2235
skipped by semantic prefilter: 97765
feature extraction elapsed: 394.9s
```

This transformer pass is an offline feature-precomputation step. The final scoring/ranking step from feature records is fast and CPU-only.

## Reproduce Final Transformer Submission

Install optional transformer dependencies in a Python environment:

```bash
pip install -r requirements-transformer.txt
```

One-time model download/cache setup:

```bash
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')"
```

Run feature extraction with local cached model files:

```bash
python3 -m app.features \
  --input data/processed/full_risk/candidates_with_risk.jsonl \
  --out-dir data/processed/full_features_transformer \
  --semantic-backend transformer \
  --semantic-model BAAI/bge-small-en-v1.5 \
  --semantic-batch-size 64 \
  --semantic-no-fallback \
  --progress-every 1000 \
  --force
```

Run final ranking:

```bash
python3 -m app.scoring \
  --features data/processed/full_features_transformer/candidate_features.jsonl \
  --out data/processed/full_rank_transformer/submission.csv \
  --audit-out data/processed/full_rank_transformer/audit.jsonl \
  --top-n 100
```

Validate:

```bash
python3 validator/validate_submission.py data/processed/full_rank_transformer/submission.csv
```

One-command reproduction from precomputed feature records:

```bash
python3 rank.py \
  --features data/processed/full_features_transformer/candidate_features.jsonl \
  --out data/processed/full_rank_transformer/submission.csv
```

One-command full local pipeline from raw candidates:

```bash
python3 rank.py \
  --candidates data/candidates.jsonl \
  --out data/processed/repro/submission.csv \
  --work-dir data/processed/repro \
  --semantic-backend transformer \
  --semantic-model BAAI/bge-small-en-v1.5 \
  --semantic-batch-size 64 \
  --semantic-no-fallback \
  --force
```

The full local pipeline includes transformer feature precomputation and may exceed the five-minute ranking budget on CPU. The fast Stage 3 ranking step is the `--features` command over precomputed feature records.

## Fast CPU-Only Fallback

If transformer dependencies or local model files are unavailable, use the standard-library hashed semantic backend:

```bash
python3 -m app.features \
  --input data/processed/full_risk/candidates_with_risk.jsonl \
  --out-dir data/processed/full_features \
  --semantic-backend hashed \
  --force
```

Then:

```bash
python3 -m app.scoring \
  --features data/processed/full_features/candidate_features.jsonl \
  --out data/processed/full_rank/submission.csv \
  --audit-out data/processed/full_rank/audit.jsonl \
  --top-n 100
```

## Full Pipeline Commands

From raw candidates:

```bash
python3 -m app.preprocess \
  --input data/candidates.jsonl \
  --out-dir data/processed/full_clean

python3 -m app.risk \
  --input data/processed/full_clean/candidates_clean.jsonl \
  --out-dir data/processed/full_risk

python3 -m app.features \
  --input data/processed/full_risk/candidates_with_risk.jsonl \
  --out-dir data/processed/full_features_transformer \
  --semantic-backend transformer \
  --semantic-model BAAI/bge-small-en-v1.5 \
  --semantic-batch-size 64 \
  --semantic-no-fallback \
  --force

python3 -m app.scoring \
  --features data/processed/full_features_transformer/candidate_features.jsonl \
  --out data/processed/full_rank_transformer/submission.csv \
  --audit-out data/processed/full_rank_transformer/audit.jsonl \
  --top-n 100
```

## Final Top-100 Audit

Latest transformer submission audit:

```text
rows: 100
unique candidate IDs: 100
semantic encoded in top 100: 100
semantic skipped in top 100: 0
weak career evidence in top 100: 0
non-technical titles in top 100: 0
120-day notice in top 100: 0
services-only histories in top 100: 0
current services context: 1 candidate, not services-only, 60-day notice
```

## Output Format

The submission CSV follows:

```csv
candidate_id,rank,score,reasoning
```

Rules:

- exactly 100 candidate rows
- ranks 1 through 100 exactly once
- unique candidate IDs
- scores monotonically non-increasing
- UTF-8 CSV
- reasoning is grounded in candidate profile facts

## Official Submission Checklist

CSV:

- [x] CSV file exists at `data/processed/full_rank_transformer/submission.csv`
- [x] Header is exactly `candidate_id,rank,score,reasoning`
- [x] Exactly 100 data rows plus one header row
- [x] Ranks are exactly 1 through 100, each used once
- [x] Candidate IDs are unique and use `CAND_XXXXXXX`
- [x] Scores are monotonically non-increasing
- [x] Reasoning column is populated with profile-grounded 1-2 sentence explanations
- [x] Local validator passes
- [ ] Rename/copy final file to registered team filename, for example `YOUR_TEAM_ID.csv`
- [ ] Validate the renamed file before upload

Portal metadata to prepare:

- [ ] Team name
- [ ] Primary contact name
- [ ] Primary contact email
- [ ] Primary contact phone
- [ ] GitHub repository URL
- [ ] Sandbox/demo link or self-contained Docker run recipe
- [ ] AI tools declaration
- [ ] Compute environment summary
- [ ] Team member list
- [ ] Methodology summary, 200 words or fewer

Repository checklist for Stage 3:

- [x] Source code included under `app/`
- [x] README includes setup and reproduction commands
- [x] `requirements.txt` for the core pipeline
- [x] `requirements-transformer.txt` for optional transformer backend
- [x] `submission_metadata.yaml` template at repo root
- [x] `sandbox_app.py` small-sample demo app
- [x] Validator included under `validator/`
- [x] No hosted LLM/API calls during ranking
- [x] CPU-only ranking from precomputed feature records
- [ ] Fill `submission_metadata.yaml` TODO values before sharing repo
- [ ] Provide a working sandbox/demo link or Docker recipe

Manual review readiness:

- [x] Reasoning mentions specific titles, companies, years, skills, and concerns where relevant
- [x] Reasoning is varied, not one repeated template
- [x] No non-technical profiles in final top 100
- [x] No services-only profiles in final top 100
- [x] No 120-day-notice profiles in final top 100
- [x] No semantic-prefilter-skipped candidates in final top 100

## Tests

Run:

```bash
python3 -m unittest discover -s tests
```

Current result:

```text
Ran 26 tests
OK
```

## Sandbox Demo

The submission spec requires a hosted sandbox/demo link. This repo includes a small-sample Gradio app:

```bash
pip install -r requirements-sandbox.txt
python sandbox_app.py
```

The sandbox accepts a candidate `.json`, `.jsonl`, or `.jsonl.gz` upload, runs the offline hashed-backend pipeline, and returns a ranked CSV. It is meant for <=100 candidate sanity checks, as requested by the spec. The full 100k ranking is reproduced through the repository commands above.

Recommended hosting options:

- HuggingFace Spaces
- Streamlit Cloud
- Replit
- Google Colab notebook
- Docker image with a documented `docker run` command

Deploy to Hugging Face Spaces:

```bash
python3 -m venv .venv-hf
.venv-hf/bin/python -m pip install "huggingface_hub[cli]"
.venv-hf/bin/hf auth login
scripts/deploy_hf_space.sh YOUR_HF_USERNAME/redrob-candidate-ranker-sandbox
```

After deployment, add the URL to `submission_metadata.yaml`:

```text
https://huggingface.co/spaces/YOUR_HF_USERNAME/redrob-candidate-ranker-sandbox
```

## Notes On Compute Constraints

The final scoring step is CPU-only, offline, and fast from precomputed feature records. Transformer feature extraction is an optional local precompute stage; it uses no hosted LLM/API and no GPU, but it may exceed five minutes on CPU for 100,000 candidates.

For the strictest reproduction setting, use the hashed backend or provide the precomputed feature artifact expected by `app.scoring`.
