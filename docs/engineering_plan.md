# Engineering Plan: Offline Candidate Selection System

## Goal

Build an offline ranking system for the Redrob hackathon dataset that selects the top 100 candidates for the Senior AI Engineer JD.

The system must not behave like keyword matching. It should reason over the full candidate profile: career history, role trajectory, production ML/search/ranking experience, skills, education, behavioral Redrob signals, availability, and contradictions that indicate traps or honeypots.

## Honest Engineering Correction

We should not claim we can remove all traps or honeypots during preprocessing. The dataset does not provide honeypot labels, and the hidden evaluator may reward some candidates that look unusual but are valid.

The correct approach is:

1. Preserve the raw candidate dataset unchanged.
2. Create a cleaned and normalized representation for ranking.
3. Add trap and honeypot risk flags.
4. Exclude only candidates with severe contradictions.
5. Penalize medium-risk candidates rather than deleting them blindly.
6. Keep an audit trail explaining every exclusion or penalty.

This is more defensible than hard-deleting candidates based on fragile rules.

## Stage 1: Data Ingestion

Inputs:

- `candidates.jsonl` or `candidates.jsonl.gz`
- `schema/candidate_schema.json`
- `docs/job_description.docx`
- `docs/redrob_signals_doc.docx`

Planned outputs:

- `data/processed/candidates_clean.jsonl`
- `data/processed/candidate_features.jsonl`
- `data/processed/rejected_candidates.jsonl`
- `data/processed/preprocessing_report.json`

Required behavior:

- Stream candidates line by line to support 100,000 records within memory limits.
- Validate required fields from the schema.
- Normalize strings, dates, numeric fields, skill names, company sizes, and missing optional values.
- Never modify the original dataset.

## Stage 2: Basic Data Cleaning

Clean and normalize:

- Empty strings, nulls, malformed dates, invalid durations.
- Duplicated skills with inconsistent casing.
- Skill proficiency into numeric scale.
- Endorsements and duration fields into bounded numeric values.
- Country, location, work mode, relocation, and notice period values.
- Career history sorted by date.
- Current role consistency between `profile.current_title` and current career entry.

Candidates should be rejected only when required fields are missing or impossible to use safely.

## Stage 3: Trap And Honeypot Detection

Create a `risk_flags` object for each candidate.

High-severity flags:

- Career duration contradicts profile years of experience by a large margin.
- Current job has impossible date logic.
- Skill marked expert with zero or near-zero duration across many skills.
- Candidate has many advanced AI skills but no supporting career evidence.
- Profile says senior AI/ML role, but career history is unrelated and generic.
- Honeypot-like impossible company timeline or experience timeline.
- Repeated copy-paste career descriptions across unrelated roles.

Medium-severity flags:

- AI keyword stuffing in skills without matching job descriptions.
- Very high endorsements but weak profile completeness or no assessment support.
- Excellent on-paper match but inactive for many months.
- Very low recruiter response rate.
- Long notice period for an urgent startup role.
- All experience from pure services/consulting companies with no product-company exposure.
- Mainly CV/speech/robotics background without NLP, retrieval, ranking, or recommendation evidence.

Low-severity flags:

- Missing GitHub.
- Missing LinkedIn.
- Slightly outside 5-9 year experience band.
- Not currently in Pune/Noida but willing to relocate.

Preprocessing decision:

- Severe impossible profiles: reject from ranking pool.
- Multiple high-severity flags: reject or heavily penalize.
- Medium-risk profiles: keep but downweight.
- Low-risk profiles: keep with small penalty only if relevant.

## Stage 4: Candidate Attribute Extraction

Extract structured attributes that matter for the JD:

- Total years of experience.
- Years in applied ML, AI, NLP, retrieval, ranking, recommendations, or search.
- Production evidence from career descriptions.
- Product-company vs pure services/consulting background.
- Startup or high-ownership environment signal.
- Python strength.
- Vector search, hybrid retrieval, FAISS, Milvus, Pinecone, Elasticsearch, OpenSearch, Qdrant, Weaviate.
- Embeddings, sentence-transformers, BGE, E5, RAG.
- Ranking evaluation: NDCG, MRR, MAP, A/B testing, offline evaluation, online metrics.
- LLM fine-tuning, LoRA, QLoRA, PEFT as optional positives.
- Evidence of shipping systems to real users.
- Leadership or mentoring without being too far from hands-on coding.
- Location, relocation, notice period, availability.
- Redrob behavioral engagement.

## Stage 5: Scoring Model

Use a transparent hybrid scoring model first.

Suggested score groups:

- JD fit score: production AI/search/ranking experience.
- Offline semantic alignment: cached bi-encoder style JD/profile similarity.
- Career evidence score: actual responsibilities and shipped systems.
- Skill support score: skills validated by duration, proficiency, endorsements, and assessment scores.
- Company/context score: product-company, startup, scale, non-consulting fit.
- Availability score: active, open to work, response rate, notice period, relocation.
- Risk penalty: honeypot/trap signals and contradictions.

Avoid relying on skill count alone. Skill lists are intentionally noisy. The
default semantic backend is a dependency-free hashed bi-encoder fallback, and
the optional `transformer` backend uses a real local sentence-transformer or
Hugging Face AutoModel with positive and negative JD anchors. Transformer mode
must use local model files during final ranking; network download is only a
setup/precomputation step.

Transformer-mode example:

```bash
python3 -m app.features \
  --input data/processed/full_risk/candidates_with_risk.jsonl \
  --out-dir data/processed/full_features_transformer \
  --semantic-backend transformer \
  --semantic-model /path/to/local/bge-small-en-v1.5 \
  --semantic-no-fallback
```

## Stage 6: Ranking Output

Final ranker should:

- Score all non-rejected candidates.
- Sort by score descending.
- Break ties deterministically by candidate ID ascending.
- Select exactly 100.
- Generate 1-2 sentence reasoning using only facts present in the candidate profile.
- Mention concerns honestly for lower-ranked or risky candidates.
- Write `candidate_id,rank,score,reasoning`.
- Validate with `validator/validate_submission.py`.

## Stage 7: Reproducibility

The final repo should include:

- A clear `README.md`.
- One command to generate the submission CSV.
- Requirements file.
- Preprocessing script.
- Ranking script.
- Validation command.
- Metadata YAML.

Target command:

```bash
python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
```

## Proposed File Structure

```text
app/
  __init__.py
  io.py
  preprocess.py
  features.py
  risk.py
  scoring.py
  reasoning.py
  rank.py

data/
  processed/

tests/
  test_preprocess.py
  test_risk.py
  test_scoring.py

rank.py
requirements.txt
README.md
submission_metadata.yaml
```

## First Implementation Milestone

The first build should only do preprocessing and risk detection:

1. Load sample candidates.
2. Normalize candidate records.
3. Compute risk flags.
4. Produce a preprocessing report.
5. Print examples of rejected, penalized, and clean candidates.

Only after this works should we implement final scoring.
