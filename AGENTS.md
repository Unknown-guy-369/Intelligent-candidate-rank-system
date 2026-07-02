# AGENTS.md

Guidance for agents working in this repository.

## Project Mission

Build an offline candidate selection and ranking system for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

The system must rank the top 100 candidates for the Senior AI Engineer - Founding Team JD. It must analyze full candidate profiles, not just count keywords.

## First Files To Read

Read these before making implementation decisions:

1. `docs/job_description.docx`
2. `docs/submission_spec.docx`
3. `docs/redrob_signals_doc.docx`
4. `schema/candidate_schema.json`
5. `docs/candidate_tiers.md`
6. `docs/engineering_plan.md`
7. `sample/sample_candidates.json`
8. `validator/validate_submission.py`

Note: root `README.md` may be empty until implementation docs are written. The bundled README content is currently in `docs/README.docx`.

## Core Engineering Principle

Do not build a keyword matcher.

The JD explicitly warns that strong submissions must reason about what the JD means, not only what words appear in the skills list. A candidate with many AI keywords but unrelated career history should be penalized. A candidate who built production recommendation/search/ranking systems in plain language may be highly suitable.

## Target Role Summary

The JD is for a founding Senior AI Engineer at Redrob AI, focused on the intelligence layer:

- candidate/job ranking
- retrieval
- matching
- search
- embeddings
- hybrid retrieval
- LLM integration
- ranking evaluation
- recruiter feedback loops

The ideal candidate is roughly:

- 6-8 years total experience
- 4-5 years applied ML/AI in product companies
- shipped at least one ranking, search, recommendation, retrieval, or matching system to real users
- strong Python
- hands-on production engineer, not only architect/researcher
- comfortable with a scrappy Series A product environment
- active and reachable
- Pune/Noida or willing to relocate from a Tier-1 Indian city

## Architecture Direction

Use an offline evidence-based ranking pipeline:

```text
candidates.jsonl.gz
  -> streaming loader
  -> cleaner / normalizer
  -> risk and honeypot detector
  -> feature extractor
  -> suitability tier estimator
  -> scoring engine
  -> top-100 selector
  -> factual reasoning generator
  -> submission.csv
```

Prefer a transparent feature-based ranker first. This is a hackathon with one JD, 100,000 candidates, and strict compute limits. A simple defensible system is better than an overbuilt model that cannot be reproduced.

## Compute And Submission Constraints

The ranking step must satisfy:

- CPU only
- no network calls
- no hosted LLM APIs
- no GPU
- <= 5 minutes wall-clock
- <= 16 GB RAM
- <= 5 GB intermediate disk
- output exactly 100 candidates

CSV columns must be exactly:

```csv
candidate_id,rank,score,reasoning
```

Validate with:

```bash
python validator/validate_submission.py submission.csv
```

## Recommended Technology

Use Python standard library where possible:

- `gzip`
- `json`
- `csv`
- `re`
- `datetime`
- `math`
- `heapq`
- `argparse`
- `pathlib`
- `collections`
- `statistics`

Optional dependencies should be minimal:

- `orjson` for faster JSON parsing
- `pytest` for tests
- `tqdm` for progress display
- `scikit-learn` only if lightweight text scoring is needed

Avoid:

- hosted LLM APIs during ranking
- heavy local LLM inference
- vector databases for the core ranking path
- GPU-only workflows
- black-box scoring that cannot be defended

## Data Handling Rules

- Never modify the raw candidate dataset.
- Stream full candidate files line by line.
- Keep memory bounded.
- Preserve an audit trail for rejected or heavily penalized candidates.
- Do not blindly delete suspicious profiles. Reject only severe contradictions; penalize uncertain risk.

Planned processed artifacts:

```text
data/processed/candidates_clean.jsonl
data/processed/candidate_features.jsonl
data/processed/rejected_candidates.jsonl
data/processed/preprocessing_report.json
```

## Suitability Tiers

Use `docs/candidate_tiers.md` as the tier rubric:

- Tier 5: ideal match
- Tier 4: strong match with minor gaps
- Tier 3: plausible / transferable match
- Tier 2: weak adjacent fit
- Tier 1: poor fit / keyword trap
- Tier 0: reject / honeypot / disqualifier

The tier is an estimated suitability category, not a ground-truth label. Final rank should be based on evidence and scoring.

## Scoring Guidance

The final score should combine:

- JD fit
- career evidence
- production ML/search/ranking experience
- skill depth and support
- product-company/startup fit
- behavioral availability
- location and notice-period logistics
- risk penalties

Career evidence should outweigh skills. Skills are noisy and intentionally contain traps.

## Trap And Honeypot Awareness

Watch for:

- impossible timelines
- years of experience inconsistent with career history
- expert skills with zero or near-zero duration
- AI keyword stuffing without career evidence
- non-technical current roles with dense AI skills
- recent LangChain/OpenAI-wrapper-only experience
- pure research with no production deployment
- senior architects/managers with no recent production coding
- consulting-only careers with no product-company evidence
- CV/speech/robotics-only backgrounds without NLP/IR/retrieval overlap
- inactive candidates with very low response rates

## Reasoning Requirements

Reasoning should be 1-2 sentences per selected candidate.

Rules:

- Use only facts present in the candidate profile.
- Connect the candidate to specific JD requirements.
- Mention concerns honestly when relevant.
- Avoid hallucinated skills, employers, metrics, or achievements.
- Avoid repeated templated text.
- Make tone consistent with rank.

## Suggested Implementation Files

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

rank.py
requirements.txt
README.md
submission_metadata.yaml
tests/
```

## Git And Existing Changes

The worktree may contain user changes. Do not revert or overwrite them unless explicitly asked.

At the time this file was created, `sample/sample_submission.csv` was already modified. Treat that as user-owned unless the user asks to edit it.

