# Transformer Semantic Backend Setup

The real transformer backend needs two things:

1. Python packages from `requirements-transformer.txt`
2. A real local model directory, or a one-time model download before final ranking

## Install Runtime

Use Python 3.10-3.12 when possible. Python 3.14 may work only if compatible
wheels are available for your machine.

```bash
python3 -m venv .venv-transformer
source .venv-transformer/bin/activate
pip install -r requirements-transformer.txt
```

## One-Time Model Download

This uses the network once to populate the Hugging Face cache. Do not rely on
network access during final ranking.

```bash
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')"
```

After that, run transformer mode using the Hugging Face model id with local
files only:

```bash
python3 -m app.features \
  --input data/processed/full_risk/candidates_with_risk.jsonl \
  --out-dir data/processed/full_features_transformer \
  --semantic-backend transformer \
  --semantic-model BAAI/bge-small-en-v1.5 \
  --semantic-batch-size 64 \
  --semantic-no-fallback
```

## Local Folder Alternative

If you manually download/export the model to a folder, replace the placeholder
with the real path:

```bash
python3 -m app.features \
  --input data/processed/full_risk/candidates_with_risk.jsonl \
  --out-dir data/processed/full_features_transformer \
  --semantic-backend transformer \
  --semantic-model /absolute/path/to/bge-small-en-v1.5 \
  --semantic-batch-size 64 \
  --semantic-no-fallback
```

`/path/to/local/bge-small-en-v1.5` is only a placeholder. It will fail unless
that directory actually exists.

## First Run With Download Allowed

For setup only, this can download cache-missing files:

```bash
python3 -m app.features \
  --input data/processed/sample_risk/candidates_with_risk.jsonl \
  --out-dir data/processed/sample_features_transformer \
  --semantic-backend transformer \
  --semantic-model BAAI/bge-small-en-v1.5 \
  --semantic-batch-size 64 \
  --semantic-allow-download \
  --semantic-no-fallback \
  --force
```

For final/offline ranking, omit `--semantic-allow-download`.

For Apple Silicon CPU, start with `--semantic-batch-size 64`. If memory spikes,
drop to `32`. If CPU is underused and memory is fine, try `128`.

By default, transformer mode now uses a cheap JD-evidence prefilter. It only
encodes plausible candidates and assigns no semantic boost to profiles with no
cheap retrieval/search/ranking/product-ML evidence. This is intentional: it is
faster and avoids lifting HR/accounting/marketing keyword traps.

To force transformer encoding for every candidate, add:

```bash
--semantic-no-prefilter
```

That full encode path is usually too slow on CPU for 100,000 candidates and is
not recommended for the final workflow.
