"""Offline semantic alignment helpers.

The default backend is a dependency-free hashed bi-encoder fallback so the
submission pipeline always runs on a clean offline machine.

The optional ``transformer`` backend uses a real local sentence-transformer or
Hugging Face AutoModel when the model files and runtime dependencies are already
available. It is intentionally local-files-only by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from app.io import iter_candidate_records, write_json, write_jsonl
from app.preprocess import normalize_label, normalize_text


SEMANTIC_VERSION = "semantic_v2"
DEFAULT_DIMENSIONS = 768
DEFAULT_TRANSFORMER_MODEL = "BAAI/bge-small-en-v1.5"

JD_SEMANTIC_PROFILE = """
Senior AI Engineer founding team role building the intelligence layer for
candidate discovery, job matching, retrieval, ranking, search, embeddings,
hybrid retrieval, evaluation, recruiter feedback loops, and production Python
systems. Ideal candidates have hands-on product engineering experience shipping
recommendation, search, ranking, retrieval, or matching systems to real users.
"""

CONCEPT_EXPANSIONS = {
    "retrieval": [
        "retrieval",
        "information retrieval",
        "candidate retrieval",
        "document retrieval",
        "semantic search",
        "hybrid search",
    ],
    "ranking": [
        "ranking",
        "ranker",
        "learning to rank",
        "reranking",
        "recommendation",
        "recommender",
        "matching",
        "job matching",
    ],
    "embeddings": [
        "embedding",
        "embeddings",
        "sentence transformer",
        "vector representation",
        "semantic similarity",
    ],
    "vector_search": [
        "vector search",
        "vector database",
        "faiss",
        "milvus",
        "pinecone",
        "qdrant",
        "weaviate",
    ],
    "evaluation": [
        "ndcg",
        "mrr",
        "map",
        "offline evaluation",
        "online evaluation",
        "a b test",
        "ab test",
        "relevance evaluation",
    ],
    "production": [
        "production",
        "shipped",
        "deployed",
        "latency",
        "monitoring",
        "real users",
        "scale",
        "on call",
    ],
    "python": ["python", "pyspark", "fastapi", "flask", "django"],
}

POSITIVE_ANCHORS = [
    "Senior AI engineer who shipped production candidate retrieval, job matching, search ranking, recommendation, and hybrid retrieval systems.",
    "Machine learning engineer building embeddings, vector search, semantic search, ranking evaluation, A/B tests, and feedback loops for real users.",
    "Product-focused Python engineer owning retrieval infrastructure, recommender systems, learning-to-rank models, latency, monitoring, and offline-online relevance metrics.",
]

NEGATIVE_ANCHORS = [
    "HR manager, recruiter, marketing, sales, operations, finance, or customer support profile using AI tools without building production AI systems.",
    "Computer vision, speech, robotics, or pure research profile with little retrieval, search, ranking, recommendation, or matching product experience.",
    "Services or consulting-only AI profile focused on demos, wrappers, presentations, generic GenAI tools, or keyword-stuffed skills without shipped product ownership.",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}


def candidate_semantic_text(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_company", ""),
        profile.get("current_industry", ""),
    ]
    for job in candidate.get("career_history", []):
        parts.extend(
            [
                job.get("title", ""),
                job.get("company", ""),
                job.get("industry", ""),
                job.get("description", ""),
            ]
        )
    for skill in candidate.get("skills", []):
        name = normalize_text(skill.get("name"))
        if name:
            parts.append(name)
    return normalize_label(" ".join(parts))


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", normalize_label(text)) if token not in STOPWORDS]


def stable_bucket(token: str, dimensions: int) -> tuple[int, int]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    number = int.from_bytes(digest, "big")
    return number % dimensions, 1 if number & 1 else -1


def encode_text(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> dict[int, float]:
    tokens = tokenize(text)
    weighted_terms: Counter[str] = Counter(tokens)
    weighted_terms.update(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))

    normalized = normalize_label(text)
    for concept, phrases in CONCEPT_EXPANSIONS.items():
        hits = sum(1 for phrase in phrases if phrase in normalized)
        if hits:
            weighted_terms[f"concept_{concept}"] += hits * 4

    vector: dict[int, float] = {}
    for term, count in weighted_terms.items():
        bucket, sign = stable_bucket(term, dimensions)
        vector[bucket] = vector.get(bucket, 0.0) + sign * (1.0 + math.log(count))
    return vector


@lru_cache(maxsize=8)
def reference_vector(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> tuple[tuple[int, float], ...]:
    return tuple(sorted(encode_text(text, dimensions).items()))


def vector_from_items(items: tuple[tuple[int, float], ...]) -> dict[int, float]:
    return dict(items)


def cosine_similarity(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(index, 0.0) for index, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, dot / (left_norm * right_norm))


def dense_cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def concept_alignment(text: str) -> float:
    normalized = normalize_label(text)
    weights = {
        "retrieval": 0.18,
        "ranking": 0.2,
        "embeddings": 0.12,
        "vector_search": 0.1,
        "evaluation": 0.16,
        "production": 0.14,
        "python": 0.1,
    }
    score = 0.0
    for concept, weight in weights.items():
        phrases = CONCEPT_EXPANSIONS[concept]
        if any(phrase in normalized for phrase in phrases):
            score += weight
    return min(1.0, score)


class TextEncoder(Protocol):
    backend_name: str
    model_name: str

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str, local_files_only: bool = True) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc

        try:
            self.model = SentenceTransformer(model_name, device="cpu", local_files_only=local_files_only)
        except TypeError:
            if local_files_only:
                raise RuntimeError("installed sentence-transformers does not support local_files_only") from None
            try:
                self.model = SentenceTransformer(model_name, device="cpu")
            except Exception as exc:
                raise RuntimeError(f"could not load sentence-transformer model {model_name!r}: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"could not load sentence-transformer model {model_name!r}: {exc}") from exc
        self.backend_name = "sentence_transformer"
        self.model_name = model_name

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size)
        return [list(map(float, row)) for row in embeddings]


class TransformersAutoModelEncoder:
    def __init__(self, model_name: str, local_files_only: bool = True) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for transformer semantic backend") from exc

        self.torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        except Exception as exc:
            raise RuntimeError(f"could not load transformers model {model_name!r}: {exc}") from exc
        self.model.eval()
        self.backend_name = "transformers_auto_model"
        self.model_name = model_name

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        embeddings = []
        for start in range(0, len(texts), batch_size):
            embeddings.extend(self._encode_batch(texts[start : start + batch_size]))
        return embeddings

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with self.torch.no_grad():
            output = self.model(**encoded)
        token_embeddings = output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = (token_embeddings * attention_mask).sum(1) / attention_mask.sum(1).clamp(min=1e-9)
        pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
        return [[float(value) for value in row] for row in pooled.tolist()]


@lru_cache(maxsize=4)
def load_transformer_encoder(model_name: str, local_files_only: bool = True) -> TextEncoder:
    errors = []
    for encoder_type in (SentenceTransformerEncoder, TransformersAutoModelEncoder):
        try:
            return encoder_type(model_name, local_files_only=local_files_only)
        except RuntimeError as exc:
            errors.append(str(exc))
    raise RuntimeError("; ".join(errors))


@lru_cache(maxsize=8)
def encoded_anchors(model_name: str, local_files_only: bool = True) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...], str]:
    encoder = load_transformer_encoder(model_name, local_files_only=local_files_only)
    positive = encoder.encode(POSITIVE_ANCHORS)
    negative = encoder.encode(NEGATIVE_ANCHORS)
    return (
        tuple(tuple(row) for row in positive),
        tuple(tuple(row) for row in negative),
        encoder.backend_name,
    )


def transformer_alignment_score(
    candidate: dict[str, Any],
    model_name: str = DEFAULT_TRANSFORMER_MODEL,
    local_files_only: bool = True,
) -> dict[str, Any]:
    return transformer_alignment_scores([candidate], model_name=model_name, local_files_only=local_files_only)[0]


def transformer_alignment_scores(
    candidates: list[dict[str, Any]],
    model_name: str = DEFAULT_TRANSFORMER_MODEL,
    local_files_only: bool = True,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    encoder = load_transformer_encoder(model_name, local_files_only=local_files_only)
    positive, negative, backend_name = encoded_anchors(model_name, local_files_only=local_files_only)
    candidate_texts = [candidate_semantic_text(candidate) for candidate in candidates]
    candidate_embeddings = encoder.encode(candidate_texts, batch_size=batch_size)
    results = []
    for candidate_embedding in candidate_embeddings:
        positive_similarity = max(dense_cosine(candidate_embedding, list(anchor)) for anchor in positive)
        negative_similarity = max(dense_cosine(candidate_embedding, list(anchor)) for anchor in negative)
        contrastive = positive_similarity - negative_similarity
        score = max(0.0, min(1.0, (contrastive + 0.35) / 0.7))
        results.append(
            {
                "backend": backend_name,
                "model_name": model_name,
                "stage_version": SEMANTIC_VERSION,
                "positive_similarity": round(positive_similarity, 6),
                "negative_similarity": round(negative_similarity, 6),
                "contrastive_similarity": round(contrastive, 6),
                "score": round(score, 4),
            }
        )
    return results


def hashed_alignment_score(candidate: dict[str, Any], jd_text: str = JD_SEMANTIC_PROFILE) -> dict[str, Any]:
    candidate_text = candidate_semantic_text(candidate)
    raw_similarity = cosine_similarity(vector_from_items(reference_vector(jd_text)), encode_text(candidate_text))
    concept_score = concept_alignment(candidate_text)
    raw_score = min(1.0, raw_similarity / 0.34)
    score = 0.25 * raw_score + 0.75 * concept_score
    return {
        "backend": "hashed_bi_encoder",
        "stage_version": SEMANTIC_VERSION,
        "concept_alignment": round(concept_score, 4),
        "raw_similarity": round(raw_similarity, 6),
        "score": round(score, 4),
    }


def semantic_alignment_score(
    candidate: dict[str, Any],
    jd_text: str = JD_SEMANTIC_PROFILE,
    backend: str = "hashed",
    model_name: str = DEFAULT_TRANSFORMER_MODEL,
    local_files_only: bool = True,
    allow_fallback: bool = True,
    batch_size: int = 32,
) -> dict[str, Any]:
    return semantic_alignment_scores(
        [candidate],
        jd_text=jd_text,
        backend=backend,
        model_name=model_name,
        local_files_only=local_files_only,
        allow_fallback=allow_fallback,
        batch_size=batch_size,
    )[0]


def semantic_alignment_scores(
    candidates: list[dict[str, Any]],
    jd_text: str = JD_SEMANTIC_PROFILE,
    backend: str = "hashed",
    model_name: str = DEFAULT_TRANSFORMER_MODEL,
    local_files_only: bool = True,
    allow_fallback: bool = True,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    if backend == "hashed":
        return [hashed_alignment_score(candidate, jd_text=jd_text) for candidate in candidates]
    if backend != "transformer":
        raise ValueError(f"unknown semantic backend: {backend}")
    try:
        return transformer_alignment_scores(
            candidates,
            model_name=model_name,
            local_files_only=local_files_only,
            batch_size=batch_size,
        )
    except RuntimeError as exc:
        if not allow_fallback:
            raise
        results = []
        for candidate in candidates:
            fallback = hashed_alignment_score(candidate, jd_text=jd_text)
            fallback["backend"] = "hashed_bi_encoder_fallback"
            fallback["requested_backend"] = "transformer"
            fallback["fallback_reason"] = str(exc)
            fallback["model_name"] = model_name
            results.append(fallback)
        return results


def annotate_file(
    input_path: str | Path,
    output_path: str | Path,
    backend: str = "hashed",
    model_name: str = DEFAULT_TRANSFORMER_MODEL,
    local_files_only: bool = True,
    allow_fallback: bool = True,
    batch_size: int = 32,
) -> dict[str, Any]:
    total = 0
    annotated = []
    backend_counts: Counter[str] = Counter()
    for total, candidate in enumerate(iter_candidate_records(input_path), start=1):
        semantic = semantic_alignment_scores(
            [candidate],
            backend=backend,
            model_name=model_name,
            local_files_only=local_files_only,
            allow_fallback=allow_fallback,
            batch_size=batch_size,
        )[0]
        backend_counts[semantic.get("backend", "unknown")] += 1
        annotated.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "semantic_alignment": semantic,
            }
        )
    write_jsonl(output_path, annotated)
    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_records": total,
        "stage_version": SEMANTIC_VERSION,
        "requested_backend": backend,
        "effective_backend_counts": dict(sorted(backend_counts.items())),
        "model_name": model_name if backend == "transformer" else None,
        "local_files_only": local_files_only,
    }
    write_json(Path(output_path).with_suffix(".report.json"), report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute offline semantic alignment artifacts.")
    parser.add_argument("--input", required=True, help="Path to risk-annotated candidates")
    parser.add_argument("--out", required=True, help="Output JSONL path for semantic features")
    parser.add_argument("--backend", choices=["hashed", "transformer"], default="hashed", help="Semantic backend")
    parser.add_argument("--model", default=DEFAULT_TRANSFORMER_MODEL, help="Local transformer model path or HF model id")
    parser.add_argument("--allow-download", action="store_true", help="Allow model loading to use network/cache misses")
    parser.add_argument("--no-fallback", action="store_true", help="Fail instead of falling back when transformer backend is unavailable")
    parser.add_argument("--batch-size", type=int, default=32, help="Transformer encoding batch size")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = annotate_file(
        args.input,
        args.out,
        backend=args.backend,
        model_name=args.model,
        local_files_only=not args.allow_download,
        allow_fallback=not args.no_fallback,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
