# Candidate Suitability Tiers For The Senior AI Engineer JD

This rubric converts the job description into practical candidate tiers. It should guide preprocessing, scoring, ranking, and reasoning generation.

Important correction: these tiers are not labels from the dataset. They are our interpretation of the JD. The ranker should estimate a candidate's tier from evidence in the profile, not assign tiers from keywords alone.

## JD Anchors

The target role in `docs/job_description.docx` is not a generic AI Engineer role. It is a founding Senior AI Engineer role for Redrob's intelligence layer: ranking, retrieval, matching, and candidate/job search.

The strongest candidates should match these JD-specific anchors:

- Senior but still hands-on: roughly 5-9 years, with the ideal profile around 6-8 years.
- 4-5 years of applied ML/AI work in product companies, not only services or research.
- Has shipped at least one end-to-end ranking, search, recommendation, retrieval, or matching system to real users.
- Knows embeddings, vector search, hybrid retrieval, and LLM integration from production experience.
- Understands ranking evaluation: NDCG, MRR, MAP, offline-to-online correlation, A/B tests, or feedback loops.
- Strong Python and current production coding ability.
- Product-engineering mindset: can ship a useful v2 ranker quickly, then improve it with evidence.
- Suitable for a changing Series A environment: owns ambiguous problems, writes clearly, works with PM/recruiter workflows, and can mentor future hires.
- Reachable and hireable: active, responsive, open to work, reasonable notice period, and Pune/Noida or willing to relocate from a Tier-1 Indian city.

## Tier 5: Ideal Match

Candidate should be ranked near the top unless behavioral signals are poor.

Expected evidence:

- 6-8 years total experience, or slightly outside that range with unusually strong senior-engineer judgment.
- 4-5 years in applied ML, NLP, retrieval, ranking, recommendations, search, or matching systems.
- Has shipped production systems to real users.
- Direct experience with embeddings-based retrieval, vector databases, hybrid search, ranking, recommendation systems, or candidate/job matching.
- Strong Python and hands-on production coding.
- Understands ranking evaluation: NDCG, MRR, MAP, offline benchmarks, A/B tests, or recruiter/user feedback loops.
- Product-company, marketplace, SaaS, HR-tech, search, recommendation, or AI platform experience rather than only services delivery.
- Shows product-engineering judgment: can ship, measure, learn, and iterate.
- Has opinions about hybrid vs dense retrieval, offline vs online evaluation, and fine-tuning vs prompting, backed by real systems.
- Active and reachable: recent activity, open to work, good recruiter response rate.
- Location/logistics reasonable: Pune/Noida, nearby Tier-1 Indian city, or willing to relocate.

Typical profile:

- Senior ML Engineer, AI Engineer, Search Engineer, Ranking Engineer, Recommendation Systems Engineer, Applied Scientist with production deployment, or backend/data engineer who clearly owned production retrieval/recommendation systems.

## Tier 4: Strong Match With Minor Gaps

Candidate is suitable and likely belongs in top 100.

Expected evidence:

- Strong ML/search/retrieval background but missing one major ideal signal.
- May have less explicit ranking evaluation experience but has shipped related production systems and understands metrics.
- May be slightly outside the 5-9 year band but still senior enough and hands-on.
- May have location, salary, or notice-period friction, but not severe.
- May come from a larger company but still has startup/product-building attitude.
- May have product-company experience but limited HR-tech/recruiting exposure.

Common gaps:

- Good retrieval experience but limited LLM fine-tuning.
- Good ML platform/search experience but no HR-tech exposure.
- Strong backend/data plus search/recommendation experience, but not titled as AI Engineer.
- Excellent profile but 30-60 day notice period.

## Tier 3: Plausible / Transferable Match

Candidate may be worth considering, but should usually rank below Tier 4.

Expected evidence:

- Adjacent experience in data engineering, backend systems, analytics engineering, ML infrastructure, or NLP.
- Some production exposure, but retrieval/ranking/recommendation evidence is weak or indirect.
- Has Python and useful systems experience.
- Skills suggest AI capability, but career history only partially supports it.
- Behavioral signals are good enough to make them reachable.
- Could ramp into Redrob's intelligence-layer work, but would need support on ranking evaluation or retrieval architecture.

Typical profile:

- Backend/data engineer who built feature pipelines for ML teams.
- ML engineer with model-building experience but limited ranking/search systems.
- Data scientist with production deployment experience but limited engineering ownership.

Risk:

- These candidates can be good hidden gems, but they should not outrank candidates with direct production retrieval/ranking evidence.

## Tier 2: Weak Adjacent Fit

Candidate has some relevant terms or broad technical ability, but is not a strong fit for this JD.

Expected evidence:

- AI/ML skills listed, but little career evidence of production AI systems.
- Mostly consulting/services background with no product-company experience.
- Experience mainly in dashboards, analytics, academic projects, tutorials, or internal prototypes.
- Current role is not close to AI/search/ranking.
- Behavioral signals may be average or weak.
- Work may be technically competent, but not clearly aligned with owning Redrob's ranking/retrieval product layer.

Typical profile:

- Generic software engineer with AI side projects.
- Data analyst with ML keywords.
- Services-company engineer with many tools listed but no ownership evidence.

Ranking decision:

- Usually should not enter top 100 unless the candidate has unusually strong supporting evidence elsewhere.

## Tier 1: Poor Fit / Keyword Trap

Candidate should generally be excluded from top 100.

Expected evidence:

- Many AI keywords but unrelated career path.
- Current title and career history are in marketing, HR, sales, accounting, design, support, or non-technical roles.
- Skills include LLM/RAG/vector DB terms, but descriptions do not show production engineering work.
- Mostly recent LangChain/OpenAI demos without pre-LLM retrieval or ML systems depth.
- Title progression suggests title-chasing or frequent seniority jumps without deep ownership.
- Poor availability: inactive for months, very low recruiter response rate, not open to work.

Typical profile:

- Marketing Manager with "RAG, LLM, Pinecone" in skills.
- HR Manager with many AI skill endorsements but no technical career evidence.
- Content/profile keyword stuffing.

Ranking decision:

- Keep in data for audit, but strongly penalize.

## Tier 0: Reject / Honeypot / Disqualifier

Candidate should be removed from ranking pool or receive a near-zero score.

Expected evidence:

- Impossible timeline or contradictory experience.
- Claimed years of experience conflict strongly with career history.
- Expert in many skills with zero or near-zero duration.
- Current role dates are invalid or impossible.
- Profile claims senior AI production experience, but career history has no matching evidence.
- Pure research profile with no production deployment.
- Senior architect/manager who has not written production code recently.
- AI experience is mainly under-12-month LangChain/OpenAI wrapper projects with no substantial pre-LLM retrieval/ranking or ML production background.
- Entire career only in pure consulting/services firms with no product-company evidence, unless there is strong prior product experience.
- Primary expertise is only computer vision, speech, or robotics without meaningful NLP, IR, retrieval, ranking, or recommendation experience.
- Closed-source-only work for many years with no visible external validation, if the profile gives no other way to assess how the candidate thinks.

Ranking decision:

- Reject only when evidence is severe.
- If uncertain, keep candidate but apply a large risk penalty.

## Tier-to-Score Guidance

Suggested score bands:

- Tier 5: `0.85-1.00`
- Tier 4: `0.70-0.84`
- Tier 3: `0.50-0.69`
- Tier 2: `0.30-0.49`
- Tier 1: `0.10-0.29`
- Tier 0: `0.00-0.09`

The ranker should compute the final score from evidence, not assign a fixed score only from the tier. Behavioral signals and risk flags can move candidates up or down within a band.

## Signals That Should Move A Candidate Up

- Built search, ranking, recommendation, matching, or retrieval systems.
- Mentions production deployment, real users, scale, index refresh, embedding drift, retrieval-quality regression, evaluation, monitoring, or iteration.
- Python is central to their work.
- Experience with FAISS, Milvus, Pinecone, Qdrant, Weaviate, Elasticsearch, OpenSearch, BGE, E5, sentence-transformers, RAG, or embeddings.
- Has ranking evaluation experience.
- Has product/startup ownership.
- Has marketplace, HR-tech, recruiting-tech, search, SaaS, or AI-platform context.
- Recent Redrob activity and good recruiter response rate.

## Signals That Should Move A Candidate Down

- Skill list is stronger than career history.
- Heavy keyword stuffing.
- Pure research without deployment.
- Only tutorials or demo-framework experience.
- Recent LangChain/OpenAI wrapper experience without deeper retrieval/ranking background.
- Senior title but no evidence of hands-on production coding in the last 18 months.
- Only consulting/services background with no product-company exposure.
- Pure CV, speech, or robotics focus without significant NLP/IR/retrieval overlap.
- Long inactivity or very low response rate.
- Notice period above 60 days.
- No willingness to relocate when location is far from Pune/Noida.
- Contradictory timelines or impossible skill claims.
