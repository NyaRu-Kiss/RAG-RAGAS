<!--
Sync Impact Report
===================
Version change: (template) → 1.0.0
Rationale: Initial ratification. The constitution file previously contained
only unfilled template placeholders, so this is the first concrete adoption
of governing principles for this project — a MAJOR-tier initial version
(1.0.0), not an amendment to a prior baseline.

Modified principles: n/a (first fill of template placeholders)

Added sections:
- Core Principles: I. LlamaIndex-First (NON-NEGOTIABLE), II. Postgres +
  pgvector as the Storage Exception, III. Config-Driven Behavior — No
  Hardcoding, IV. Test-Backed Changes, V. Design-Before-Code for Non-Trivial
  Changes
- Technology Stack Constraints
- Development Workflow
- Governance (versioning policy, compliance review expectations)

Removed sections: none

Templates requiring updates:
- .specify/templates/plan-template.md — ✅ no change needed; its
  "Constitution Check" gate is populated dynamically per-feature from this
  file and contains no stale principle references.
- .specify/templates/spec-template.md — ✅ no change needed; generic,
  technology-agnostic, no principle-specific references to update.
- .specify/templates/tasks-template.md — ✅ no change needed; task
  categories (setup/foundational/user-story/polish) are generic and already
  compatible with the test-backed-changes and design-before-code principles.
- .specify/templates/checklist-template.md — ✅ no change needed; generic.
- README.md — ✅ already documents the LlamaIndex-first / pgvector-storage
  architecture this constitution codifies; no contradictions found.
- No CLAUDE.md/AGENTS.md agent-guidance file exists in this repo — nothing
  to reconcile.

Follow-up TODOs: none. RATIFICATION_DATE set to the date of this initial
adoption since no prior ratified constitution existed to date backward to.
-->

# LlamaIndex RAG Constitution

## Core Principles

### I. LlamaIndex-First (NON-NEGOTIABLE)

Every RAG capability — document loading, node parsing/chunking, embeddings,
retrieval, query fusion, reranking, LLM invocation, chat/query engines —
MUST be built on official LlamaIndex abstractions and integration packages
(e.g. `SimpleDirectoryReader`, the `NodeParser` family, `VectorStoreIndex`,
`QueryFusionRetriever`, `SentenceTransformerRerank`, `OpenAILike`) rather
than hand-rolled equivalents. A custom implementation is permitted only when
no LlamaIndex-native option exists, or the native option is materially
inconvenient (unsupported provider, missing feature, significant
performance gap). Every such deviation MUST be recorded with its rationale
in a design doc before the change is merged.

Rationale: LlamaIndex is this project's chosen orchestration framework.
Reinventing functionality it already provides increases maintenance cost,
diverges from its upgrade path, and duplicates effort the framework already
solved.

### II. Postgres + pgvector as the Storage Exception

Vector persistence MUST use self-hosted PostgreSQL with the pgvector
extension, accessed through LlamaIndex's own `PGVectorStore` integration
(`llama-index-vector-stores-postgres`) — never LlamaIndex's in-memory/simple
vector store for anything beyond throwaway tests, and never a third-party
hosted vector database. This is the one deliberate, standing exception to
Principle I's default of using LlamaIndex-native tooling end to end:
LlamaIndex remains the *access layer*, but the persisted data lives in a
self-hosted Postgres instance this project fully controls (schema, extra
indexes, backups, migrations).

Rationale: keeps embedding data local and inspectable, avoids vendor
lock-in to a hosted vector DB, and lets the project manage
performance-critical concerns (e.g. HNSW/IVFFlat indexes) directly in SQL
when the default configuration is not enough.

### III. Config-Driven Behavior — No Hardcoding

All tunable behavior (LLM provider/model, chunking strategy, reranker,
hybrid search, query transformation, PDF parser, Postgres connection
parameters) MUST be exposed as `Settings` fields in `app/config.py`
(pydantic-settings, `.env`-backed) with sensible defaults and validated via
`model_validator` where cross-field constraints apply. New features MUST
extend this pattern instead of introducing ad-hoc environment reads,
hardcoded literals, or parallel configuration mechanisms. Secrets (API
keys, database passwords) MUST NOT be committed to the repository, and
`.env.example` (or `.env.eval.example` for the evaluation subsystem) MUST
be kept in sync with every new setting.

Rationale: a single, typed, validated configuration surface keeps the
provider/parser/retrieval switches (already numerous in this project)
discoverable and safe to combine, instead of scattering environment logic
across modules.

### IV. Test-Backed Changes

Every new or modified RAG behavior (chunking mode, retriever, PDF parser,
API endpoint, config flag) MUST ship with `pytest` coverage under `tests/`.
Changes that affect retrieval or generation quality MUST additionally be
validated against the offline `ragas` evaluation pipeline in `eval/` before
being considered done.

Rationale: RAG correctness regressions (e.g. a silent snippet-truncation
bug that degraded answer quality) are easy to introduce and hard to notice
without both unit-level tests and end-to-end quality metrics catching them.

### V. Design-Before-Code for Non-Trivial Changes

Any change that adds a new dependency, alters the retrieval/indexing
pipeline, or introduces a deviation under Principle I MUST be captured in a
short design doc (following the existing `design.md` pattern) stating: the
requirement, confirmed decisions, dependency/version impact, and the
concrete architecture diff — before implementation begins. Open questions
MUST be resolved with the user rather than silently assumed.

Rationale: this project's history already shows non-trivial
framework/version interactions (e.g. the docling integration); a
lightweight, written design record prevents silent scope creep and keeps
irreversible decisions auditable after the fact.

## Technology Stack Constraints

- Backend: FastAPI on Python 3.12.
- RAG orchestration: LlamaIndex (`llama-index-core` 0.11.x line and matching
  official integration packages) — see Principle I.
- Vector storage: PostgreSQL + the pgvector extension, accessed via
  `llama-index-vector-stores-postgres` — see Principle II.
- Embeddings: local `BAAI/bge-m3` via `llama-index-embeddings-huggingface`;
  the default path MUST NOT depend on a remote embedding API.
- LLM providers: Gemini and DeepSeek, both wired through LlamaIndex-compatible
  wrappers (`OpenAILike`), switchable at runtime via `LLM_PROVIDER`.
- Evaluation: `ragas`, run offline through `eval/` with its own `.env.eval`,
  isolated from the main request path and from the main application's data.

## Development Workflow

- New dependencies added to `requirements.txt` MUST use a pinned,
  compatible-release version range consistent with existing entries, and
  MUST be traceable to the requirement or design doc that introduced them.
- Any environment/config change MUST update `.env.example` (and
  `.env.eval.example` when evaluation-related) plus the corresponding
  section of `README.md` in the same change.
- Before a change is considered complete, the relevant `pytest` subset MUST
  pass, and — for retrieval/generation-affecting changes — an
  `eval/cli run` pass against the checked-in dataset MUST be reviewed for
  regressions.

## Governance

This constitution supersedes ad-hoc conventions for this repository.
Amendments require: (1) a stated rationale for the change, (2) a version
bump following the semantic versioning rules below, and (3) propagation of
the change to any templates or docs that reference the amended principle.
All feature plans and code reviews MUST verify compliance with these
principles; unresolved violations block merge unless explicitly justified
in a plan's Complexity Tracking section.

Versioning policy:
- MAJOR: backward-incompatible principle removal or redefinition that
  invalidates prior compliance assumptions.
- MINOR: a new principle is added, or existing guidance is materially
  expanded.
- PATCH: wording clarifications, typo fixes, or other non-semantic
  refinements.

**Version**: 1.0.0 | **Ratified**: 2026-07-16 | **Last Amended**: 2026-07-16
