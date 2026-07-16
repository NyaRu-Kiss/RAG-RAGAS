# Code Review — DOCLING-01

Scope: `app/config.py`, `app/rag.py`, `app/main.py`, `requirements.txt`, `.env.example`, `tests/test_rag_docling_parser.py`

## Findings

No critical issues.

1. **Metadata correctness (verified against installed package source, not just docs)**
   - `DoclingReader.lazy_load_data()` reuses one `extra_info` dict across all yielded `Document`s when given a list of paths in a single call — would have silently collapsed every PDF/Office file's `file_name` to the same value in a multi-file rebuild. Fixed by calling it once per path with a distinct `extra_info`.
   - `DoclingNodeParser` replaces `node.metadata` entirely with docling's own chunk metadata, dropping `file_name`/`file_path`. Fixed by restoring them from `node.relationships[NodeRelationship.SOURCE].metadata` via `setdefault` (so docling's own keys, e.g. `heading`, are never clobbered).

2. **Import cost**: `DoclingReader`/`DoclingNodeParser`/`NodeRelationship` imports are scoped inside `if docling_paths:` in `_build_docling_nodes()`, not module-level — an empty upload dir or a docling-free batch never pays the import cost of the (heavy, optional) docling stack. Covered by `test_empty_dir_returns_no_nodes_without_importing_docling`, which deliberately does not mock `sys.modules` and still passes.

3. **Shared dispatch helper**: `_parse_documents_to_nodes()` is used by all three `PDF_PARSER` branches (`default`, `pymupdf4llm`, and the non-docling fallback inside the `docling` branch), so `CHUNK_MODE=layout_aware` behaves identically regardless of which PDF parser is active — matches design.md's intent that `CHUNK_MODE` is orthogonal to `PDF_PARSER`.

4. **Naming/style**: new methods follow the existing `_build_*`/`_parse_*` private-method convention already used in `RagService` (`_build_node_parser`, `_build_reranker`, `_build_llamaindex_llm`). `Literal[...]` + alias `Settings` fields follow the existing `config.py` pattern exactly, including the `# ---` comment-block header style.

5. **Error handling**: no new error-handling paths were introduced beyond what `rebuild_index()` already had — a docling conversion failure or a malformed PDF propagates as an exception the same way a `SimpleDirectoryReader` load failure already did, consistent with existing behavior (caught and turned into a 500 by the FastAPI route in `main.py`, unchanged).

6. **Security**: no new user-controlled input is interpreted as a path, command, or template beyond what already existed (`upload_dir` is server-configured, not request-supplied); no new shell/eval/format-string sinks introduced.

## Recommendations (non-blocking)

- `app/main.py`, and `RagService.__init__`/`_vector_store`/`_storage_context`/`_ensure_index`/`_build_reranker`/`_build_llamaindex_llm` (lines 46–152 of `app/rag.py`) have **zero test coverage** — this predates DOCLING-01 (confirmed: no `rebuild_index` test existed before this change either) and would need a FastAPI `TestClient` + mocked Postgres/embedding stack to close. Out of scope for this design doc; flagged for a future task rather than silently left unmentioned.
- Consider a follow-up `CHANGELOG`/README note documenting the new `PDF_PARSER=docling` and `CHUNK_MODE=layout_aware` env vars for end users, beyond the inline `.env.example` comments already added.

## Verdict

✓ No critical issues. Matches design.md. Approved.
