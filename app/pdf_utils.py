"""PDF preprocessing utilities for RAG indexing."""
import shutil
import tempfile
from pathlib import Path


def convert_pdfs_to_markdown_temp(
    upload_dir: Path, paths: list[Path] | None = None
) -> tuple[Path, dict[str, Path]]:
    """Convert PDF files to Markdown via pymupdf4llm.

    By default (paths=None) walks the whole upload_dir, preserving the
    existing full-rebuild behaviour. Pass an explicit `paths` subset (e.g.
    only new/changed files) to scope the conversion for incremental index
    updates instead of re-converting every PDF on disk.

    Non-PDF files are symlinked into the temp dir unchanged so that
    SimpleDirectoryReader can process the full mix in a single pass.

    Returns a tuple of (temp_dir, name_to_original), where name_to_original
    maps each flattened file name inside temp_dir back to its true source
    Path under upload_dir — callers use this to restore correct
    file_path/file_name metadata (and thus stable relpath-based doc ids)
    on Documents loaded from the temp dir. The caller is responsible for
    deleting temp_dir (via shutil.rmtree) after indexing completes.
    """
    import pymupdf4llm  # optional dependency; only imported when PDF_PARSER=pymupdf4llm

    source_paths = paths if paths is not None else [p for p in upload_dir.rglob("*") if p.is_file()]

    temp_dir = Path(tempfile.mkdtemp(prefix="rag_pdf_"))
    name_to_original: dict[str, Path] = {}
    for path in source_paths:
        if path.suffix.lower() == ".pdf":
            md_text = pymupdf4llm.to_markdown(str(path))
            temp_name = path.with_suffix(".md").name
            (temp_dir / temp_name).write_text(md_text, encoding="utf-8")
        else:
            # Symlink non-PDF files so they are visible to SimpleDirectoryReader
            # without copying their (potentially large) content.
            temp_name = path.name
            (temp_dir / temp_name).symlink_to(path.resolve())
        name_to_original[temp_name] = path
    return temp_dir, name_to_original
