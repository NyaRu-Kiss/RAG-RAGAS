"""PDF preprocessing utilities for RAG indexing."""
import shutil
import tempfile
from pathlib import Path


def convert_pdfs_to_markdown_temp(upload_dir: Path) -> Path:
    """Convert PDF files in upload_dir to Markdown via pymupdf4llm.

    Non-PDF files are symlinked into the temp dir unchanged so that
    SimpleDirectoryReader can process the full mix in a single pass.

    Returns the path to the temporary directory.  The caller is responsible
    for deleting it (via shutil.rmtree) after indexing completes.
    """
    import pymupdf4llm  # optional dependency; only imported when PDF_PARSER=pymupdf4llm

    temp_dir = Path(tempfile.mkdtemp(prefix="rag_pdf_"))
    for path in upload_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            md_text = pymupdf4llm.to_markdown(str(path))
            (temp_dir / path.with_suffix(".md").name).write_text(md_text, encoding="utf-8")
        else:
            # Symlink non-PDF files so they are visible to SimpleDirectoryReader
            # without copying their (potentially large) content.
            (temp_dir / path.name).symlink_to(path.resolve())
    return temp_dir
