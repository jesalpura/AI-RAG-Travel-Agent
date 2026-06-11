"""
Build a FAISS vector database from PDF files in knowledge_base/.

This module is both a standalone script and an importable utility for Streamlit.
Run directly with:

    python build_vector_db.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from db import db


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def _list_pdf_files() -> list[Path]:
    """Return all PDF files currently stored in the knowledge base folder."""
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(path for path in KNOWLEDGE_BASE_DIR.glob("*.pdf") if path.is_file())


def _load_pdf_pages(pdf_path: Path) -> list[Any]:
    """Load one PDF and attach a clean source filename to each page."""
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    for page in pages:
        page.metadata["source"] = pdf_path.name
        page.metadata["source_path"] = str(pdf_path)
    return pages


def build_vector_db() -> dict[str, Any]:
    """
    Build and persist a FAISS vector database from all PDFs in knowledge_base/.

    Corrupt or unreadable PDFs are skipped and reported in the returned stats.
    Raises:
        FileNotFoundError: if knowledge_base/ has no PDF files.
        RuntimeError: if embeddings or FAISS indexing fails.
    """
    start_time = time.perf_counter()
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = _list_pdf_files()
    db.log_tool_call(
        "Vector DB Builder",
        "build_vector_db",
        f"Vector build started for {len(pdf_files)} PDF file(s).",
    )

    if not pdf_files:
        message = "No PDF files found in knowledge_base/. Upload PDFs before rebuilding."
        db.log_error(message, tool_name="Vector DB Builder")
        raise FileNotFoundError(message)

    all_pages: list[Any] = []
    skipped_files: list[dict[str, str]] = []

    for pdf_file in pdf_files:
        try:
            all_pages.extend(_load_pdf_pages(pdf_file))
        except Exception as exc:
            skipped_files.append({"file": pdf_file.name, "error": str(exc)})
            db.log_error(
                f"Skipped corrupt or unreadable PDF: {pdf_file.name}",
                tool_name="Vector DB Builder",
                exc=exc,
            )

    if not all_pages:
        message = "No readable PDF pages found. Check that uploaded PDFs are valid."
        db.log_error(message, tool_name="Vector DB Builder")
        raise FileNotFoundError(message)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(all_pages)

    if not chunks:
        message = "PDFs were loaded, but no text chunks could be created."
        db.log_error(message, tool_name="Vector DB Builder")
        raise RuntimeError(message)

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as exc:
        message = (
            "Could not load or download the HuggingFace embedding model "
            f"'{EMBEDDING_MODEL_NAME}'. Check your internet connection and package setup."
        )
        db.log_error(message, tool_name="Vector DB Builder", exc=exc)
        raise RuntimeError(message) from exc

    try:
        vector_db = FAISS.from_documents(chunks, embeddings)
        vector_db.save_local(str(FAISS_INDEX_DIR))
    except Exception as exc:
        message = "FAISS indexing failed. Check the PDF text and FAISS installation."
        db.log_error(message, tool_name="Vector DB Builder", exc=exc)
        raise RuntimeError(message) from exc

    elapsed_seconds = time.perf_counter() - start_time
    stats: dict[str, Any] = {
        "files_loaded": len(pdf_files) - len(skipped_files),
        "files_seen": len(pdf_files),
        "pages_loaded": len(all_pages),
        "chunks_indexed": len(chunks),
        "indexing_time_seconds": round(elapsed_seconds, 2),
        "skipped_files": skipped_files,
        "index_path": str(FAISS_INDEX_DIR),
    }

    db.log_tool_call(
        "Vector DB Builder",
        "build_vector_db",
        (
            "Vector build completed: "
            f"{stats['files_loaded']} file(s), "
            f"{stats['pages_loaded']} page(s), "
            f"{stats['chunks_indexed']} chunk(s)."
        ),
        duration_ms=int(elapsed_seconds * 1000),
    )
    return stats


def _print_stats(stats: dict[str, Any]) -> None:
    """Print indexing stats for command-line usage."""
    print("FAISS vector database built successfully.")
    print(f"Files loaded: {stats['files_loaded']} of {stats['files_seen']}")
    print(f"Pages loaded: {stats['pages_loaded']}")
    print(f"Chunks indexed: {stats['chunks_indexed']}")
    print(f"Indexing time: {stats['indexing_time_seconds']} seconds")
    if stats["skipped_files"]:
        print("Skipped files:")
        for skipped in stats["skipped_files"]:
            print(f"- {skipped['file']}: {skipped['error']}")


if __name__ == "__main__":
    _print_stats(build_vector_db())
