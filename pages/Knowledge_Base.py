"""Streamlit page for managing the travel PDF knowledge base."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_vector_db import KNOWLEDGE_BASE_DIR, build_vector_db


def _safe_pdf_name(filename: str) -> str:
    """Return a filesystem-safe PDF filename."""
    name = Path(filename).name
    stem = Path(name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return f"{safe_stem or 'uploaded_document'}.pdf"


def _list_uploaded_pdfs() -> list[Path]:
    """Return PDFs currently stored in the knowledge base folder."""
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(KNOWLEDGE_BASE_DIR.glob("*.pdf"))


st.set_page_config(page_title="Knowledge Base", page_icon="KB", layout="wide")

st.title("Knowledge Base")
st.caption("Upload travel guides, brochures, destination documents, and itineraries.")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    saved_count = 0
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    for uploaded_file in uploaded_files:
        filename = _safe_pdf_name(uploaded_file.name)
        destination = KNOWLEDGE_BASE_DIR / filename
        destination.write_bytes(uploaded_file.getbuffer())
        saved_count += 1
    st.success(f"Uploaded {saved_count} PDF file(s) to knowledge_base/.")

st.subheader("Uploaded Files")
pdf_files = _list_uploaded_pdfs()
if pdf_files:
    for pdf_file in pdf_files:
        size_mb = pdf_file.stat().st_size / (1024 * 1024)
        st.write(f"- {pdf_file.name} ({size_mb:.2f} MB)")
else:
    st.info("No PDFs uploaded yet. Add travel documents above, then rebuild the vector database.")

st.divider()

if st.button("Rebuild Vector Database", type="primary"):
    with st.spinner("Building vector database from uploaded PDFs..."):
        try:
            stats = build_vector_db()
            try:
                from agent import reset_rag_cache

                reset_rag_cache()
            except Exception:
                pass
            st.success("Vector database rebuilt successfully.")
            st.metric("Chunks indexed", stats["chunks_indexed"])
            st.write(
                f"Loaded {stats['files_loaded']} of {stats['files_seen']} file(s), "
                f"{stats['pages_loaded']} page(s), "
                f"in {stats['indexing_time_seconds']} seconds."
            )
            if stats["skipped_files"]:
                st.warning("Some PDFs were skipped because they could not be read.")
                for skipped in stats["skipped_files"]:
                    st.write(f"- {skipped['file']}: {skipped['error']}")
        except FileNotFoundError as exc:
            st.error(str(exc))
        except RuntimeError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Vector database rebuild failed: {exc}")
