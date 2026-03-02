"""
PyMuPDF-based PDF reader that extracts structured text with section headings.

Unlike pypdf which produces flat text with no font metadata, this reader uses
PyMuPDF (fitz) to detect headings via font-size and bold heuristics, producing
Markdown-style heading markers (## / ### / ####) that Ragas HeadlineSplitter
can locate via text.find().

Usage:
    from pymupdf_reader import load_pdfs_with_headings
    # Returns list of LangChain Document objects (one per PDF, full text)
    docs = load_pdfs_with_headings("./prompt-engineering-papers")
"""

import os
import re
import statistics
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # pymupdf


def _extract_block_text(block: dict) -> Tuple[str, int, int, float]:
    """
    Join spans within a block, preserving whitespace between spans
    and performing line-end dehyphenation.

    Returns (text, total_chars, bold_chars, max_font_size).
    """
    if "lines" not in block:
        return "", 0, 0, 0.0

    line_texts: list[str] = []
    total_chars = 0
    bold_chars = 0
    max_size = 0.0

    for line in block["lines"]:
        parts: list[str] = []
        prev_right_x: Optional[float] = None
        for span in line["spans"]:
            text = span["text"]
            origin_x = span["origin"][0]
            # Insert space if gap between consecutive spans
            if prev_right_x is not None and origin_x > prev_right_x + 1:
                parts.append(" ")
            parts.append(text)
            prev_right_x = span["bbox"][2]

            stripped = text.strip()
            if stripped:
                total_chars += len(stripped)
                if span["flags"] & (1 << 4):  # bold flag
                    bold_chars += len(stripped)
                max_size = max(max_size, round(span["size"], 1))

        line_texts.append("".join(parts).strip())

    # Join lines with dehyphenation
    result = ""
    for lt in line_texts:
        if not lt:
            continue
        if result and result.endswith("-"):
            # Dehyphenate: remove trailing hyphen and join directly
            result = result[:-1] + lt
        elif result:
            result += " " + lt
        else:
            result = lt

    return result.strip(), total_chars, bold_chars, max_size


def _classify_block(
    block: dict, body_size: float
) -> Tuple[str, bool, int]:
    """
    Classify a text block as heading or body text.

    Returns (text, is_heading, heading_level).
    Heading levels:
        1 = paper title (font >> body)
        2 = section heading (font > body, bold, short)
        3 = subsection heading (body size, all bold, short)
    """
    text, total_chars, bold_chars, max_size = _extract_block_text(block)
    if not text or total_chars < 2:
        return text, False, 0

    bold_ratio = bold_chars / total_chars if total_chars > 0 else 0

    # Filter out arXiv identifiers, page numbers, etc.
    if re.match(r"arXiv:", text):
        return text, False, 0

    # Title: font size notably larger than body AND mostly bold
    if max_size > body_size + 2 and bold_ratio > 0.5:
        return text, True, 1

    # Section heading: font larger than body, mostly bold, not too long
    if max_size > body_size and bold_ratio > 0.5 and total_chars < 100:
        return text, True, 2

    # Subsection heading: body-size font, entirely bold, short text
    if bold_ratio > 0.8 and total_chars < 80 and max_size >= body_size:
        return text, True, 3

    return text, False, 0


def extract_pdf_with_headings(pdf_path: str) -> Tuple[str, List[str]]:
    """
    Extract full text from a PDF with Markdown heading markers.

    Returns:
        (full_text, list_of_headings)
    where full_text contains '## Title', '### Section', '#### Subsection'
    markers, and list_of_headings contains the heading text strings.
    """
    doc = fitz.open(pdf_path)

    # Pass 1: determine body text size (statistical mode of all spans)
    all_sizes: list[float] = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip() and len(span["text"].strip()) > 2:
                        all_sizes.append(round(span["size"], 1))

    if not all_sizes:
        doc.close()
        return "", []

    body_size = statistics.mode(all_sizes)

    # Pass 2: extract text with heading classification
    result_parts: list[str] = []
    headings: list[str] = []

    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            text, is_heading, level = _classify_block(block, body_size)
            if not text:
                continue
            if is_heading:
                prefix = "#" * (level + 1)  # ## title, ### section, #### subsection
                result_parts.append(f"\n{prefix} {text}\n")
                headings.append(text)
            else:
                result_parts.append(text)

    doc.close()

    full_text = "\n".join(result_parts)
    # Clean up excessive whitespace
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    return full_text, headings


def load_pdfs_with_headings(
    directory: str,
    glob_pattern: str = "*.pdf",
    max_files: int | None = None,
) -> list:
    """
    Load all PDFs from a directory, extracting structured text with headings.

    Returns a list of LangChain Document objects, one per PDF file.
    Each document's page_content has Markdown heading markers that
    Ragas HeadlineSplitter can locate via text.find().

    Metadata includes:
        - file_name: PDF filename
        - file_path: full path
        - headings: list of extracted heading strings
    """
    from langchain_core.documents import Document as LCDocument

    pdf_dir = Path(directory)
    pdf_files = sorted(pdf_dir.glob(glob_pattern))
    if max_files is not None:
        pdf_files = pdf_files[:max_files]

    documents: list = []
    for pdf_path in pdf_files:
        full_text, headings = extract_pdf_with_headings(str(pdf_path))
        if not full_text.strip():
            continue
        doc = LCDocument(
            page_content=full_text,
            metadata={
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "headings": headings,
            },
        )
        documents.append(doc)

    return documents


def load_pdfs_as_llamaindex_docs(
    directory: str,
    glob_pattern: str = "*.pdf",
    max_files: int | None = None,
) -> list:
    """
    Load PDFs with PyMuPDF heading extraction, returning LlamaIndex Document objects.

    This replaces LlamaIndex's SimpleDirectoryReader (which uses pypdf and
    loses all font/heading information).  Each PDF becomes ONE LlamaIndex
    Document whose .text contains Markdown heading markers.

    Using these docs for both the VectorStoreIndex AND Ragas testset generation
    ensures HeadlineSplitter always sees the headings.
    """
    from llama_index.core import Document as LIDocument

    lc_docs = load_pdfs_with_headings(directory, glob_pattern, max_files)
    li_docs = []
    for lc_doc in lc_docs:
        li_doc = LIDocument(
            text=lc_doc.page_content,
            metadata={
                "file_name": lc_doc.metadata["file_name"],
                "file_path": lc_doc.metadata["file_path"],
            },
        )
        li_docs.append(li_doc)
    return li_docs


if __name__ == "__main__":
    import sys

    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    docs = load_pdfs_with_headings(directory)
    for doc in docs:
        print(f"\n{'='*60}")
        print(f"File: {doc.metadata['file_name']}")
        print(f"Headings ({len(doc.metadata['headings'])}):")
        for h in doc.metadata["headings"]:
            print(f"  - {h}")
        print(f"Text length: {len(doc.page_content)} chars")
        print(f"Preview: {doc.page_content[:200]}...")
