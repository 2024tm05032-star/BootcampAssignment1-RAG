"""
PDF Parser module
Extracts text, tables, and images from PDF documents using PyMuPDF
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List
import fitz  # PyMuPDF
import pymupdf4llm


@dataclass
class ParsedChunk:
    """Represents a single extracted chunk from a PDF"""
    content: str
    chunk_type: str     # "text", "table", or "image"
    page_number: int
    source_file: str
    chunk_index: int


def parse_pdf(pdf_path: str) -> List[ParsedChunk]:
    """
    Parse a PDF file and extract text, tables, and images as chunks.

    Args:
        pdf_path: Full path to the PDF file

    Returns:
        List of ParsedChunk objects
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Parsing PDF: {pdf_path.name}")

    chunks = []
    chunk_index = 0
    filename = pdf_path.name

    # Open PDF
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    print(f"Total pages: {total_pages}")

    # --- Extract TEXT chunks using pymupdf4llm ---
    print("Extracting text chunks...")
    md_text = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    for page_data in md_text:
        page_num = page_data["metadata"]["page"] + 1
        content = page_data["text"].strip()

        if len(content) < 30:
            continue

        # Split large pages into smaller chunks
        paragraphs = content.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) < 30:
                continue

            chunks.append(ParsedChunk(
                content=para,
                chunk_type="text",
                page_number=page_num,
                source_file=filename,
                chunk_index=chunk_index
            ))
            chunk_index += 1

    # --- Extract TABLE chunks ---
    print("Extracting table chunks...")
    for page_num in range(total_pages):
        page = doc[page_num]
        tables = page.find_tables()

        for table in tables:
            try:
                # Convert table to markdown
                df = table.to_pandas()
                table_md = df.to_markdown(index=False)

                if len(table_md.strip()) < 10:
                    continue

                chunks.append(ParsedChunk(
                    content=table_md,
                    chunk_type="table",
                    page_number=page_num + 1,
                    source_file=filename,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
            except Exception as e:
                print(f"  Could not extract table on page {page_num + 1}: {e}")

    # --- Extract IMAGE chunks ---
    print("Extracting image chunks...")
    image_dir = Path("data/images") / pdf_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)

    for page_num in range(total_pages):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Save image
                image_path = image_dir / f"page{page_num + 1}_img{img_index}.{image_ext}"
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                # Skip tiny images (logos, decorations)
                if len(image_bytes) < 5000:
                    continue

                chunks.append(ParsedChunk(
                    content=f"[IMAGE: {image_path}]",
                    chunk_type="image",
                    page_number=page_num + 1,
                    source_file=filename,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
            except Exception as e:
                print(f"  Could not extract image on page {page_num + 1}: {e}")

    doc.close()

    print(f"\nParsing complete: {len(chunks)} chunks extracted")
    print(f"  Text:   {sum(1 for c in chunks if c.chunk_type == 'text')}")
    print(f"  Tables: {sum(1 for c in chunks if c.chunk_type == 'table')}")
    print(f"  Images: {sum(1 for c in chunks if c.chunk_type == 'image')}")

    return chunks