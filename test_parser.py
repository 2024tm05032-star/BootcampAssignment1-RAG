"""
Quick test for the PDF parser
Run: python test_parser.py
"""

import os
# Tell docling to use local cache
os.environ["DOCLING_CACHE_DIR"] = ".cache/docling"

from src.ingestion.parser import parse_pdf

PDF_PATH = "sample_documents/is.3028.1998.pdf"

chunks = parse_pdf(PDF_PATH)

print("\n--- SAMPLE CHUNKS ---")

# Show one of each type
for chunk_type in ["text", "table", "image"]:
    matches = [c for c in chunks if c.chunk_type == chunk_type]
    if matches:
        c = matches[0]
        print(f"\nType: {c.chunk_type}")
        print(f"Page: {c.page_number}")
        print(f"Content preview: {c.content[:300]}")
        print("-" * 40)
