"""
Test PDF parser + image summarisation
Run: python test_parser.py
"""

import os
os.environ["DOCLING_CACHE_DIR"] = ".cache/docling"

from src.ingestion.parser import parse_pdf
from src.models.vision import summarise_all_images

PDF_PATH = "sample_documents/is.3028.1998.pdf"

# Step 1: Parse PDF
chunks = parse_pdf(PDF_PATH)

# Step 2: Summarise only first 2 images (to save API cost during testing)
image_chunks = [c for c in chunks if c.chunk_type == "image"]
test_images = image_chunks[:2]

print(f"\nTesting vision on first 2 images...")
for chunk in test_images:
    image_path = chunk.content.replace("[IMAGE: ", "").replace("]", "")
    
    from src.models.vision import summarise_image
    summary = summarise_image(image_path, chunk.page_number, chunk.source_file)
    print(f"\nPage {chunk.page_number}:")
    print(summary)
    print("-" * 60)