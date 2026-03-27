"""
Test full pipeline: parse → embed → retrieve
Run: python test_parser.py
"""

import os
os.environ["DOCLING_CACHE_DIR"] = ".cache/docling"

from src.ingestion.parser import parse_pdf
from src.models.vision import summarise_all_images
from src.ingestion.embedder import get_embedding_model, embed_chunks, save_index, get_index_stats
from src.retrieval.retriever import retrieve_chunks

PDF_PATH = "sample_documents/is.3028.1998.pdf"

# Step 1: Parse PDF
print("=" * 50)
print("STEP 1: Parsing PDF")
print("=" * 50)
chunks = parse_pdf(PDF_PATH)

# Step 2: Summarise images
print("\n" + "=" * 50)
print("STEP 2: Summarising images")
print("=" * 50)
chunks = summarise_all_images(chunks)

# Step 3: Embed all chunks
print("\n" + "=" * 50)
print("STEP 3: Embedding chunks")
print("=" * 50)
model = get_embedding_model()
embeddings = embed_chunks(chunks, model)

# Step 4: Save to FAISS
print("\n" + "=" * 50)
print("STEP 4: Saving to FAISS index")
print("=" * 50)
save_index(chunks, embeddings)

# Step 5: Check stats
stats = get_index_stats()
print(f"\nIndex stats: {stats}")

# Step 6: Test retrieval
print("\n" + "=" * 50)
print("STEP 5: Testing retrieval")
print("=" * 50)

test_queries = [
    "What is the noise limit for passenger vehicles?",
    "What are the test conditions for hybrid vehicles?",
    "What does Table 1 say about load schedule?"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    results = retrieve_chunks(query, model, top_k=2)
    for r in results:
        print(f"  [{r['chunk_type'].upper()}] Page {r['page_number']} "
              f"(score: {r['relevance_score']:.3f})")
        print(f"  {r['content'][:150]}...")
    print("-" * 40)