"""
NVH Compliance RAG System
Main FastAPI application entry point
"""

import os
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
import uvicorn

app = FastAPI(
    title="NVH Compliance RAG System",
    description="Multimodal RAG system for automotive NVH compliance documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


def auto_ingest_sample_documents():
    """
    Automatically ingest all PDFs in sample_documents/
    if the FAISS index is empty or missing.
    Runs once at startup.
    """
    from src.ingestion.embedder import get_index_stats, get_embedding_model, embed_chunks, save_index
    from src.ingestion.parser import parse_pdf
    from src.models.vision import summarise_all_images

    stats = get_index_stats()

    if stats["total_chunks"] > 0:
        print(f"Index already has {stats['total_chunks']} chunks — skipping auto-ingest")
        return

    sample_dir = Path("sample_documents")
    pdf_files = list(sample_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in sample_documents/ — skipping auto-ingest")
        return

    print(f"Index is empty — auto-ingesting {len(pdf_files)} PDF(s)...")
    model = get_embedding_model()

    for pdf_path in pdf_files:
        try:
            print(f"  Ingesting: {pdf_path.name}")
            chunks = parse_pdf(str(pdf_path))
            chunks = summarise_all_images(chunks)
            embeddings = embed_chunks(chunks, model)
            save_index(chunks, embeddings)
            print(f"  Done: {len(chunks)} chunks from {pdf_path.name}")
        except Exception as e:
            print(f"  Failed to ingest {pdf_path.name}: {e}")

    final_stats = get_index_stats()
    print(f"Auto-ingest complete: {final_stats['total_chunks']} total chunks indexed")


@app.on_event("startup")
async def startup_event():
    """Run auto-ingest when server starts"""
    auto_ingest_sample_documents()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)