"""
FastAPI route definitions
All API endpoints are defined here
"""

import time
import os
import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from src.api.models import (
    HealthResponse, IngestResponse,
    QueryRequest, QueryResponse, SourceReference
)
from src.ingestion.parser import parse_pdf
from src.models.vision import summarise_all_images
from src.ingestion.embedder import (
    get_embedding_model, embed_chunks,
    save_index, load_index, get_index_stats
)
from src.retrieval.retriever import retrieve_chunks
from src.models.llm import generate_answer

router = APIRouter()

# Track server start time for uptime
START_TIME = time.time()

# Load embedding model once at startup (not on every request)
print("Loading embedding model at startup...")
EMBEDDING_MODEL = get_embedding_model()
print("Embedding model ready!")


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Returns system status including model readiness,
    indexed documents, and uptime.
    """
    stats = get_index_stats()
    uptime = time.time() - START_TIME

    return HealthResponse(
        status="ok",
        model_ready=True,
        indexed_documents=stats["indexed_documents"],
        total_chunks=stats["total_chunks"],
        chunk_type_counts=stats["chunk_type_counts"],
        uptime_seconds=round(uptime, 2)
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Accept a PDF upload, parse it, embed chunks,
    and add to the vector index.
    """
    start_time = time.time()

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )

    # Save uploaded file temporarily
    temp_dir = Path("data/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Step 1: Parse PDF
        chunks = parse_pdf(str(temp_path))

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Could not extract any content from PDF"
            )

        # Step 2: Summarise images
        chunks = summarise_all_images(chunks)

        # Step 3: Embed chunks
        embeddings = embed_chunks(chunks, EMBEDDING_MODEL)

        # Step 4: Save to index
        save_index(chunks, embeddings)

        # Count chunk types
        type_counts = {}
        for chunk in chunks:
            type_counts[chunk.chunk_type] = (
                type_counts.get(chunk.chunk_type, 0) + 1
            )

        processing_time = round(time.time() - start_time, 2)

        return IngestResponse(
            message="Document ingested successfully",
            filename=file.filename,
            chunks_added=len(chunks),
            chunk_type_counts=type_counts,
            processing_time_seconds=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Accept a natural language question, retrieve
    relevant chunks, and generate a grounded answer.
    """
    # Check index exists
    index, metadata = load_index()
    if index is None or index.ntotal == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents indexed yet. Please ingest a PDF first."
        )

    # Retrieve relevant chunks
    retrieved = retrieve_chunks(
        query=request.question,
        model=EMBEDDING_MODEL,
        top_k=request.top_k
    )

    if not retrieved:
        raise HTTPException(
            status_code=404,
            detail="No relevant chunks found for this query."
        )

    # Generate answer
    answer = generate_answer(request.question, retrieved)

    # Build source references
    sources = []
    for chunk in retrieved:
        sources.append(SourceReference(
            filename=chunk["source_file"],
            page_number=chunk["page_number"],
            chunk_type=chunk["chunk_type"],
            relevance_score=round(chunk["relevance_score"], 3),
            content_preview=chunk["content"][:200]
        ))

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        chunks_retrieved=len(retrieved)
    )


@router.get("/documents")
def list_documents():
    """
    List all currently indexed documents.
    Bonus endpoint for API completeness.
    """
    stats = get_index_stats()
    return {
        "indexed_documents": stats["indexed_documents"],
        "total_chunks": stats["total_chunks"],
        "chunk_type_counts": stats["chunk_type_counts"]
    }