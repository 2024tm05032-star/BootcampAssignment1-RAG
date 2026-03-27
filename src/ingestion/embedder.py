"""
Embedding and Vector Store module
Converts chunks into vectors and stores them in FAISS
"""

import os
import json
import pickle
from pathlib import Path
from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.ingestion.parser import ParsedChunk


# Path where FAISS index and metadata are saved
INDEX_PATH = Path("data/faiss_index")


def get_embedding_model():
    """Load the BGE embedding model"""
    print("Loading embedding model (BGE)...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return model


def embed_chunks(chunks: List[ParsedChunk], model: SentenceTransformer) -> np.ndarray:
    """
    Convert chunks to embedding vectors.
    
    Args:
        chunks: List of ParsedChunk objects
        model: SentenceTransformer model
        
    Returns:
        numpy array of embeddings
    """
    print(f"Embedding {len(chunks)} chunks...")
    
    # Prepare texts for embedding
    # BGE works best with a prefix for retrieval
    texts = [f"Represent this NVH document chunk: {chunk.content}" 
             for chunk in chunks]
    
    # Generate embeddings in batches
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True  # normalize for cosine similarity
    )
    
    return embeddings


def save_index(chunks: List[ParsedChunk], embeddings: np.ndarray):
    """
    Save FAISS index and chunk metadata to disk.
    
    Args:
        chunks: List of ParsedChunk objects
        embeddings: numpy array of embeddings
    """
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings.astype(np.float32))
    
    # Save FAISS index
    faiss.write_index(index, str(INDEX_PATH / "index.faiss"))
    
    # Save chunk metadata as JSON
    metadata = []
    for i, chunk in enumerate(chunks):
        metadata.append({
            "index": i,
            "content": chunk.content,
            "chunk_type": chunk.chunk_type,
            "page_number": chunk.page_number,
            "source_file": chunk.source_file,
            "chunk_index": chunk.chunk_index
        })
    
    with open(INDEX_PATH / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Index saved: {index.ntotal} vectors at {INDEX_PATH}")


def load_index():
    """
    Load FAISS index and metadata from disk.
    
    Returns:
        tuple: (faiss_index, metadata_list) or (None, None) if not found
    """
    index_file = INDEX_PATH / "index.faiss"
    metadata_file = INDEX_PATH / "metadata.json"
    
    if not index_file.exists() or not metadata_file.exists():
        return None, None
    
    index = faiss.read_index(str(index_file))
    
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    return index, metadata


def get_index_stats():
    """Return stats about the current index"""
    index, metadata = load_index()
    
    if index is None:
        return {
            "total_chunks": 0,
            "indexed_documents": [],
            "chunk_type_counts": {}
        }
    
    # Count by type
    type_counts = {}
    documents = set()
    for item in metadata:
        chunk_type = item["chunk_type"]
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        documents.add(item["source_file"])
    
    return {
        "total_chunks": index.ntotal,
        "indexed_documents": list(documents),
        "chunk_type_counts": type_counts
    }