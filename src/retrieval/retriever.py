"""
Retriever module
Searches FAISS index for relevant chunks given a query
"""

import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from src.ingestion.embedder import load_index, get_embedding_model


def retrieve_chunks(
    query: str,
    model: SentenceTransformer,
    top_k: int = 5
) -> List[Dict]:
    """
    Retrieve most relevant chunks for a given query.
    
    Args:
        query: Natural language question
        model: SentenceTransformer embedding model
        top_k: Number of chunks to retrieve
        
    Returns:
        List of relevant chunk dictionaries with scores
    """
    # Load index
    index, metadata = load_index()
    
    if index is None or index.ntotal == 0:
        return []
    
    # Embed query with same prefix style
    query_text = f"Represent this NVH document chunk: {query}"
    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True
    ).astype(np.float32)
    
    # Search FAISS
    scores, indices = index.search(query_embedding, top_k)
    
    # Build results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue
        
        chunk = metadata[idx].copy()
        chunk["relevance_score"] = float(score)
        results.append(chunk)
    
    return results