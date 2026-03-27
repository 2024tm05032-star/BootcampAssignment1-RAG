"""
LLM module
Generates answers from retrieved chunks using GPT-4o-mini
"""

import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def generate_answer(query: str, retrieved_chunks: List[Dict]) -> str:
    """
    Generate an answer using retrieved chunks as context.
    
    Args:
        query: User's natural language question
        retrieved_chunks: List of relevant chunks from retriever
        
    Returns:
        Generated answer string
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[Source {i+1}] "
            f"Type: {chunk['chunk_type'].upper()} | "
            f"File: {chunk['source_file']} | "
            f"Page: {chunk['page_number']}\n"
            f"{chunk['content']}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    # NVH-specific system prompt
    system_prompt = """You are an expert NVH (Noise, Vibration and Harshness) compliance engineer 
with deep knowledge of automotive noise standards including IS 3028, ECE R51, ISO 362, and AVAS regulations.

Your role is to answer questions about NVH compliance documents accurately and technically.

Rules:
1. Answer ONLY based on the provided context chunks
2. If the context does not contain enough information, say so clearly
3. Always cite your sources using [Source N] references
4. For limit values, always include units (dB, dB(A), etc.)
5. For test conditions, be precise about vehicle configurations
6. If a table is relevant, explain what the rows/columns mean
7. Never make up values or standards that are not in the context"""

    user_prompt = f"""Based on the following excerpts from NVH compliance documents, 
answer this question: {query}

CONTEXT:
{context}

Provide a clear, technical answer with source references."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.1  # low temperature for factual accuracy
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"