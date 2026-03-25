"""
Vision Language Model module
Sends extracted images to GPT-4o-mini for text summarisation
"""

import base64
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string for API transmission"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def summarise_image(image_path: str, page_number: int, source_file: str) -> str:
    """
    Send an image to GPT-4o-mini and get a text description.
    
    Args:
        image_path: Path to the image file
        page_number: Page number where image was found
        source_file: Original PDF filename
        
    Returns:
        Text description of the image
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Check image exists
    if not Path(image_path).exists():
        return f"[Image not found: {image_path}]"
    
    # Encode image
    image_data = encode_image_to_base64(image_path)
    
    # NVH-specific prompt for better descriptions
    prompt = """You are analysing a figure from an automotive NVH (Noise, Vibration and Harshness) 
compliance standard document (IS 3028 - Indian Standard for vehicle noise measurement).

Describe this figure in detail focusing on:
1. What type of figure is this? (test track layout, measurement setup, graph, diagram, table, etc.)
2. What are the key values, measurements, or dimensions shown?
3. What vehicle configurations or test conditions does it relate to?
4. What is the main technical information an NVH engineer would extract from this figure?

Be specific and technical. Include any numbers, labels, or annotations visible in the figure."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        summary = response.choices[0].message.content
        return f"[Figure on page {page_number} from {source_file}]: {summary}"
        
    except Exception as e:
        print(f"  Vision API error for {image_path}: {e}")
        return f"[Figure on page {page_number}: Could not process image - {str(e)}]"


def summarise_all_images(chunks: list) -> list:
    """
    Process all image chunks and replace path placeholders with VLM summaries.
    
    Args:
        chunks: List of ParsedChunk objects
        
    Returns:
        Updated list with image chunks containing text summaries
    """
    image_chunks = [c for c in chunks if c.chunk_type == "image"]
    total = len(image_chunks)
    
    print(f"\nSummarising {total} images with GPT-4o-mini...")
    
    for i, chunk in enumerate(image_chunks):
        # Extract image path from content string "[IMAGE: path/to/image.png]"
        image_path = chunk.content.replace("[IMAGE: ", "").replace("]", "")
        
        print(f"  Processing image {i+1}/{total}: {image_path}")
        
        summary = summarise_image(
            image_path=image_path,
            page_number=chunk.page_number,
            source_file=chunk.source_file
        )
        
        # Replace path with actual text summary
        chunk.content = summary
        print(f"  ✓ Done")
    
    print(f"Image summarisation complete!")
    return chunks