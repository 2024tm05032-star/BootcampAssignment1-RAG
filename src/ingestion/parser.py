"""
PDF Parser module
Extracts text, tables, and images from PDF documents using Docling
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat


@dataclass
class ParsedChunk:
    """Represents a single extracted chunk from a PDF"""
    content: str        # text content
    chunk_type: str     # "text", "table", or "image"
    page_number: int    # page where this chunk was found
    source_file: str    # original PDF filename
    chunk_index: int    # position in document


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
    
    # Configure Docling pipeline
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False          # skip OCR for now
    pipeline_options.do_table_structure = True  # extract tables
    pipeline_options.images_scale = 2.0      # image resolution
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True  # extract figures
    
    # Create converter
    converter = DocumentConverter()
    
    # Convert PDF
    print("Running Docling conversion...")
    result = converter.convert(str(pdf_path))
    doc = result.document
    
    chunks = []
    chunk_index = 0
    filename = pdf_path.name
    
    # --- Extract TEXT chunks ---
    print("Extracting text chunks...")
    for text_item in doc.texts:
        content = text_item.text.strip()
        if len(content) < 30:  # skip very short fragments
            continue
        
        page_num = 1
        if text_item.prov:
            page_num = text_item.prov[0].page_no
            
        chunks.append(ParsedChunk(
            content=content,
            chunk_type="text",
            page_number=page_num,
            source_file=filename,
            chunk_index=chunk_index
        ))
        chunk_index += 1
    
    # --- Extract TABLE chunks ---
    print("Extracting table chunks...")
    for table_item in doc.tables:
        # Convert table to markdown format
        try:
            table_md = table_item.export_to_markdown()
        except Exception:
            table_md = str(table_item)
            
        if len(table_md.strip()) < 10:
            continue
        
        page_num = 1
        if table_item.prov:
            page_num = table_item.prov[0].page_no
            
        chunks.append(ParsedChunk(
            content=table_md,
            chunk_type="table",
            page_number=page_num,
            source_file=filename,
            chunk_index=chunk_index
        ))
        chunk_index += 1
    
    # --- Extract IMAGE chunks ---
    print("Extracting image chunks...")
    image_dir = Path("data/images") / pdf_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)
    
    for pic_index, picture_item in enumerate(doc.pictures):
        page_num = 1
        if picture_item.prov:
            page_num = picture_item.prov[0].page_no
        
        # Save image to disk
        image_path = image_dir / f"page{page_num}_img{pic_index}.png"
        
        try:
            with open(image_path, "wb") as f:
                picture_item.image.pil_image.save(f, format="PNG")
            
            # Store image path as content for now
            # Phase 5 will replace this with VLM summary
            chunks.append(ParsedChunk(
                content=f"[IMAGE: {image_path}]",
                chunk_type="image",
                page_number=page_num,
                source_file=filename,
                chunk_index=chunk_index
            ))
            chunk_index += 1
        except Exception as e:
            print(f"  Could not save image {pic_index}: {e}")
    
    print(f"Parsing complete: {len(chunks)} chunks extracted")
    print(f"  Text: {sum(1 for c in chunks if c.chunk_type == 'text')}")
    print(f"  Tables: {sum(1 for c in chunks if c.chunk_type == 'table')}")
    print(f"  Images: {sum(1 for c in chunks if c.chunk_type == 'image')}")
    
    return chunks