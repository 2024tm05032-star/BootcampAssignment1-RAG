🚗 Automobile Owner’s Manual RAG Assistant

A Multimodal Retrieval-Augmented Generation (RAG) system designed to help users interact with automobile owner’s manuals intelligently. This system enables natural language querying over complex vehicle manuals, extracting information from text, tables, and diagrams.

📌 Problem Statement

Domain: Automotive — Owner’s Manuals & User Assistance

Background

Modern vehicles come with extensive owner’s manuals containing hundreds of pages of information covering:

Dashboard warning symbols
Maintenance procedures
Feature usage instructions
Safety guidelines
Troubleshooting steps

These manuals are often:

Dense and hard to navigate
Filled with technical jargon
Poorly searchable using traditional keyword search
The Problem

Vehicle owners frequently need quick answers such as:

“What does this warning light mean?”
“How do I reset the service indicator?”
“What is the recommended tire pressure?”
“How do I connect Bluetooth?”

Currently, users must:

Manually browse PDFs
Use Ctrl+F (which fails on scanned pages/images)
Rely on external sources or mechanics

This process is inefficient and frustrating.

Why This Problem Is Challenging

Owner’s manuals are inherently multimodal:

Text: Instructions and explanations
Tables: Specifications (e.g., oil capacity, tire pressure)
Images: Dashboard icons, diagrams, control layouts

Challenges include:

Information spread across multiple sections
Visual elements not searchable via text
Context-dependent answers (vehicle variant, feature set)
Why RAG Is the Right Approach

A Retrieval-Augmented Generation system solves this effectively:

✅ Retrieves relevant content directly from manuals
✅ Works without retraining when new manuals are added
✅ Handles multimodal data (text + tables + images)
✅ Provides grounded answers with source references
Expected Outcomes

Users can ask:

“What does the engine warning light indicate?”
“How to adjust seat height?”
“What is the battery specification?”

And receive:

✅ Accurate answers
✅ Extracted from actual manual
✅ With source references (page, section)
🏗️ Architecture Overview
Ingestion Pipeline
PDF Manual
    → PyMuPDF Parser
        → Text Chunks
        → Table Chunks
        → Images (symbols, diagrams)
            → Vision Model → Image Summaries
    → Embedding Model (BGE)
    → FAISS Vector Store
Query Pipeline
User Query
    → Embedding Model
    → FAISS Retrieval (Top-K)
    → Context Builder
    → LLM (GPT-4o-mini)
    → Answer + Sources
⚙️ Technology Stack
Component	Choice	Reason
Document Parsing	PyMuPDF	Efficient PDF parsing & table extraction
Embeddings	BGE-small-en-v1.5	Strong performance on technical text
Vector Store	FAISS	Fast in-memory similarity search
LLM	GPT-4o-mini	Accurate, cost-effective responses
Vision Model	GPT-4o-mini	Handles diagrams & symbols
Backend	FastAPI	Lightweight API framework
🚀 Setup Instructions
Prerequisites
Python 3.10+
OpenAI API key
Git
1. Clone Repository
git clone https://github.com/YOUR_USERNAME/automobile-owners-manual-rag.git
cd automobile-owners-manual-rag
2. Install Dependencies
bash setup.sh
3. Configure API Key
cp .env.example .env

Add your OpenAI API key in .env

4. Build Index (First Time)
python -c "
from src.ingestion.parser import parse_pdf
from src.models.vision import summarise_all_images
from src.ingestion.embedder import get_embedding_model, embed_chunks, save_index

chunks = parse_pdf('sample_documents/car_manual.pdf')
chunks = summarise_all_images(chunks)

model = get_embedding_model()
embeddings = embed_chunks(chunks, model)
save_index(chunks, embeddings)

print('Index ready!')
"
5. Run Server
python main.py
6. Access API

Open:

http://localhost:8000/docs
📡 API Endpoints
GET /health

Check system status

POST /ingest

Upload a car manual PDF

POST /query

Ask questions like:

{
  "question": "What does the battery warning light mean?",
  "top_k": 5
}
GET /documents

List uploaded manuals

📸 Features
🔍 Natural language search over manuals
📊 Table-aware querying (specifications)
🖼️ Image understanding (symbols & diagrams)
📖 Source-cited answers
⚡ Fast retrieval with FAISS

⚠️ Limitations
Works best with well-structured PDFs
May extract non-relevant images (logos/icons)
No multi-vehicle comparison yet
Index rebuild required for new documents

🔮 Future Improvements
Multi-language support (Hindi, regional languages)
Voice-based assistant
Mobile/web UI integration
Hybrid search (keyword + semantic)
Vehicle-specific personalization

📊 Evaluation
High factual accuracy (grounded responses)
Strong performance on procedural queries
Reliable extraction from tables and diagrams

🏁 Conclusion
This system transforms static automobile manuals into an interactive intelligent assistant, significantly improving user experience and reducing time spent searching for information.
