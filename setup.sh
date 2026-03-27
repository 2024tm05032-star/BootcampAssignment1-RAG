#!/bin/bash
# Run this once after every Codespaces restart
# Usage: bash setup.sh

echo "Installing dependencies..."
pip install pymupdf pymupdf4llm
pip install openai==1.35.0 httpx==0.27.0
pip install faiss-cpu sentence-transformers
pip install fastapi==0.111.0 uvicorn==0.30.1 python-multipart==0.0.9
pip install langchain==0.2.6 langchain-openai==0.1.14 langchain-community==0.2.6
pip install pydantic==2.7.4 python-dotenv==1.0.1 pillow==10.3.0 pandas==2.2.2 tabulate
pip install "numpy<2.0.0" --force-reinstall

echo ""
echo "Setting up .env file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env — add your OpenAI API key!"
else
    echo ".env already exists"
fi

echo ""
echo "Done! Remember to add your API key to .env"