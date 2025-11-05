# 🐍 Use lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching efficiency
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project into container
COPY . .

# Create required directories
RUN mkdir -p chromadb_store upload_files data

# Expose FastAPI port
EXPOSE 8000

# Environment variables (optional defaults)
ENV PYTHONUNBUFFERED=1
ENV UVICORN_WORKERS=2

# Start the multi-provider API by default
CMD ["uvicorn", "api.multi_provider_rag_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
