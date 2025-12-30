# RAG Knowledge Base API

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, ChromaDB, and AWS Bedrock. Upload documents in multiple formats, process them with AI-powered extraction, and query your knowledge base using advanced hybrid search with vision-enabled responses.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🌟 Features

### Document Processing
- **📄 Multi-Format Support**: PDF, DOCX, PPTX, TXT, Markdown, Images (PNG, JPG, etc.)
- **🖼️ AI-Powered Image Analysis**: Automatic description of images, charts, and figures
- **📊 Table Extraction**: Intelligent table structure inference and extraction
- **🔤 OCR Support**: Extract text from scanned documents and images
- **📑 Page-Aware Chunking**: Maintains document structure and page references

### Search & Retrieval
- **🔍 Hybrid Search**: Combines BM25 keyword search with semantic vector search
- **⚡ Configurable Weights**: Adjust semantic vs. keyword search balance
- **🎯 Context-Aware**: Returns relevant chunks with proper citations
- **👁️ Vision-Enabled Queries**: LLM can analyze images directly when answering questions
- **🔄 Conversation History**: Multi-turn conversations with context preservation

### Architecture
- **🏗️ Clean Architecture**: Modular design with clear separation of concerns
- **🔐 User Isolation**: Multi-tenant support with user-level data segregation
- **📈 Monitoring**: Built-in performance metrics and logging
- **🛡️ Type Safety**: Full Pydantic validation and type hints
- **🔧 Extensible**: Plugin architecture for adding new document extractors

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Architecture](#-architecture)
- [Development](#-development)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- AWS Account with Bedrock access
- 4GB+ RAM recommended
- 10GB+ disk space for document storage

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd rag-knowledge-base
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

Create a `.env` file in the project root:

```env
# AWS Bedrock Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
# OR use AWS profile
AWS_PROFILE_NAME=your_profile_name

# Model Configuration
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022
EMBEDDING_MODEL_ID=cohere.embed-english-v3
BEDROCK_VISION_MODEL_ID=us.anthropic.claude-3-5-sonnet-20241022-v2:0

# Model Parameters
TEMPERATURE=0.5
MAX_TOKENS=4000
VISION_TEMPERATURE=0.3
VISION_MAX_TOKENS=2000

# Storage Configuration
UPLOAD_DIR=./uploads
MARKDOWN_DIR=./markdown_output
DATABASE_URL=sqlite:///./rag_system.db
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=rag_documents

# File Upload Limits
MAX_FILE_SIZE_MB=50
MAX_FILENAME_LENGTH=255

# Chunking Configuration
CHUNK_SIZE=1500
CHUNK_OVERLAP=200
MAX_CHUNK_SIZE=6000

# RAG Configuration
DEFAULT_TOP_K=5
MAX_TOP_K=20
MAX_CONTEXT_LENGTH=8000
DEFAULT_KEYWORD_WEIGHT=0.3
DEFAULT_SEMANTIC_WEIGHT=0.7

# Conversation Configuration
MAX_CONVERSATION_TURNS=5
CONVERSATION_CLEANUP_MINUTES=60

# Feature Flags
ENABLE_AI_IMAGE_ANALYSIS=true
EXTRACT_IMAGES=true
INFER_TABLE_STRUCTURE=true
ENABLE_OCR=true
ENABLE_CONVERSATION_HISTORY=true
ENABLE_HALLUCINATION_DETECTION=false

# Logging
LOG_LEVEL=INFO
ENABLE_QUERY_LOGGING=true
ENABLE_PERFORMANCE_METRICS=true
LOG_FILE_PATH=./logs/rag_system.log

# Error Handling
MAX_RETRIES=3
RETRY_DELAY_SECONDS=1.0
```

### Step 5: Initialize Application

```bash
# Run database migrations
python -m app.db.init_db

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## ⚡ Quick Start

### 1. Upload a Document

```bash
curl -X POST "http://localhost:8000/api/upload/" \
  -H "X-User-ID: user123" \
  -F "file=@document.pdf"
```

Response:
```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "file_type": "pdf",
  "file_size_bytes": 1048576,
  "upload_time": "2024-01-15T10:30:00",
  "status": "uploaded",
  "message": "File 'document.pdf' uploaded successfully..."
}
```

### 2. Process the Document

```bash
curl -X POST "http://localhost:8000/api/upload/process/550e8400-e29b-41d4-a716-446655440000" \
  -H "X-User-ID: user123"
```

Response:
```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "status": "completed",
  "chunks_created": 42,
  "total_pages": 10,
  "message": "✓ Document 'document.pdf' processed successfully:\n  • Format: PDF\n  • Chunks indexed: 42\n  • Images: 5\n  • Pages: 10"
}
```

### 3. Query the Knowledge Base

```bash
curl -X POST "http://localhost:8000/api/query/" \
  -H "Content-Type: application/json" \
  -H "X-User-ID: user123" \
  -d '{
    "question": "What are the main findings in the document?",
    "top_k": 5,
    "keyword_weight": 0.3,
    "semantic_weight": 0.7
  }'
```

Response:
```json
{
  "question": "What are the main findings in the document?",
  "answer": "Based on the document, the main findings are:\n1. [Finding 1] [Source 1]\n2. [Finding 2] [Source 2]...",
  "sources": [
    {
      "content": "Relevant chunk content...",
      "filename": "document.pdf",
      "chunk_id": "550e8400_chunk_0_a1b2c3d4",
      "relevance_score": 0.95,
      "page_number": 3,
      "chunk_type": "text"
    }
  ],
  "processing_time": 2.34,
  "search_strategy": "hybrid",
  "confidence_score": 0.89,
  "keyword_weight": 0.3,
  "semantic_weight": 0.7
}
```

## ⚙️ Configuration

### Model Selection

The system supports multiple Claude models:

```python
# Recommended models:
BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022"  # Latest, best quality
BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229"    # Faster, lower cost
BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307"     # Fastest, lowest cost
```

### Chunking Strategy

Adjust chunking based on your use case:

```python
# For technical documents with detailed content:
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

# For conversational content:
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# For dense academic papers:
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250
```

### Search Configuration

Tune hybrid search weights:

```python
# More semantic (better for conceptual questions):
DEFAULT_KEYWORD_WEIGHT = 0.2
DEFAULT_SEMANTIC_WEIGHT = 0.8

# Balanced (general use):
DEFAULT_KEYWORD_WEIGHT = 0.3
DEFAULT_SEMANTIC_WEIGHT = 0.7

# More keyword-based (better for specific terms):
DEFAULT_KEYWORD_WEIGHT = 0.5
DEFAULT_SEMANTIC_WEIGHT = 0.5
```

## 📚 API Documentation

### Interactive API Docs

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Core Endpoints

#### Upload Management
- `POST /api/upload/` - Upload a document
- `POST /api/upload/process/{file_id}` - Process and index document
- `GET /api/upload/list` - List uploaded files
- `GET /api/upload/supported-formats` - Get supported file formats
- `DELETE /api/upload/{file_id}` - Delete uploaded file

#### Query & Search
- `POST /api/query/` - Query knowledge base (single query)
- `POST /api/query/conversation` - Query with conversation history
- `POST /api/query/clear-conversation/{conversation_id}` - Clear conversation

#### Knowledge Base Management
- `GET /api/knowledge-base/status` - Get knowledge base statistics
- `DELETE /api/knowledge-base/document/{file_id}` - Delete processed document
- `DELETE /api/knowledge-base/reset` - Reset entire knowledge base (⚠️ dangerous!)

#### Settings
- `GET /api/settings/model` - Get current model configuration
- `POST /api/settings/model` - Update LLM model (temporary)

#### System
- `GET /` - API information
- `GET /api/health` - Health check

### Request Examples

#### Advanced Query with Filters

```python
import requests

response = requests.post(
    "http://localhost:8000/api/query/",
    headers={"X-User-ID": "user123"},
    json={
        "question": "Explain the methodology",
        "top_k": 10,
        "file_id": "specific-file-id",  # Optional: search only this file
        "keyword_weight": 0.4,
        "semantic_weight": 0.6,
        "temperature": 0.3  # Lower temperature for factual answers
    }
)

print(response.json())
```

#### Conversational Query

```python
response = requests.post(
    "http://localhost:8000/api/query/conversation",
    headers={"X-User-ID": "user123"},
    json={
        "question": "What does the data show?",
        "conversation_id": "conv_12345",  # Same ID for continued conversation
        "top_k": 5,
        "keyword_weight": 0.3,
        "semantic_weight": 0.7
    }
)
```

#### Check Knowledge Base Status

```python
response = requests.get(
    "http://localhost:8000/api/knowledge-base/status",
    headers={"X-User-ID": "user123"}
)

print(f"Total documents: {response.json()['total_documents']}")
print(f"Total chunks: {response.json()['total_chunks']}")
```

## 🏛️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Client Layer                      │
│  (Web App, Mobile App, API Clients)                │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│                 FastAPI Application                  │
│  ┌─────────────────────────────────────────────┐   │
│  │              API Routes Layer                │   │
│  │  • Upload Routes                            │   │
│  │  • Query Routes                             │   │
│  │  • Knowledge Base Routes                    │   │
│  │  • Settings Routes                          │   │
│  └────────────────┬────────────────────────────┘   │
│                   │                                  │
│  ┌────────────────▼────────────────────────────┐   │
│  │           Services Layer                     │   │
│  │  ┌──────────────┐  ┌──────────────┐        │   │
│  │  │Vector Store  │  │ LLM Service  │        │   │
│  │  │   Service    │  │   (Bedrock)  │        │   │
│  │  └──────────────┘  └──────────────┘        │   │
│  │  ┌──────────────┐  ┌──────────────┐        │   │
│  │  │  Chunking    │  │  Extraction  │        │   │
│  │  │   Service    │  │   Factory    │        │   │
│  │  └──────────────┘  └──────────────┘        │   │
│  └────────────────┬────────────────────────────┘   │
└───────────────────┼────────────────────────────────┘
                    │
┌───────────────────▼────────────────────────────────┐
│              Storage Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────┐ │
│  │   ChromaDB   │  │    SQLite    │  │ File    │ │
│  │   (Vectors)  │  │  (Metadata)  │  │ Storage │ │
│  └──────────────┘  └──────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Extraction Layer
**Purpose**: Convert various document formats to structured markdown

**Components**:
- `ExtractorFactory`: Dynamically selects appropriate extractor
- `PDFExtractor`: Uses Docling for advanced PDF processing
- `DocxExtractor`: Handles Word documents
- `PptxExtractor`: Processes PowerPoint presentations
- `ImageExtractor`: OCR and AI-powered image analysis
- `MarkdownExtractor`, `TextExtractor`: Handle text formats

**Flow**:
```
Document → ExtractorFactory → Specific Extractor → Markdown + Metadata
```

#### 2. Chunking Layer
**Purpose**: Split documents into semantically coherent chunks

**Strategy**:
- Semantic chunking with size limits
- Respects document structure (pages, sections, images)
- Configurable overlap for context preservation
- Visual element awareness (tables, images, charts)

**Key Features**:
- Adaptive chunk sizes based on content type
- Page number preservation
- Metadata enrichment
- Strict size validation

#### 3. Vector Store Layer
**Purpose**: Store and retrieve document chunks using embeddings

**Components**:
- **ChromaDB**: Vector database with embedded embeddings
- **BM25 Index**: In-memory keyword search index
- **Hybrid Search**: Combines semantic and keyword search

**Search Process**:
```
Query → Embedding → ChromaDB Semantic Search
                  ↓
Query → Tokenize → BM25 Keyword Search
                  ↓
              Combine Results → Rank → Return Top K
```

#### 4. LLM Layer
**Purpose**: Generate contextual answers using retrieved chunks

**Features**:
- Vision-enabled responses for image-containing contexts
- Conversation history tracking
- Citation verification
- Hallucination detection (optional)
- Context compression for long documents

**Response Generation**:
```
Question + Retrieved Chunks + History → Prompt → LLM → Answer + Citations
```

## 🛠️ Development

### Project Structure

```
rag-knowledge-base/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── upload.py          # Document upload endpoints
│   │       ├── query.py           # Query endpoints
│   │       ├── knowledge_base.py  # KB management
│   │       └── settings.py        # Configuration endpoints
│   ├── core/
│   │   ├── config.py              # Application configuration
│   │   ├── exceptions.py          # Custom exceptions
│   │   ├── monitoring.py          # Metrics and logging
│   │   └── dependencies.py        # Dependency injection
│   ├── db/
│   │   ├── models.py              # SQLAlchemy models
│   │   └── session.py             # Database session management
│   ├── models/
│   │   └── schemas.py             # Pydantic schemas
│   ├── services/
│   │   ├── vector_store_service.py   # Vector operations
│   │   ├── llm_service.py            # LLM interactions
│   │   ├── chunking_service.py       # Document chunking
│   │   └── extraction/
│   │       ├── base_extractor.py         # Abstract base class
│   │       ├── extractor_factory.py      # Factory pattern
│   │       ├── pdf_extractor.py          # PDF extraction
│   │       ├── docx_extractor.py         # DOCX extraction
│   │       └── ...                       # Other extractors
│   ├── utils/
│   │   ├── file_validator.py         # File validation
│   │   └── markdown_converter.py     # Markdown utilities
│   └── main.py                        # Application entry point
├── uploads/                           # Uploaded files
├── markdown_output/                   # Processed markdown
├── chroma_db/                         # ChromaDB persistence
├── logs/                              # Application logs
├── tests/                             # Test suite
├── .env                               # Environment variables
├── .env.example                       # Environment template
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

### Adding a New Extractor

1. Create extractor class:

```python
# app/services/extraction/extractors/custom_extractor.py

from ..base_extractor import BaseExtractor
from typing import List, Tuple, Dict

class CustomExtractor(BaseExtractor):
    def _get_supported_formats(self) -> List[str]:
        return ['xyz', 'abc']  # Your file extensions
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        # Validation logic
        return True, "Valid"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        # Extraction logic
        markdown_content = "# Extracted content..."
        success = True
        metadata = {
            'filename': 'file.xyz',
            'pages': 10
        }
        return markdown_content, success, metadata
```

2. Register in factory:

```python
# app/services/extraction/extractor_factory.py

def auto_register_extractors():
    # ... existing registrations ...
    
    try:
        from .extractors.custom_extractor import CustomExtractor
        ExtractorFactory.register_extractor(CustomExtractor)
    except ImportError as e:
        logger.warning(f"Could not register CustomExtractor: {e}")
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio

# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_chunking.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code
black app/ tests/

# Lint code
pylint app/

# Type checking
mypy app/

# Security scan
bandit -r app/
```

## 🚢 Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p uploads markdown_output chroma_db logs

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads
      - ./chroma_db:/app/chroma_db
      - ./logs:/app/logs
    environment:
      - AWS_REGION=${AWS_REGION}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - BEDROCK_MODEL_ID=${BEDROCK_MODEL_ID}
    restart: unless-stopped
```

### Production Configuration

```python
# For production, use:

# 1. PostgreSQL instead of SQLite
DATABASE_URL=postgresql://user:pass@localhost:5432/rag_db

# 2. Redis for caching
REDIS_URL=redis://localhost:6379/0

# 3. Proper logging
LOG_LEVEL=WARNING
LOG_FILE_PATH=/var/log/rag/app.log

# 4. Security
ENABLE_CORS=false
ALLOWED_ORIGINS=https://yourdomain.com

# 5. Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

### AWS Deployment

Using AWS Elastic Beanstalk:

```bash
# Initialize EB CLI
eb init -p python-3.10 rag-knowledge-base

# Create environment
eb create production-env

# Deploy
eb deploy

# Check status
eb status
```

Using AWS ECS:

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t rag-api .
docker tag rag-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/rag-api:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/rag-api:latest

# Deploy to ECS (use CloudFormation or Terraform)
```

## 🔧 Troubleshooting

### Common Issues

#### 1. ChromaDB Initialization Error

```
Error: ChromaDB failed to initialize
```

**Solution**:
```bash
# Delete ChromaDB directory and restart
rm -rf chroma_db/
python -m app.main
```

#### 2. AWS Bedrock Access Denied

```
Error: botocore.exceptions.ClientError: An error occurred (AccessDeniedException)
```

**Solution**:
- Verify AWS credentials are correct
- Check IAM permissions include `bedrock:InvokeModel`
- Ensure Bedrock is enabled in your AWS region
- Try using AWS profile instead of access keys

#### 3. Out of Memory Error

```
Error: MemoryError during chunking
```

**Solution**:
```python
# Reduce chunk size in .env
CHUNK_SIZE=1000
MAX_CHUNK_SIZE=4000

# Or increase system memory
```

#### 4. File Upload Timeout

```
Error: Request timeout during upload
```

**Solution**:
```python
# Increase timeout in main.py
app = FastAPI(timeout=300)  # 5 minutes

# Or reduce max file size
MAX_FILE_SIZE_MB=20
```

#### 5. Search Returns No Results

**Possible causes**:
- No documents have been processed
- User isolation preventing access
- Query doesn't match indexed content

**Solution**:
```bash
# Check knowledge base status
curl http://localhost:8000/api/knowledge-base/status \
  -H "X-User-ID: your_user_id"

# Verify user_id matches between upload and query
```

### Debug Mode

Enable detailed logging:

```bash
# In .env
LOG_LEVEL=DEBUG
ENABLE_QUERY_LOGGING=true
ENABLE_PERFORMANCE_METRICS=true
```

View logs:

```bash
# Real-time logs
tail -f logs/rag_system.log

# Search for errors
grep "ERROR" logs/rag_system.log

# Check performance
grep "completed in" logs/rag_system.log
```

### Performance Tuning

#### Slow Document Processing

```python
# Disable features you don't need:
ENABLE_AI_IMAGE_ANALYSIS=false
ENABLE_OCR=false
INFER_TABLE_STRUCTURE=false
```

#### Slow Query Response

```python
# Reduce retrieval size:
DEFAULT_TOP_K=3

# Adjust search weights (more keyword = faster):
DEFAULT_KEYWORD_WEIGHT=0.5
DEFAULT_SEMANTIC_WEIGHT=0.5

# Reduce temperature for faster generation:
TEMPERATURE=0.3
MAX_TOKENS=2000
```

## 📊 Monitoring

### Metrics Endpoints

```python
# Get system metrics
GET /api/metrics

# Response:
{
  "uptime_seconds": 3600,
  "total_requests": 150,
  "total_errors": 2,
  "error_rate": 0.013,
  "endpoints": {
    "upload_file": {
      "request_count": 25,
      "avg_time_seconds": 1.234,
      "p95_time_seconds": 2.456
    }
  }
}
```

### Health Monitoring

```bash
# Simple health check
curl http://localhost:8000/api/health

# Detailed health check
curl http://localhost:8000/api/health/detailed
```

### Prometheus Integration

```python
# Add to requirements.txt:
prometheus-client

# In main.py:
from prometheus_client import make_asgi_app

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines first.

### Development Setup

```bash
# Fork and clone repository
git clone https://github.com/yourusername/rag-knowledge-base.git

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
pytest

# Commit with conventional commits
git commit -m "feat: add amazing feature"

# Push and create PR
git push origin feature/amazing-feature
```

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for all functions
- Add unit tests for new features
- Update README for API changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [LangChain](https://python.langchain.com/) - LLM framework
- [Docling](https://github.com/DS4SD/docling) - Document parsing
- [AWS Bedrock](https://aws.amazon.com/bedrock/) - LLM infrastructure

## 📧 Support

- Documentation: [Link to docs]
- Issues: [GitHub Issues](https://github.com/yourusername/rag-knowledge-base/issues)
- Email: support@yourdomain.com

## 🗺️ Roadmap

### Version 1.1 (Next Release)
- [ ] PostgreSQL support
- [ ] Redis caching layer
- [ ] Batch document processing
- [ ] Document versioning
- [ ] Advanced filters for search

### Version 1.2
- [ ] Multi-language support
- [ ] Custom embedding models
- [ ] GraphQL API
- [ ] Real-time collaboration
- [ ] Admin dashboard

### Version 2.0
- [ ] Distributed architecture
- [ ] Kubernetes deployment
- [ ] Advanced analytics
- [ ] Federated search
- [ ] Enterprise features

---

**Built with ❤️ using FastAPI, ChromaDB, and AWS Bedrock**