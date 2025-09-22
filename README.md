# Sentuh Tanahku Chatbot API

Enhanced conversational AI API with Indonesian NLP, vector embeddings, and PostgreSQL database integration.

## 🏗️ Project Structure (Refactored)

```
ChatbotSentuh/
├── src/                          # Source code
│   ├── api/                      # FastAPI application
│   │   ├── main.py              # Main FastAPI app
│   │   ├── models.py            # Pydantic models
│   │   └── endpoints/           # API endpoints (planned)
│   ├── core/                    # Core business logic
│   │   ├── config.py            # Configuration & feature detection
│   │   ├── nlp/                 # Indonesian NLP processing
│   │   │   ├── processor.py     # Text processing & similarity
│   │   │   ├── conversation.py  # Conversation handling
│   │   │   └── confidence.py    # Confidence evaluation
│   │   └── embeddings/          # Vector embeddings
│   │       └── vector_store.py  # ChromaDB vector storage
│   ├── database/                # Database layer
│   │   └── connection.py        # PostgreSQL connection & queries
│   ├── services/                # Business services
│   │   ├── query_processor.py   # Main query processing
│   │   └── response_generator.py # Response generation
│   └── utils/                   # Utilities
│       └── cache.py            # LRU cache implementation
├── tests/                       # Test files
│   └── test_database.py        # Database tests
├── data/                        # Data files
│   ├── csv/                     # CSV data files
│   └── embeddings/              # Vector embeddings storage
│       └── chroma_db/          # ChromaDB files
├── docker/                      # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
├── config/                      # Configuration files
│   └── credentials.json
├── app.py                       # Development entry point (with reload)
├── main.py                      # Production entry point (refactored)
└── requirements.txt
```

## 🚀 Features

- **Multi-layer Query Processing**: Vector embeddings + NLP similarity + conversation AI
- **Indonesian NLP**: Advanced text processing with Sastrawi stemming & stopword removal
- **Vector Embeddings**: ChromaDB with SentenceTransformers for semantic search
- **PostgreSQL Integration**: Persistent storage with optimized bulk operations
- **Performance Optimizations**: LRU caching, background processing, parallel execution
- **RESTful API**: FastAPI with streaming support and comprehensive endpoints

## 🛠️ Installation

1. **Clone & Setup**:
   ```bash
   git clone <repository>
   cd ChatbotSentuh
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   ```bash
   cp config/.env.example .env
   # Edit .env with your PostgreSQL credentials
   ```

3. **Database Setup**:
   ```bash
   # Create PostgreSQL database and faqs table
   # Update .env with database credentials
   ```

## 🏃‍♂️ Running the Application

### Development Mode (Recommended):
```bash
# Main entry point (with auto-reload)
python app.py

# Alternative: Direct uvicorn
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode:
```bash
# Production entry point (no reload) - using app.py but without reload
python -c "
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import uvicorn
uvicorn.run('src.api.main:app', host='0.0.0.0', port=8000, reload=False)
"

# Or: Direct module execution
python -m src.api.main
```

### Docker:
```bash
# Simple run (builds automatically)
cd docker/
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f sentuh-tanahku-api

# Stop
docker-compose down
```

## 📡 API Endpoints

### Core Endpoints:
- `GET /` - API information
- `GET /health` - Health check
- `POST /query` - Main query endpoint
- `POST /chat-stream` - Streaming chat endpoint

### Data Management:
- `POST /sync` - Sync database data
- `POST /add-faq` - Add single FAQ
- `POST /upload-csv` - Bulk upload CSV

### System:
- `GET /data-info` - Data statistics
- `GET /cache-stats` - Cache performance
- `POST /clear-cache` - Clear caches

## 🧪 Testing

### Local Testing:
```bash
# Test database connection
python tests/test_database.py

# Test imports
python -c "import sys; sys.path.insert(0, 'src'); from src.api.main import app; print('API import successful')"

# Test app.py entry point
python app.py
# Should start server on http://localhost:8000
```

### Docker Testing:
```bash
# Build and test Docker container
cd docker/
docker-compose up --build

# Test API health
curl http://localhost:8000/health

# Test main endpoint
curl http://localhost:8000/

# Clean up
docker-compose down
```

## 🔧 Architecture

### Processing Pipeline:
1. **Conversation Detection** - Greetings, goodbyes
2. **Vector Search** - Semantic similarity (ChromaDB)
3. **NLP Search** - Text similarity (optimized indexes)
4. **Confidence Evaluation** - Multi-factor scoring
5. **Response Generation** - Context-aware responses

### Performance Features:
- **Smart Caching**: LRU cache with selective caching based on confidence
- **Background Processing**: Non-blocking vector embeddings initialization  
- **Parallel Execution**: Async vector + NLP search
- **Optimized Indexing**: Pre-computed keyword lookup for ultra-fast search

## 📊 Monitoring

### Cache Statistics:
```bash
curl http://localhost:8000/cache-stats
```

### Health Check:
```bash
curl http://localhost:8000/health
```

## 🔄 Migration Notes

This is a refactored version of the original monolithic `main.py`. Key improvements:

1. **Modular Structure**: Separated concerns into logical modules
2. **Better Imports**: Clear dependency structure
3. **Testability**: Isolated components for easier testing
4. **Maintainability**: Smaller, focused files
5. **Scalability**: Easier to extend and modify

## 🐛 Troubleshooting

### Import Issues:
```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Database Connection:
```bash
# Test database
python tests/test_database.py
```

### Vector Embeddings:
- ChromaDB files stored in `data/embeddings/chroma_db/`
- Automatic fallback to NLP-only mode if embeddings fail

## 📝 TODO

- [ ] Add comprehensive API endpoint tests
- [ ] Implement API versioning
- [ ] Add monitoring/logging service
- [ ] Create migration scripts
- [ ] Add CI/CD configuration