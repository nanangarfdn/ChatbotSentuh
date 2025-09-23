#!/usr/bin/env python3
"""
Enhanced Sentuh Tanahku API with Best Practice Implementation
Combines vector embeddings, Indonesian NLP, and conversational AI
"""

import asyncio
import csv
import io
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import uvicorn
# Load environment variables
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response

# Internal imports
from src.api.models import (
    ChatRequest, QueryRequest, ChatResponse, SyncResponse,
    AddFAQRequest, AddFAQResponse, UploadCSVResponse
)
from src.core.config import VECTOR_EMBEDDINGS_AVAILABLE, SASTRAWI_AVAILABLE
from src.database.connection import db
from src.services.query_processor import EnhancedQueryProcessor
from src.utils.cache import response_cache, conversation_contexts

load_dotenv()

app = FastAPI(
    title="Enhanced Sentuh Tanahku API",
    description="Conversational AI dengan NLP Indonesia & PostgreSQL Database",
    version="2.0.0",
    timeout=300  # 5 minutes timeout for large uploads
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
knowledge_data = []
query_processor = None
thread_executor = ThreadPoolExecutor(max_workers=8)
logging_queue = asyncio.Queue(maxsize=1000)


# Database functions
async def load_knowledge_from_db():
    """Load approved knowledge data from PostgreSQL database"""
    try:
        return db.get_approved_faqs()
    except Exception as e:
        print(f"Warning: Failed to load data from database: {e}")
        return []


async def pre_warm_system():
    """Pre-warm the system to reduce first request latency with intelligent caching"""
    try:
        # Wait a bit for system to be ready
        await asyncio.sleep(3)

        if query_processor:
            print("🔥 Enhanced pre-warming system starting...")

            # Step 1: Basic system warm-up
            basic_queries = [
                "halo",
                "terima kasih"
            ]

            for query in basic_queries:
                try:
                    await query_processor.process_query(query)
                    print(f"✅ Basic warm-up: {query}")
                except Exception as e:
                    print(f"⚠️ Basic warm-up failed for '{query}': {e}")
                await asyncio.sleep(0.2)

            # Step 2: Pre-cache popular queries (intelligent caching)
            await query_processor.pre_cache_popular_queries()

            print("🔥 Enhanced pre-warming completed - system ready for optimal performance!")

    except Exception as e:
        print(f"⚠️ Pre-warming failed: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize database connection and query processor"""
    global knowledge_data, query_processor

    try:
        print("🚀 Initializing Enhanced Sentuh Tanahku API...")

        # Load data from PostgreSQL database
        print("📊 Loading data from PostgreSQL database...")
        db_data = await load_knowledge_from_db()
        
        # Convert database records to expected format
        knowledge_data = []
        for record in db_data:
            knowledge_data.append({
                'question': record['question'],
                'answer': record['answer'],
                'category': record.get('category', 'Umum')
            })
        
        print(f"✅ Loaded {len(knowledge_data)} knowledge items from PostgreSQL database")

        # Initialize query processor
        print("🧠 Initializing Enhanced Query Processor...")
        query_processor = EnhancedQueryProcessor(knowledge_data)
        print("✅ Query Processor initialized successfully")

        # Pre-warm the system for faster first requests
        print("🔥 Pre-warming system...")
        asyncio.create_task(pre_warm_system())

        print("🎯 Enhanced API is ready!")

    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🚀 Enhanced Sentuh Tanahku API v2.0",
        "description": "Conversational AI dengan NLP Indonesia & PostgreSQL Database",
        "features": [
            "Conversational AI (greetings, clarifications)",
            "Multi-layer query processing",
            "Confidence-based responses",
            "PostgreSQL database integration",
            "Indonesian NLP processing"
        ],
        "endpoints": [
            "GET /health - Health check",
            "POST /chat-stream - Dedicated streaming endpoint",
            "POST /query - Query endpoint with consistent response format",
            "POST /sync - Sync PostgreSQL database data to application memory",
            "POST /add-faq - Add single FAQ entry to database (requires manual sync)",
            "POST /upload-csv - Upload FAQ data from CSV file to database (requires manual sync)",
            "GET /user-queries - View user queries with approval status",
            "GET /data-info - Current data information",
            "GET /suggestions - Get query suggestions",
            "GET /cache-stats - Cache performance stats",
            "POST /clear-cache - Clear all caches",
            "GET /download-cache-csv - Download cache data as CSV",
            "GET /docs - API documentation"
        ],
        "data_loaded": len(knowledge_data),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Enhanced health check"""
    return {
        "status": "healthy",
        "components": {
            "postgresql_database": True,
            "query_processor": query_processor is not None,
            "nlp_processor": True,
            "vector_embeddings": VECTOR_EMBEDDINGS_AVAILABLE,
            "sastrawi": SASTRAWI_AVAILABLE
        },
        "data_count": len(knowledge_data),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/query", response_model=ChatResponse)
async def advanced_query(request: QueryRequest):
    """Advanced query endpoint with consistent response format and optional streaming"""

    if not query_processor:
        raise HTTPException(status_code=503, detail="Query processor not initialized")

    try:
        # Handle streaming response
        if request.stream:
            async def generate_stream():
                async for chunk in query_processor.process_query_stream(request.question):
                    yield chunk

            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*"
                }
            )

        # Process query using main processor
        result = await query_processor.process_query(request.question)

        # Limit sources to requested max_results
        if 'sources' in result and result['sources']:
            result['sources'] = result['sources'][:request.max_results]

        # Ensure suggestions are clean strings for frontend compatibility
        if 'suggestions' in result and result['suggestions']:
            if isinstance(result['suggestions'][0], dict):
                result['suggestions'] = [s.get('question', '') for s in result['suggestions']]

        # Return consistent response format
        return ChatResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/sync", response_model=SyncResponse)
async def sync_data():
    """Sync latest data from PostgreSQL database"""
    global knowledge_data, query_processor

    try:
        print("🔄 Syncing data from PostgreSQL database...")

        # Load fresh data from database
        old_count = len(knowledge_data)
        db_data = await load_knowledge_from_db()
        
        # Convert database records to expected format
        new_data = []
        for record in db_data:
            new_data.append({
                'question': record['question'],
                'answer': record['answer'],
                'category': record.get('category', 'Umum')
            })
        
        new_count = len(new_data)

        # Detect changes
        changes = {
            "added": new_count - old_count if new_count > old_count else 0,
            "removed": old_count - new_count if old_count > new_count else 0,
            "net_change": new_count - old_count
        }

        # Update data and processor
        knowledge_data = new_data
        query_processor = EnhancedQueryProcessor(knowledge_data)

        print(f"✅ Sync completed: {old_count} → {new_count} items")

        return SyncResponse(
            status="success",
            message="Data synchronized successfully from PostgreSQL",
            old_count=old_count,
            new_count=new_count,
            changes=changes,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        print(f"❌ Sync failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@app.post("/add-faq", response_model=AddFAQResponse)
async def add_single_faq(request: AddFAQRequest):
    """Add a single FAQ entry"""
    try:
        # Add FAQ to database
        faq_id = db.add_faq(request.question, request.answer, request.category)
        
        if not faq_id:
            raise HTTPException(status_code=400, detail="Failed to add FAQ to database")
        
        return AddFAQResponse(
            status="success",
            message="FAQ added successfully to database",
            faq_id=faq_id,
            question=request.question,
            answer=request.answer,
            category=request.category,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"❌ Add FAQ failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add FAQ: {str(e)}")


@app.post("/upload-csv", response_model=UploadCSVResponse)
async def upload_csv_file(file: UploadFile = File(...), use_fast_method: bool = True):
    """Upload FAQ data from CSV file with optimized bulk processing"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Validate file size (limit to 50MB)
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB")
        
        csv_content = content.decode('utf-8')
        print(f"📄 Processing CSV file: {file.filename} ({len(content)} bytes)")
        
        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        
        faqs_data = []
        total_processed = 0
        parsing_errors = 0
        
        for row_num, row in enumerate(csv_reader, 1):
            total_processed += 1
            
            try:
                # Map CSV columns (based on example.csv format)
                question = row.get('Pertanyaan', '').strip()
                answer = row.get('Jawaban', '').strip()
                category = row.get('Kategori', 'Umum').strip()
                
                if question and answer:
                    faqs_data.append({
                        'question': question,
                        'answer': answer,
                        'category': category
                    })
                else:
                    parsing_errors += 1
                    
            except Exception as e:
                parsing_errors += 1
                print(f"⚠️ Error parsing row {row_num}: {str(e)}")
        
        print(f"📊 CSV parsing completed: {len(faqs_data)} valid FAQs from {total_processed} rows ({parsing_errors} parsing errors)")
        
        if not faqs_data:
            return UploadCSVResponse(
                status="failure",
                message="No valid FAQ data found in CSV file",
                total_processed=total_processed,
                successful_inserts=0,
                failed_inserts=0,
                errors=["No valid FAQ data found"],
                timestamp=datetime.now().isoformat()
            )
        
        # Choose bulk insert method based on size and preference
        if use_fast_method and len(faqs_data) > 50:
            print("🚀 Using ultra-fast single transaction method for large dataset")
            result = db.bulk_add_faqs_single_transaction(faqs_data)
        else:
            print("⚡ Using optimized batch processing method")
            result = db.bulk_add_faqs(faqs_data)
        
        # Note: Data has been added to database. Use /sync endpoint to update application data.
        
        # Determine response status
        if result['successful'] == 0:
            status = "failure"
        elif result['failed'] > 0:
            status = "partial_success"
        else:
            status = "success"
        
        return UploadCSVResponse(
            status=status,
            message=f"CSV upload completed. {result['successful']} FAQs added to database, {result['failed']} failed from {len(faqs_data)} valid entries. Use /sync endpoint to update application data.",
            total_processed=total_processed,
            successful_inserts=result['successful'],
            failed_inserts=result['failed'],
            errors=result['errors'],
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ CSV upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload CSV: {str(e)}")


@app.get("/user-queries")
async def get_user_queries(limit: int = 50, approved_only: bool = False):
    """Get user queries with approval status and confidence scores"""
    try:
        # Build query based on filters
        base_query = """
        SELECT id, question, answer, approved, confident, created_at 
        FROM faqs 
        WHERE category = 'User Query'
        """
        
        if approved_only:
            base_query += " AND approved = 1"
        
        base_query += " ORDER BY created_at DESC"
        
        if limit > 0:
            base_query += f" LIMIT {min(limit, 200)}"  # Cap at 200 for performance
        
        result = db.execute_query(base_query, fetch=True)
        
        # Format response
        user_queries = []
        approved_count = 0
        pending_count = 0
        
        for row in result:
            query_data = {
                "id": row['id'],
                "question": row['question'],
                "answer": row['answer'][:200] + "..." if len(row['answer']) > 200 else row['answer'],
                "approved": bool(row['approved']),
                "confidence_score": float(row['confident']) if row['confident'] else 0.0,
                "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                "status": "approved" if row['approved'] == 1 else "pending_review"
            }
            user_queries.append(query_data)
            
            if row['approved'] == 1:
                approved_count += 1
            else:
                pending_count += 1
        
        return {
            "status": "success",
            "total_queries": len(user_queries),
            "approved_count": approved_count,
            "pending_count": pending_count,
            "queries": user_queries,
            "filters": {
                "limit": limit,
                "approved_only": approved_only
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error getting user queries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get user queries: {str(e)}")


@app.get("/data-info")
async def get_data_info():
    """Get detailed information about current data"""

    if not knowledge_data:
        return {
            "status": "no_data",
            "count": 0,
            "categories": {},
            "last_sync": None
        }

    # Count by categories
    categories = {}
    for item in knowledge_data:
        cat = item.get('category', 'Uncategorized').strip()
        if not cat:
            cat = 'Uncategorized'
        categories[cat] = categories.get(cat, 0) + 1

    # Get sample questions by category
    samples_by_category = {}
    for cat in categories.keys():
        if cat != 'Uncategorized':
            cat_items = [item for item in knowledge_data if item.get('category', '').strip() == cat]
            if cat_items:
                samples_by_category[cat] = [item['question'][:60] + "..." for item in cat_items[:2]]

    return {
        "status": "loaded",
        "total_count": len(knowledge_data),
        "categories": categories,
        "samples_by_category": samples_by_category,
        "features_enabled": {
            "vector_embeddings": VECTOR_EMBEDDINGS_AVAILABLE,
            "sastrawi_nlp": SASTRAWI_AVAILABLE,
            "conversational_ai": True,
            "postgresql_database": True
        },
        "last_updated": datetime.now().isoformat()
    }


@app.get("/suggestions")
async def get_suggestions():
    """Get query suggestions for users"""

    suggestions = [
        "Halo, apa itu Sentuh Tanahku?",
        "Bagaimana cara mendaftar di aplikasi?",
        "Bagaimana cara cek sertifikat tanah saya?",
        "Apa saja fitur utama Sentuh Tanahku?",
        "Bagaimana cara menggunakan antrian online?",
        "Apakah aplikasi ini gratis?",
        "Siapa saja yang bisa menggunakan aplikasi ini?",
        "Bagaimana cara melihat lokasi tanah di peta?"
    ]

    # Add category-based suggestions
    if knowledge_data:
        categories = list(
            set([item.get('category', '') for item in knowledge_data if item.get('category', '').strip()]))
        category_suggestions = [f"Tanyakan tentang {cat}" for cat in categories[:5] if cat and cat != 'Uncategorized']
        suggestions.extend(category_suggestions)

    return {
        "suggestions": suggestions[:10],
        "categories": list(
            set([item.get('category', '') for item in knowledge_data if item.get('category', '').strip()]))[:8]
    }


@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    """Dedicated streaming chat endpoint"""

    if not query_processor:
        raise HTTPException(status_code=503, detail="Query processor not initialized")

    try:
        # Get or create conversation context
        conversation_context = conversation_contexts.get(request.conversation_id, {})

        async def generate_stream():
            async for chunk in query_processor.process_query_stream(request.message, conversation_context):
                yield chunk

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")


@app.get("/cache-stats")
async def get_cache_stats():
    """Get cache statistics for performance monitoring"""
    import sys
    from src.utils.cache import similarity_cache, processed_text_cache

    def format_bytes(bytes_size):
        """Format bytes into human readable format"""
        if bytes_size == 0:
            return "0 B"

        sizes = ['B', 'KB', 'MB', 'GB', 'TB']
        i = 0
        while bytes_size >= 1024 and i < len(sizes) - 1:
            bytes_size /= 1024
            i += 1

        return f"{bytes_size:.2f} {sizes[i]}"

    # Calculate actual memory usage of caches
    def get_size(obj):
        """Get size of object in bytes"""
        try:
            return sys.getsizeof(obj)
        except:
            return 0

    # Calculate cache sizes
    similarity_cache_bytes = sum(get_size(k) + get_size(v) for k, v in similarity_cache.items())
    response_cache_bytes = sum(get_size(k) + get_size(v) for k, v in response_cache.items())
    text_cache_bytes = sum(get_size(k) + get_size(v) for k, v in processed_text_cache.items())

    return {
        "cache_counts": {
            "similarity_cache_entries": len(similarity_cache),
            "response_cache_entries": len(response_cache),
            "processed_text_cache_entries": len(processed_text_cache),
            "conversation_contexts": len(conversation_contexts)
        },
        "cache_sizes_bytes": {
            "similarity_cache_bytes": similarity_cache_bytes,
            "response_cache_bytes": response_cache_bytes,
            "processed_text_cache_bytes": text_cache_bytes,
            "total_cache_bytes": similarity_cache_bytes + response_cache_bytes + text_cache_bytes
        },
        "cache_sizes_human": {
            "similarity_cache": format_bytes(similarity_cache_bytes),
            "response_cache": format_bytes(response_cache_bytes),
            "processed_text_cache": format_bytes(text_cache_bytes),
            "total_cache": format_bytes(similarity_cache_bytes + response_cache_bytes + text_cache_bytes)
        },
        "system_info": {
            "logging_queue_size": logging_queue.qsize(),
            "thread_pool_workers": thread_executor._max_workers,
            "cache_policy": "no query response caching - fresh synced data only"
        },
        "performance_optimizations": [
            "Query response caching disabled - fresh data only",
            "Async logging queue",
            "Enhanced text caching for NLP processing",
            "Two-pass similarity search",
            "Increased thread workers",
            "Optimized thresholds",
            "Lazy vector embeddings from synced data"
        ],
        "cache_policy": {
            "query_response_caching": "disabled",
            "knowledge_vector_source": "synced_data_only",
            "note": "Queries are processed fresh using synced knowledge data"
        },
        "cache_efficiency": {
            "avg_similarity_cache_size_per_entry": format_bytes(similarity_cache_bytes / max(len(similarity_cache), 1)),
            "avg_response_cache_size_per_entry": format_bytes(response_cache_bytes / max(len(response_cache), 1)),
            "avg_text_cache_size_per_entry": format_bytes(text_cache_bytes / max(len(processed_text_cache), 1))
        }
    }


@app.post("/clear-cache")
async def clear_cache():
    """Clear remaining caches (response cache no longer used for queries)"""
    from src.utils.cache import similarity_cache, processed_text_cache

    # Clear remaining caches (response cache not used for queries anymore)
    similarity_cache.clear()
    response_cache.clear()  # Keep for compatibility but not used for query responses
    processed_text_cache.clear()
    conversation_contexts.clear()

    # Clear logging queue
    while not logging_queue.empty():
        try:
            logging_queue.get_nowait()
        except:
            break

    return {
        "status": "success",
        "message": "All caches cleared (note: query responses are no longer cached - fresh synced data used)",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/download-cache-csv")
async def download_cache_csv(cache_type: str = "all"):
    """Download cache data in CSV format
    
    Args:
        cache_type: Type of cache to download ('similarity', 'response', 'text', 'conversations', 'all')
    """
    try:
        from src.utils.cache import similarity_cache, processed_text_cache
        
        output = io.StringIO()
        writer = csv.writer(output)

        if cache_type == "similarity" or cache_type == "all":
            # Similarity cache CSV
            if cache_type == "all":
                writer.writerow(["=== SIMILARITY CACHE ==="])
                writer.writerow([])

            writer.writerow(["Cache Key", "Similarity Score", "Timestamp"])
            for cache_key, score in similarity_cache.items():
                # Handle both old format (just score) and new format (score, timestamp)
                if isinstance(score, tuple):
                    score_val, timestamp = score
                    timestamp_str = datetime.fromtimestamp(timestamp).isoformat()
                else:
                    score_val = score
                    timestamp_str = "N/A"

                writer.writerow([cache_key, score_val, timestamp_str])

            if cache_type == "all":
                writer.writerow([])
                writer.writerow([])

        if cache_type == "response" or cache_type == "all":
            # Response cache CSV
            if cache_type == "all":
                writer.writerow(["=== RESPONSE CACHE ==="])
                writer.writerow([])

            writer.writerow(
                ["Cache Key", "Response", "Response Type", "Confidence Score", "Sources Count", "Suggestions Count",
                 "Timestamp"])
            for cache_key, response_data in response_cache.items():
                if isinstance(response_data, tuple):
                    response_data, timestamp = response_data
                    timestamp_str = datetime.fromtimestamp(timestamp).isoformat()
                else:
                    timestamp_str = "N/A"
                    
                sources_count = len(response_data.get('sources', []))
                suggestions_count = len(response_data.get('suggestions', []))

                # Truncate long responses for CSV readability
                response_text = response_data.get('response', '')[:200]
                if len(response_data.get('response', '')) > 200:
                    response_text += "..."

                writer.writerow([
                    cache_key,
                    response_text,
                    response_data.get('response_type', ''),
                    response_data.get('confidence_score', 0),
                    sources_count,
                    suggestions_count,
                    timestamp_str
                ])

            if cache_type == "all":
                writer.writerow([])
                writer.writerow([])

        if cache_type == "text" or cache_type == "all":
            # Processed text cache CSV
            if cache_type == "all":
                writer.writerow(["=== PROCESSED TEXT CACHE ==="])
                writer.writerow([])

            writer.writerow(["Text Hash", "Processed Text"])
            for text_hash, processed_text in processed_text_cache.items():
                writer.writerow([text_hash, processed_text])

            if cache_type == "all":
                writer.writerow([])
                writer.writerow([])

        if cache_type == "conversations" or cache_type == "all":
            # Conversation contexts CSV
            if cache_type == "all":
                writer.writerow(["=== CONVERSATION CONTEXTS ==="])
                writer.writerow([])

            writer.writerow(["Conversation ID", "Last Query", "Last Response", "Timestamp"])
            for conv_id, context in conversation_contexts.items():
                # Truncate long texts for CSV readability
                last_query = context.get('last_query', '')[:150]
                if len(context.get('last_query', '')) > 150:
                    last_query += "..."

                last_response = context.get('last_response', '')[:150]
                if len(context.get('last_response', '')) > 150:
                    last_response += "..."

                timestamp_str = context.get('timestamp', datetime.now()).isoformat()

                writer.writerow([conv_id, last_query, last_response, timestamp_str])

        # Prepare CSV content
        csv_content = output.getvalue()
        output.close()

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sentuh_tanahku_cache_{cache_type}_{timestamp}.csv"

        # Return CSV file as download
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Cache-Control": "no-cache"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate CSV: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )