#!/usr/bin/env python3
"""
Enhanced Sentuh Tanahku API with Best Practice Implementation
Combines vector embeddings, Indonesian NLP, and conversational AI
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
import uvicorn
import json
import re
import asyncio
import time
import csv
import io
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, AsyncGenerator, Set
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, OrderedDict
from dotenv import load_dotenv
from database import db

# Additional imports for optimization
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available, using fallback implementations")

try:
    from pybloom_live import BloomFilter

    BLOOM_FILTER_AVAILABLE = True
except ImportError:
    BLOOM_FILTER_AVAILABLE = False
    print("Info: Bloom filter not available, using set-based filtering")

# Import for Indonesian text processing
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    print("Warning: Sastrawi not available, using basic text processing")

# Import vector embeddings if available
try:
    from vector_embeddings import IndonesianVectorEmbeddings

    VECTOR_EMBEDDINGS_AVAILABLE = True
except ImportError:
    VECTOR_EMBEDDINGS_AVAILABLE = False
    print("Info: Vector embeddings not available, using NLP only")

# Load environment variables
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


# Pydantic models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    stream: bool = False


class QueryRequest(BaseModel):
    question: str
    max_results: int = 5
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    confidence_score: float
    response_type: str  # confident, clarification, greeting, not_found
    sources: List[Dict] = []
    suggestions: List[Union[str, Dict]] = []
    response_time: float


class SyncResponse(BaseModel):
    status: str
    message: str
    old_count: int
    new_count: int
    changes: Dict
    timestamp: str


class AddFAQRequest(BaseModel):
    question: str
    answer: str
    category: Optional[str] = "Umum"


class AddFAQResponse(BaseModel):
    status: str
    message: str
    faq_id: int
    question: str
    answer: str
    category: str
    timestamp: str


class UploadCSVResponse(BaseModel):
    status: str
    message: str
    total_processed: int
    successful_inserts: int
    failed_inserts: int
    errors: List[str]
    timestamp: str


# Global variables
knowledge_data = []
vector_embeddings = None
conversation_contexts = {}

# Performance optimization variables - Enhanced
similarity_cache = {}
processed_text_cache = {}  # Cache for processed texts
cache_ttl = 300  # 5 minutes
thread_executor = ThreadPoolExecutor(max_workers=8)  # Increased workers


# New LRU Cache implementation for responses
class LRUCache:
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used
            self.cache.popitem(last=False)
        self.cache[key] = value

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.put(key, value)

    def __delitem__(self, key):
        if key in self.cache:
            del self.cache[key]

    def __contains__(self, key):
        return key in self.cache

    def clear(self):
        self.cache.clear()

    def __len__(self):
        return len(self.cache)

    def items(self):
        return self.cache.items()


# Replace simple dict with LRU cache
response_cache = LRUCache(capacity=5000)

# Knowledge data optimization structures
knowledge_index = {}
keyword_lookup = defaultdict(set)
processed_knowledge_cache = {}
bloom_filter = None

# Async logging queue
logging_queue = asyncio.Queue(maxsize=1000)


class IndonesianNLPProcessor:
    """Enhanced Indonesian text processing with advanced features"""

    def __init__(self):
        if SASTRAWI_AVAILABLE:
            self.stemmer = StemmerFactory().create_stemmer()
            self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        else:
            self.stemmer = None
            self.stopword_remover = None

        # Enhanced stopwords for better processing
        self.custom_stopwords = {
            'yang', 'ini', 'itu', 'dari', 'untuk', 'dengan', 'pada', 'dalam', 'ke', 'di',
            'atau', 'dan', 'adalah', 'akan', 'telah', 'sudah', 'bisa', 'dapat', 'harus',
            'jika', 'kalau', 'ketika', 'saat', 'waktu', 'cara', 'bagaimana', 'apa', 'siapa'
        }

        # Pre-computed knowledge indexes for ultra-fast lookup
        self.knowledge_index = {}  # question_id -> processed data
        self.keyword_lookup = defaultdict(set)  # keyword -> set of question_ids
        self.processed_knowledge = {}  # question_hash -> processed_text
        self.question_words_cache = {}  # question_id -> set of words
        self.bloom_filter = None  # For negative lookups

        # Fast similarity pre-computation
        self.similarity_matrix = {}  # Cache common similarity calculations
        self.indexed_knowledge_data = []  # Copy of knowledge data with indexes

    def clean_text(self, text: str) -> str:
        """Advanced text cleaning for Indonesian"""
        if not text:
            return ""

        text = text.lower()
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove quotes
        text = re.sub(r'^"(.*)"$', r'\1', text)
        # Remove special characters but keep Indonesian letters
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def process_text(self, text: str) -> str:
        """Process Indonesian text with stemming and stopword removal with caching"""
        if not text:
            return ""

        # Check cache first
        cache_key = hash(text)
        if cache_key in processed_text_cache:
            return processed_text_cache[cache_key]

        text = self.clean_text(text)

        # Remove custom stopwords
        words = text.split()
        words = [word for word in words if word not in self.custom_stopwords]
        text = ' '.join(words)

        if SASTRAWI_AVAILABLE and self.stopword_remover and self.stemmer:
            text = self.stopword_remover.remove(text)
            text = self.stemmer.stem(text)

        # Cache result permanently
        processed_text_cache[cache_key] = text

        # Optional: Only warn if cache gets very large
        if len(processed_text_cache) > 10000:
            print(f"⚠️  Text processing cache very large ({len(processed_text_cache)} entries)")

        return text

    def build_knowledge_index(self, knowledge_data: List[Dict]) -> None:
        """Build optimized indexes for ultra-fast search - called once at startup"""
        print(f"🔧 Building knowledge index for {len(knowledge_data)} items...")
        start_time = time.time()

        # Clear existing indexes
        self.knowledge_index.clear()
        self.keyword_lookup.clear()
        self.question_words_cache.clear()
        self.indexed_knowledge_data = knowledge_data.copy()

        # Initialize bloom filter if available
        if BLOOM_FILTER_AVAILABLE and len(knowledge_data) > 100:
            self.bloom_filter = BloomFilter(capacity=len(knowledge_data) * 10, error_rate=0.1)

        # Process each knowledge item
        for idx, item in enumerate(knowledge_data):
            question = item.get('question', '')
            answer = item.get('answer', '')

            if not question:
                continue

            # Process and cache question text
            processed_question = self.process_text(question)
            processed_answer = self.process_text(answer)

            # Store in main index
            self.knowledge_index[idx] = {
                'original_question': question,
                'original_answer': answer,
                'processed_question': processed_question,
                'processed_answer': processed_answer,
                'category': item.get('category', 'Umum'),
                'question_words': set(processed_question.split()) if processed_question else set(),
                'answer_words': set(processed_answer.split()) if processed_answer else set()
            }

            # Build keyword lookup for fast filtering
            question_words = set(processed_question.split()) if processed_question else set()
            self.question_words_cache[idx] = question_words

            for word in question_words:
                if len(word) > 2:  # Skip very short words
                    self.keyword_lookup[word].add(idx)
                    if self.bloom_filter:
                        self.bloom_filter.add(word)

        build_time = time.time() - start_time
        print(f"✅ Knowledge index built in {build_time:.2f}s - {len(self.keyword_lookup)} keywords indexed")

    def get_candidates_fast(self, query: str, max_candidates: int = 50) -> Set[int]:
        """Ultra-fast candidate selection using pre-built indexes"""
        if not self.keyword_lookup:
            return set(range(min(max_candidates, len(self.indexed_knowledge_data))))

        processed_query = self.process_text(query)
        if not processed_query:
            return set()

        query_words = set(processed_query.split())
        candidates = set()

        # Use bloom filter for negative lookups if available
        if self.bloom_filter:
            query_words = {word for word in query_words if word in self.bloom_filter}

        # Fast intersection using pre-computed keyword lookup
        for word in query_words:
            if word in self.keyword_lookup:
                candidates.update(self.keyword_lookup[word])
                if len(candidates) >= max_candidates * 2:  # Early termination
                    break

        # If no candidates found, use fallback strategy
        if not candidates and query_words:
            # Try partial matches for longer words
            for word in query_words:
                if len(word) > 4:
                    for keyword in self.keyword_lookup:
                        if word in keyword or keyword in word:
                            candidates.update(self.keyword_lookup[keyword])
                            if len(candidates) >= max_candidates:
                                break
                    if len(candidates) >= max_candidates:
                        break

        return candidates

    def calculate_similarity_fast(self, query_words: Set[str], item_idx: int) -> float:
        """Ultra-fast similarity calculation using pre-computed data"""
        if item_idx not in self.knowledge_index:
            return 0.0

        item_data = self.knowledge_index[item_idx]
        item_words = item_data['question_words']

        if not query_words or not item_words:
            return 0.0

        # Optimized Jaccard similarity with pre-computed sets
        intersection_len = len(query_words & item_words)
        union_len = len(query_words | item_words)

        if union_len == 0:
            return 0.0

        jaccard = intersection_len / union_len

        # Boost for exact word matches
        exact_boost = min(intersection_len * 0.1, 0.3)

        return min(jaccard + exact_boost, 1.0)

    def calculate_similarity(self, query: str, text: str) -> float:
        """Calculate enhanced similarity between query and text with optimized caching"""
        # Create cache key using faster hashing
        cache_key = f"{hash(query)}_{hash(text)}"

        # Check cache without timestamp check for speed
        if cache_key in similarity_cache:
            return similarity_cache[cache_key]

        query_processed = self.process_text(query)
        text_processed = self.process_text(text)

        if not query_processed or not text_processed:
            similarity_cache[cache_key] = 0.0
            return 0.0

        # Use pre-computed sets if available
        query_words = set(query_processed.split())
        text_words = set(text_processed.split())

        if not query_words or not text_words:
            similarity_cache[cache_key] = 0.0
            return 0.0

        # Optimized similarity calculation
        intersection_len = len(query_words & text_words)
        union_len = len(query_words | text_words)

        jaccard = intersection_len / union_len if union_len else 0.0

        # Boost for exact word matches
        exact_boost = min(intersection_len * 0.1, 0.3)

        similarity = min(jaccard + exact_boost, 1.0)

        # Only cache results with meaningful similarity (avoid caching random/irrelevant matches)
        if similarity >= 0.1:  # Only cache if similarity is at least 10%
            similarity_cache[cache_key] = similarity

            # Optional: Only limit cache if it gets extremely large (10K+ entries)
            if len(similarity_cache) > 10000:
                print(f"⚠️  Similarity cache very large ({len(similarity_cache)} entries)")
                # Could add cleanup here if needed, but keeping all cache for now
        # Don't cache very low similarity scores to avoid cache pollution

        return similarity

    def _clean_cache(self, current_time: float):
        """Remove expired cache entries"""
        expired_keys = [
            key for key, (_, timestamp) in similarity_cache.items()
            if current_time - timestamp > cache_ttl
        ]
        for key in expired_keys:
            del similarity_cache[key]


class ConversationHandler:
    """Handle conversational aspects like greetings and clarifications"""

    def __init__(self):
        self.greetings = [
            'halo', 'hai', 'hello', 'selamat', 'assalamualaikum', 'salam',
            'pagi', 'siang', 'sore', 'malam', 'good', 'morning'
        ]

        self.clarification_keywords = [
            'lebih detail', 'spesifik', 'jelaskan', 'maksudnya', 'artinya',
            'contoh', 'bagaimana', 'gimana', 'caranya', 'tolong'
        ]

        self.goodbye_keywords = [
            'terima kasih', 'makasih', 'thanks', 'bye', 'sampai jumpa', 'selesai'
        ]

    def is_greeting(self, text: str) -> bool:
        """Detect if text is a greeting"""
        text_lower = text.lower()
        return any(greeting in text_lower for greeting in self.greetings)

    def is_clarification_request(self, text: str) -> bool:
        """Detect if text is asking for clarification"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.clarification_keywords)

    def is_goodbye(self, text: str) -> bool:
        """Detect if text is a goodbye"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.goodbye_keywords)

    def generate_greeting_response(self) -> str:
        """Generate friendly greeting response"""
        responses = [
            "Halo! Saya asisten Sentuh Tanahku. Ada yang bisa saya bantu tentang aplikasi Sentuh Tanahku?",
            "Hai! Selamat datang di layanan bantuan Sentuh Tanahku. Silakan tanyakan apa saja tentang aplikasi ini.",
            "Halo! Saya siap membantu Anda dengan pertanyaan seputar Sentuh Tanahku. Ada yang ingin ditanyakan?"
        ]
        import random
        return random.choice(responses)

    def generate_goodbye_response(self) -> str:
        """Generate friendly goodbye response"""
        responses = [
            "Terima kasih telah menggunakan layanan Sentuh Tanahku! Semoga informasinya bermanfaat.",
            "Sama-sama! Jangan ragu untuk bertanya lagi kapan saja tentang Sentuh Tanahku.",
            "Senang bisa membantu! Sampai jumpa dan semoga sukses dengan urusan pertanahan Anda."
        ]
        import random
        return random.choice(responses)


class ConfidenceEvaluator:
    """Evaluate confidence levels and determine response strategy"""

    def __init__(self):
        self.confidence_thresholds = {
            'high': 0.7,
            'medium': 0.4,
            'low': 0.2
        }

    def evaluate_confidence(self, vector_score: float, nlp_score: float,
                            exact_match: bool = False) -> Dict:
        """Evaluate confidence and determine response type"""

        # Combine scores with weights
        if vector_score > 0:
            combined_score = (vector_score * 0.7) + (nlp_score * 0.3)
        else:
            combined_score = nlp_score

        # Boost for exact matches
        if exact_match:
            combined_score = min(combined_score + 0.2, 1.0)

        # Determine response type
        if combined_score >= self.confidence_thresholds['high']:
            response_type = 'confident'
        elif combined_score >= self.confidence_thresholds['medium']:
            response_type = 'clarification'
        else:
            response_type = 'not_found'

        return {
            'score': combined_score,
            'type': response_type,
            'vector_score': vector_score,
            'nlp_score': nlp_score
        }


class ResponseGenerator:
    """Generate appropriate responses based on confidence and context"""

    def __init__(self):
        self.conversation_handler = ConversationHandler()

    def generate_response(self, query: str, matches: List[Dict],
                          confidence_data: Dict, conversation_context: Dict = None) -> Dict:
        """Generate appropriate response based on confidence and matches"""

        confidence_score = confidence_data['score']
        response_type = confidence_data['type']

        if response_type == 'confident':
            return self._generate_confident_response(matches[0], confidence_score)
        elif response_type == 'clarification':
            return self._generate_clarification_response(query, matches, confidence_score)
        else:
            return self._generate_not_found_response(query, confidence_score)

    def _generate_confident_response(self, best_match: Dict, confidence: float) -> Dict:
        """Generate confident response with best match"""
        return {
            'response': best_match['answer'],
            'response_type': 'confident',
            'confidence_score': confidence,
            'sources': [best_match],
            'suggestions': []
        }

    def _generate_clarification_response(self, query: str, matches: List[Dict],
                                         confidence: float) -> Dict:
        """Generate clarification request with suggestions using top 3 similarity scores"""

        # Sort matches by similarity score and get top 3
        sorted_matches = sorted(matches, key=lambda x: x.get('similarity_score', 0), reverse=True)
        top_matches = sorted_matches[:3]

        # Get unique categories from top matches
        categories = list(set([match.get('category', 'Umum') for match in top_matches]))
        categories = [cat for cat in categories if cat and cat.strip() and cat != 'Uncategorized']

        if categories:
            category_text = ", ".join(categories[:3])
            response = f"Saya menemukan beberapa informasi terkait. Apakah Anda bertanya tentang: {category_text}? Bisa lebih spesifik?"
        else:
            response = "Saya menemukan beberapa informasi terkait, tapi bisa lebih spesifik pertanyaannya? Misalnya tentang pendaftaran, sertifikat, atau fitur tertentu."

        # Generate suggestions from top 3 similarity scores (full questions for frontend)
        suggestions = []
        for match in top_matches:
            # Use full question without truncation and without scores for better frontend UX
            suggestions.append(match['question'])

        return {
            'response': response,
            'response_type': 'clarification',
            'confidence_score': confidence,
            'sources': top_matches,
            'suggestions': suggestions
        }

    def _generate_not_found_response(self, query: str, confidence: float) -> Dict:
        """Generate helpful not found response"""

        responses = [
            "Maaf, saya tidak menemukan informasi spesifik tentang itu. Bisa coba pertanyaan yang lebih spesifik tentang Sentuh Tanahku?",
            "Saya belum menemukan jawaban yang tepat untuk pertanyaan tersebut. Coba tanyakan tentang fitur, cara pendaftaran, atau masalah teknis Sentuh Tanahku.",
            "Informasi tentang itu belum tersedia. Silakan tanyakan hal lain seputar aplikasi Sentuh Tanahku yang bisa saya bantu."
        ]

        import random
        response = random.choice(responses)

        # Add helpful suggestions from random questions in the sheet
        suggestions = []
        if knowledge_data:
            import random
            # Get random questions from the knowledge base
            random_items = random.sample(knowledge_data, min(4, len(knowledge_data)))
            suggestions = [item['question'] for item in random_items]
        else:
            # Fallback if no data available
            suggestions = [
                "Cara mendaftar di Sentuh Tanahku",
                "Bagaimana cek sertifikat tanah",
                "Fitur utama aplikasi Sentuh Tanahku",
                "Cara menggunakan antrian online"
            ]

        return {
            'response': response,
            'response_type': 'not_found',
            'confidence_score': confidence,
            'sources': [],
            'suggestions': suggestions
        }


class EnhancedQueryProcessor:
    """Main query processing system combining multiple approaches"""

    def __init__(self, knowledge_data: List[Dict]):
        self.knowledge_data = knowledge_data
        self.nlp_processor = IndonesianNLPProcessor()
        self.conversation_handler = ConversationHandler()
        self.confidence_evaluator = ConfidenceEvaluator()
        self.response_generator = ResponseGenerator()

        # Build optimized knowledge index immediately for faster searches
        if knowledge_data:
            print("🔧 Building optimized knowledge index...")
            self.nlp_processor.build_knowledge_index(knowledge_data)
            print("✅ Knowledge index ready for ultra-fast searches")

        # Background initialization for vector embeddings (avoid blocking startup)
        self.vector_embeddings = None
        self._vector_embeddings_initialized = False
        self._vector_embeddings_failed = False
        self._vector_init_task = None

        # Pre-compute knowledge data hash for change detection
        self._knowledge_hash = hash(str(sorted([item['question'] for item in knowledge_data]))) if knowledge_data else 0

        # Start background vector embeddings initialization
        if VECTOR_EMBEDDINGS_AVAILABLE and knowledge_data:
            print("🚀 Starting background vector embeddings initialization...")
            self._vector_init_task = asyncio.create_task(self._init_vector_embeddings_background())

    async def _init_vector_embeddings_background(self) -> None:
        """Initialize vector embeddings in background without blocking main operations"""
        try:
            # Small delay to let main system start up first
            await asyncio.sleep(1)

            print("🧠 Initializing vector embeddings in background...")

            def init_embeddings():
                try:
                    vector_embeddings = IndonesianVectorEmbeddings()
                    if self.knowledge_data:
                        vector_embeddings.add_knowledge_to_database(self.knowledge_data)
                    return vector_embeddings
                except Exception as e:
                    print(f"❌ Vector embeddings initialization failed: {e}")
                    return None

            # Run in thread executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            self.vector_embeddings = await loop.run_in_executor(
                thread_executor,
                init_embeddings
            )

            if self.vector_embeddings:
                self._vector_embeddings_initialized = True
                print("✅ Vector embeddings initialized successfully in background")
            else:
                self._vector_embeddings_failed = True
                print("⚠️ Vector embeddings initialization failed, using NLP-only mode")

        except Exception as e:
            print(f"❌ Background vector initialization error: {e}")
            self._vector_embeddings_failed = True

    async def process_query(self, query: str, conversation_context: Dict = None) -> Dict:
        """Main query processing with multi-layer approach and performance optimizations"""
        start_time = datetime.now()

        # Check response cache first (LRU cache for better memory management)
        cache_key = f"query_{hash(query)}"
        cached_result = response_cache.get(cache_key)
        if cached_result:
            cached_result = cached_result.copy()  # Create copy to avoid mutation
            cached_result['response_time'] = (datetime.now() - start_time).total_seconds()
            cached_result['cached'] = True
            return cached_result

        # Layer 1: Conversation detection (fastest)
        if self.conversation_handler.is_greeting(query):
            return {
                'response': self.conversation_handler.generate_greeting_response(),
                'response_type': 'greeting',
                'confidence_score': 1.0,
                'sources': [],
                'suggestions': [],
                'response_time': (datetime.now() - start_time).total_seconds()
            }

        if self.conversation_handler.is_goodbye(query):
            return {
                'response': self.conversation_handler.generate_goodbye_response(),
                'response_type': 'goodbye',
                'confidence_score': 1.0,
                'sources': [],
                'suggestions': [],
                'response_time': (datetime.now() - start_time).total_seconds()
            }

        # Parallel processing for performance
        tasks = []

        # Layer 2: Vector similarity (lazy initialization) - async
        vector_available = await self._ensure_vector_embeddings_async()
        if vector_available:
            tasks.append(self._get_vector_results_async(query))

        # Layer 3: NLP similarity search - async (always available)
        tasks.append(self._get_nlp_results_async(query))

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        vector_results = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else []
        nlp_results = results[-1] if len(results) > 0 and not isinstance(results[-1], Exception) else []

        vector_score = vector_results[0].get('similarity_score', 0.0) if vector_results else 0.0
        nlp_score = nlp_results[0]['similarity_score'] if nlp_results else 0.0

        # Layer 4: Combine results and evaluate confidence
        all_results = self._combine_results(vector_results, nlp_results)

        # Early termination for high confidence results
        if all_results and all_results[0]['similarity_score'] >= 0.8:
            response_data = {
                'response': all_results[0]['answer'],
                'response_type': 'confident',
                'confidence_score': all_results[0]['similarity_score'],
                'sources': all_results[:1],
                'suggestions': [],
                'response_time': (datetime.now() - start_time).total_seconds()
            }
            # Always cache high confidence results (≥80% similarity)
            response_cache[cache_key] = response_data
            return response_data

        # Check for exact matches
        exact_match = any(result['similarity_score'] >= 0.9 for result in all_results[:3])

        confidence_data = self.confidence_evaluator.evaluate_confidence(
            vector_score, nlp_score, exact_match
        )

        # Layer 5: Generate response
        response_data = self.response_generator.generate_response(
            query, all_results, confidence_data, conversation_context
        )

        response_data['response_time'] = (datetime.now() - start_time).total_seconds()

        # Only cache responses with decent confidence (avoid caching poor/irrelevant responses)
        confidence_score = response_data.get('confidence_score', 0.0)
        if confidence_score >= 0.2:  # Only cache if confidence is at least 20%
            response_cache[cache_key] = response_data

            # LRU cache automatically manages size, no need for manual cleanup
            if len(response_cache) > 4000:  # Only warn at higher threshold
                print(f"ℹ️  Response cache growing ({len(response_cache)} entries) - LRU cleanup active")
        # Don't cache low-confidence responses to avoid serving poor results

        return response_data

    async def _ensure_vector_embeddings_async(self) -> bool:
        """Ensure vector embeddings are initialized (lazy loading)"""
        if self._vector_embeddings_failed:
            return False

        if self._vector_embeddings_initialized:
            return True

        if not VECTOR_EMBEDDINGS_AVAILABLE:
            self._vector_embeddings_failed = True
            return False

        try:
            # Initialize in background thread to avoid blocking
            loop = asyncio.get_event_loop()

            def init_vector_embeddings():
                try:
                    vector_embeddings = IndonesianVectorEmbeddings()
                    if self.knowledge_data:
                        vector_embeddings.add_knowledge_to_database(self.knowledge_data)
                    return vector_embeddings
                except Exception as e:
                    print(f"Vector embeddings initialization failed: {e}")
                    return None

            self.vector_embeddings = await loop.run_in_executor(
                thread_executor,
                init_vector_embeddings
            )

            if self.vector_embeddings:
                self._vector_embeddings_initialized = True
                print("✅ Vector embeddings initialized successfully")
                return True
            else:
                self._vector_embeddings_failed = True
                return False

        except Exception as e:
            print(f"Failed to initialize vector embeddings: {e}")
            self._vector_embeddings_failed = True
            return False

    async def _get_vector_results_async(self, query: str) -> List:
        """Get vector search results asynchronously"""
        try:
            if not self.vector_embeddings:
                return []

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                thread_executor,
                lambda: self.vector_embeddings.search_similar(query, n_results=5)
            )
        except Exception as e:
            print(f"Vector search failed: {e}")
            return []

    async def _get_nlp_results_async(self, query: str) -> List[Dict]:
        """Get NLP search results asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                thread_executor,
                lambda: self._nlp_similarity_search(query)
            )
        except Exception as e:
            print(f"NLP search failed: {e}")
            return []

    def _clean_response_cache(self, current_time: float):
        """Remove expired response cache entries"""
        expired_keys = [
            key for key, (_, timestamp) in response_cache.items()
            if current_time - timestamp > cache_ttl
        ]
        for key in expired_keys:
            del response_cache[key]

    async def process_query_stream(self, query: str, conversation_context: Dict = None) -> AsyncGenerator[str, None]:
        """Process query with streaming response"""
        yield f"data: {{\"type\": \"processing\", \"message\": \"Memproses pertanyaan: {query}\"}}\n\n"

        # Get the complete response first
        response = await self.process_query(query, conversation_context)

        # Stream metadata first
        yield f"data: {{\"type\": \"start\", \"response_time\": {response.get('response_time', 0)}, \"response_type\": \"{response.get('response_type', '')}\", \"confidence_score\": {response.get('confidence_score', 0)}}}\n\n"

        # Stream the answer word by word
        answer = response.get('response', '')
        words = answer.split()

        for i, word in enumerate(words):
            yield f"data: {{\"type\": \"content\", \"content\": \"{word} \"}}\n\n"
            await asyncio.sleep(0.2)  # Small delay for streaming effect

        # Send suggestions if available
        if 'suggestions' in response and response['suggestions']:
            suggestions_json = json.dumps(response['suggestions'])
            yield f"data: {{\"type\": \"suggestions\", \"suggestions\": {suggestions_json}}}\n\n"

        # Send sources if available
        if response.get('sources'):
            sources_json = json.dumps(response['sources'])
            yield f"data: {{\"type\": \"sources\", \"sources\": {sources_json}}}\n\n"

        yield "data: {\"type\": \"done\"}\n\n"

    def _nlp_similarity_search(self, query: str) -> List[Dict]:
        """Ultra-fast NLP search using pre-computed indexes"""
        results = []
        threshold = 0.15

        # Use optimized index if available
        if hasattr(self.nlp_processor, 'knowledge_index') and self.nlp_processor.knowledge_index:
            return self._nlp_similarity_search_optimized(query, threshold)

        # Fallback to original method if index not available
        return self._nlp_similarity_search_fallback(query, threshold)

    def _nlp_similarity_search_optimized(self, query: str, threshold: float = 0.15) -> List[Dict]:
        """Ultra-fast search using pre-computed knowledge index"""
        # Get candidates using optimized index
        candidates = self.nlp_processor.get_candidates_fast(query, max_candidates=30)

        if not candidates:
            return []

        # Pre-process query once
        query_processed = self.nlp_processor.process_text(query)
        if not query_processed:
            return []

        query_words = set(query_processed.split())
        if not query_words:
            return []

        results = []

        # Calculate similarity for candidates using fast method
        for item_idx in candidates:
            if item_idx not in self.nlp_processor.knowledge_index:
                continue

            similarity_score = self.nlp_processor.calculate_similarity_fast(query_words, item_idx)

            if similarity_score >= threshold:
                item_data = self.nlp_processor.knowledge_index[item_idx]
                results.append({
                    'question': item_data['original_question'],
                    'answer': item_data['original_answer'],
                    'category': item_data['category'],
                    'similarity_score': similarity_score,
                    'question_match': similarity_score,
                    'answer_match': 0.0,
                    'source': 'NLP-Optimized'
                })

                # Early termination for excellent matches
                if similarity_score >= 0.85:
                    results.sort(key=lambda x: x['similarity_score'], reverse=True)
                    return results[:5]

        # Sort and return top results
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:8]

    def _nlp_similarity_search_fallback(self, query: str, threshold: float = 0.15) -> List[Dict]:
        """Fallback method when optimized index is not available"""
        results = []

        # Pre-process query once
        query_processed = self.nlp_processor.process_text(query)
        if not query_processed:
            return []

        query_words = set(query_processed.split())
        if not query_words:
            return []

        # Fast first pass: keyword matching
        candidates = []
        for i, item in enumerate(self.knowledge_data):
            question_lower = item['question'].lower()
            # Quick keyword check
            if any(word in question_lower for word in query.lower().split()[:3]):
                candidates.append((i, item))
            if len(candidates) >= 50:  # Limit candidates
                break

        # Second pass: detailed similarity on candidates only
        for i, item in candidates:
            question_similarity = self.nlp_processor.calculate_similarity(
                query, item['question']
            )

            if question_similarity >= threshold:
                results.append({
                    'question': item['question'],
                    'answer': item['answer'],
                    'category': item.get('category', 'Umum'),
                    'similarity_score': question_similarity,
                    'question_match': question_similarity,
                    'answer_match': 0.0,
                    'source': 'NLP-Fallback'
                })

                # Early termination for excellent matches
                if question_similarity >= 0.85:
                    results.sort(key=lambda x: x['similarity_score'], reverse=True)
                    return results[:5]

        # Sort and limit results
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:8]

    def _combine_results(self, vector_results: List, nlp_results: List) -> List[Dict]:
        """Combine and deduplicate results from different sources"""

        # Convert vector results to standard format
        standardized_vector = []
        for result in vector_results:
            if hasattr(result, 'metadata'):
                standardized_vector.append({
                    'question': result.metadata.get('question', ''),
                    'answer': result.metadata.get('answer', ''),
                    'category': result.metadata.get('category', 'Umum'),
                    'similarity_score': result.get('similarity_score', 0.0),
                    'source': 'Vector'
                })

        # Combine results
        all_results = standardized_vector + nlp_results

        # Remove duplicates based on question similarity
        unique_results = []
        seen_questions = set()

        for result in all_results:
            question_key = result['question'][:50]  # Use first 50 chars as key
            if question_key not in seen_questions:
                seen_questions.add(question_key)
                unique_results.append(result)

        # Sort by similarity score
        unique_results.sort(key=lambda x: x['similarity_score'], reverse=True)

        return unique_results

    async def pre_cache_popular_queries(self) -> None:
        """Pre-cache responses for popular/common queries"""
        try:
            print("🔥 Pre-caching popular queries...")

            # Define popular queries based on common user needs
            popular_queries = [
                "cara mendaftar sentuh tanahku",
                "apa itu sentuh tanahku",
                "bagaimana cek sertifikat tanah",
                "syarat pendaftaran tanah",
                "cara menggunakan aplikasi",
                "fitur utama sentuh tanahku",
                "biaya pendaftaran tanah",
                "dokumen yang diperlukan",
                "cara cek status permohonan",
                "alamat kantor pertanahan",
                "jam operasional pelayanan",
                "kontak customer service",
                "cara download sertifikat",
                "proses penerbitan sertifikat",
                "masa berlaku sertifikat"
            ]

            cached_count = 0
            for query in popular_queries:
                try:
                    # Pre-cache the query
                    await self.process_query(query)
                    cached_count += 1

                    # Small delay to avoid overwhelming the system
                    await asyncio.sleep(0.1)

                except Exception as e:
                    print(f"⚠️ Failed to pre-cache query '{query}': {e}")

            print(f"✅ Pre-cached {cached_count}/{len(popular_queries)} popular queries")

        except Exception as e:
            print(f"❌ Pre-caching failed: {e}")


# Database functions
async def load_knowledge_from_db():
    """Load knowledge data from PostgreSQL database"""
    try:
        return db.get_all_faqs()
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


# Initialize processors
nlp_processor = IndonesianNLPProcessor()
query_processor = None


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


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🚀 Enhanced Sentuh Tanahku API v2.0",
        "description": "Conversational AI dengan NLP Indonesia & Google Sheets",
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
            "POST /sync - Sync PostgreSQL database data",
            "POST /add-faq - Add single FAQ entry",
            "POST /upload-csv - Upload FAQ data from CSV file",
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

        # Log user question for analytics (removed Google Sheets dependency)

        # Process query using main processor (same as /chat endpoint)
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
    """Add a single FAQ entry and auto-sync"""
    try:
        # Add FAQ to database
        faq_id = db.add_faq(request.question, request.answer, request.category)
        
        if not faq_id:
            raise HTTPException(status_code=400, detail="Failed to add FAQ to database")
        
        # Auto-sync after adding
        await sync_data()
        
        return AddFAQResponse(
            status="success",
            message="FAQ added successfully and data synced",
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
        import io
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
        
        # Auto-sync after upload if any inserts were successful
        if result['successful'] > 0:
            print("🔄 Auto-syncing application data...")
            await sync_data()
            print("✅ Auto-sync completed")
        
        # Determine response status
        if result['successful'] == 0:
            status = "failure"
        elif result['failed'] > 0:
            status = "partial_success"
        else:
            status = "success"
        
        return UploadCSVResponse(
            status=status,
            message=f"CSV upload completed. {result['successful']} FAQs added, {result['failed']} failed from {len(faqs_data)} valid entries.",
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
            "cache_policy": "permanent (no auto-cleanup)"
        },
        "performance_optimizations": [
            "Selective caching (quality-based)",
            "Async logging queue",
            "Enhanced text caching",
            "Two-pass similarity search",
            "Increased thread workers",
            "Optimized thresholds",
            "Lazy vector embeddings"
        ],
        "cache_quality_filters": {
            "similarity_cache_min_threshold": 0.2,
            "response_cache_min_confidence": 0.2,
            "high_confidence_always_cached": "≥80% similarity"
        },
        "cache_efficiency": {
            "avg_similarity_cache_size_per_entry": format_bytes(similarity_cache_bytes / max(len(similarity_cache), 1)),
            "avg_response_cache_size_per_entry": format_bytes(response_cache_bytes / max(len(response_cache), 1)),
            "avg_text_cache_size_per_entry": format_bytes(text_cache_bytes / max(len(processed_text_cache), 1))
        }
    }


@app.post("/clear-cache")
async def clear_cache():
    """Clear all caches for fresh performance"""
    global similarity_cache, response_cache, conversation_contexts, processed_text_cache

    # Clear all caches
    similarity_cache.clear()
    response_cache.clear()
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
        "message": "All caches and queues cleared",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/download-cache-csv")
async def download_cache_csv(cache_type: str = "all"):
    """Download cache data in CSV format

    Args:
        cache_type: Type of cache to download ('similarity', 'response', 'text', 'conversations', 'all')
    """
    try:
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
            for cache_key, (response_data, timestamp) in response_cache.items():
                timestamp_str = datetime.fromtimestamp(timestamp).isoformat()
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
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )