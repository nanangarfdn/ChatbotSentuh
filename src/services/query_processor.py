"""
Enhanced query processing system combining multiple approaches
"""
import asyncio
from datetime import datetime
from typing import List, Dict, Set, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from src.core.config import VECTOR_EMBEDDINGS_AVAILABLE
from src.core.nlp.processor import IndonesianNLPProcessor
from src.core.nlp.conversation import ConversationHandler
from src.core.nlp.confidence import ConfidenceEvaluator
from src.services.response_generator import ResponseGenerator
from src.database.connection import db

# Try to import vector embeddings
try:
    from src.core.embeddings.vector_store import IndonesianVectorEmbeddings
except ImportError:
    IndonesianVectorEmbeddings = None

thread_executor = ThreadPoolExecutor(max_workers=8)


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
        if VECTOR_EMBEDDINGS_AVAILABLE and knowledge_data and IndonesianVectorEmbeddings:
            print("🚀 Starting background vector embeddings initialization...")
            # Note: Create task when event loop is available

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
        """Main query processing with multi-layer approach - no query caching"""
        start_time = datetime.now()

        # Skip query caching - always process fresh using synced knowledge data

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
            
            # Save user query to database based on confidence score (async, non-blocking)
            asyncio.create_task(self._save_user_query_async(query, response_data))
            
            # No caching - always use fresh synced data
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

        # Save user query to database based on confidence score (async, non-blocking)
        asyncio.create_task(self._save_user_query_async(query, response_data))

        # No query response caching - always use fresh synced knowledge data
        return response_data

    async def _ensure_vector_embeddings_async(self) -> bool:
        """Ensure vector embeddings are initialized (lazy loading)"""
        if self._vector_embeddings_failed:
            return False

        if self._vector_embeddings_initialized:
            return True

        if not VECTOR_EMBEDDINGS_AVAILABLE or not IndonesianVectorEmbeddings:
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
        """Pre-caching disabled - system now uses fresh synced data only"""
        print("ℹ️ Pre-caching disabled - system uses fresh synced knowledge data for all queries")

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
            import json
            suggestions_json = json.dumps(response['suggestions'])
            yield f"data: {{\"type\": \"suggestions\", \"suggestions\": {suggestions_json}}}\n\n"

        # Send sources if available
        if response.get('sources'):
            import json
            sources_json = json.dumps(response['sources'])
            yield f"data: {{\"type\": \"sources\", \"sources\": {sources_json}}}\n\n"

        yield "data: {\"type\": \"done\"}\n\n"

    async def _save_user_query_async(self, query: str, response_data: Dict) -> None:
        """
        Save user query to database asynchronously based on confidence score
        
        Args:
            query: User's original question
            response_data: Response data containing confidence score and answer
        """
        try:
            # Skip saving for greeting/goodbye responses (not actual FAQ content)
            response_type = response_data.get('response_type', '')
            if response_type in ['greeting', 'goodbye']:
                return
            
            # Extract data for saving
            confidence_score = response_data.get('confidence_score', 0.0)
            answer = response_data.get('response', '')
            
            # Skip if no meaningful content
            if not query.strip() or not answer.strip():
                return
            
            # Skip if confidence is exactly 1.0 (likely system generated response)
            if confidence_score == 1.0 and response_type in ['greeting', 'goodbye']:
                return
            
            # Save to database in background thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                thread_executor,
                lambda: db.add_user_query(query, answer, confidence_score)
            )
            
        except Exception as e:
            # Log error but don't fail the main response
            print(f"⚠️ Failed to save user query: {e}")
            # Could add proper logging here if needed