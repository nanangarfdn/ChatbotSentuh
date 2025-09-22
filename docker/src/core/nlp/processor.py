"""
Indonesian NLP text processing with advanced features
"""
import re
import time
from typing import List, Dict, Set
from collections import defaultdict

from src.core.config import SASTRAWI_AVAILABLE, BLOOM_FILTER_AVAILABLE
from src.utils.cache import processed_text_cache, similarity_cache

if SASTRAWI_AVAILABLE:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

if BLOOM_FILTER_AVAILABLE:
    from pybloom_live import BloomFilter


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

        return similarity