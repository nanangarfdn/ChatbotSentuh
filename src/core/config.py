"""
Configuration and feature detection
"""
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Feature availability flags
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

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    print("Warning: Sastrawi not available, using basic text processing")

try:
    from src.core.embeddings.vector_store import IndonesianVectorEmbeddings
    VECTOR_EMBEDDINGS_AVAILABLE = True
except ImportError:
    VECTOR_EMBEDDINGS_AVAILABLE = False
    print("Info: Vector embeddings not available, using NLP only")

# Performance optimization settings
CACHE_TTL = 300  # 5 minutes
THREAD_POOL_WORKERS = 8
LRU_CACHE_CAPACITY = 5000