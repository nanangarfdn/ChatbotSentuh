"""
Cache utilities and LRU cache implementation
"""
from collections import OrderedDict
from typing import Any, Optional


class LRUCache:
    """LRU Cache implementation for responses"""
    
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used
            self.cache.popitem(last=False)
        self.cache[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.put(key, value)

    def __delitem__(self, key: str) -> None:
        if key in self.cache:
            del self.cache[key]

    def __contains__(self, key: str) -> bool:
        return key in self.cache

    def clear(self) -> None:
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)

    def items(self):
        return self.cache.items()


# Global cache instances
similarity_cache = {}
processed_text_cache = {}
response_cache = LRUCache(capacity=5000)
conversation_contexts = {}