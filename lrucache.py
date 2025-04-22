from cachetools import LRUCache
import sqlglot
import hashlib

# Create an LRU cache with 100 max entries
cache = LRUCache(maxsize=128)

# Normalize + hash the query to get a consistent cache key
def get_normalized_cache_key(query):
    try:
        parsed = sqlglot.parse_one(query)
        normalized = parsed.sql(dialect="postgres")
    except Exception as e:
        print("Warning: Failed to parse query. Falling back to lower+strip:", e)
        normalized = query.strip().lower()
    
    return hashlib.md5(normalized.encode()).hexdigest()


