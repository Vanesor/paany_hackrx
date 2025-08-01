# cache.py
import redis
import pickle
import logging
import time
from config import REDIS_URL

logger = logging.getLogger(__name__)
redis_client = redis.from_url(REDIS_URL, decode_responses=False)

def set_cached_data(key: str, data: object):
    """Serializes a Python object and stores it in Redis."""
    start_time = time.time()
    try:
        serialized_object = pickle.dumps(data)
        data_size = len(serialized_object)
        redis_client.setex(key, 3600, serialized_object)
        cache_time = time.time() - start_time
        logger.info(f"💾 Cached {data_size} bytes for key: {key[:20]}... in {cache_time:.3f}s")
    except Exception as e:
        cache_time = time.time() - start_time
        logger.error(f"❌ Failed to cache data after {cache_time:.3f}s: {e}")

def get_cached_data(key: str) -> object | None:
    """Retrieves and deserializes an object from Redis."""
    start_time = time.time()
    try:
        cached_object = redis_client.get(key)
        if cached_object:
            data = pickle.loads(cached_object)
            cache_time = time.time() - start_time
            data_size = len(cached_object)
            logger.info(f"⚡ Retrieved {data_size} bytes from cache for key: {key[:20]}... in {cache_time:.3f}s")
            return data
        else:
            cache_time = time.time() - start_time
            logger.debug(f"🚫 Cache miss for key: {key[:20]}... (checked in {cache_time:.3f}s)")
    except Exception as e:
        cache_time = time.time() - start_time
        logger.error(f"❌ Failed to retrieve data from cache after {cache_time:.3f}s: {e}")
    return None