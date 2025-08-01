# cache.py
import redis
import pickle
from config import REDIS_URL

redis_client = redis.from_url(REDIS_URL, decode_responses=False)

def set_cached_data(key: str, data: object):
    """Serializes a Python object and stores it in Redis."""
    try:
        serialized_object = pickle.dumps(data)
        redis_client.setex(key, 3600, serialized_object)
        print(f"Successfully cached data for key: {key}")
    except Exception as e:
        print(f"Failed to cache data: {e}")

def get_cached_data(key: str) -> object | None:
    """Retrieves and deserializes an object from Redis."""
    try:
        cached_object = redis_client.get(key)
        if cached_object:
            print(f"Found cached data for key: {key}")
            return pickle.loads(cached_object)
    except Exception as e:
        print(f"Failed to retrieve data from cache: {e}")
    return None