# cache.py
import redis
import pickle
import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0)

def set_cached_data(key: str, data: object):
    """Serializes a Python object using pickle and stores it in Redis."""
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