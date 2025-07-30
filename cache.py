# cache.py
import redis
import pickle
from contextlib import contextmanager

# Establish a connection to the Redis server.
# Assumes Redis is running on localhost:6379.
# The 'decode_responses=False' is important because we will handle
# serialization with pickle, which works with bytes.
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

@contextmanager
def get_redis_connection():
    """Provides a managed connection to Redis."""
    try:
        redis_client.ping() # Check if the connection is alive
        yield redis_client
    except redis.exceptions.ConnectionError as e:
        print(f"Could not connect to Redis: {e}")
        # Yield None so the application can proceed without caching
        # in case Redis is down.
        yield None

def set_vector_store(key: str, value: object):
    """
    Serializes a Python object using pickle and stores it in Redis.
    """
    with get_redis_connection() as conn:
        if conn:
            try:
                # Serialize the object to a byte string
                serialized_object = pickle.dumps(value)
                # Set the key-value pair with a time-to-live (TTL) of 1 hour (3600 seconds)
                conn.setex(key, 3600, serialized_object)
                print(f"Cached object with key: {key}")
            except Exception as e:
                print(f"Failed to cache object: {e}")

def get_vector_store(key: str) -> object:
    """
    Retrieves and deserializes an object from Redis.
    """
    with get_redis_connection() as conn:
        if conn:
            try:
                # Retrieve the byte string from Redis
                cached_object = conn.get(key)
                if cached_object:
                    print(f"Found object in cache for key: {key}")
                    # Deserialize the byte string back into a Python object
                    return pickle.loads(cached_object)
            except Exception as e:
                print(f"Failed to retrieve object from cache: {e}")
        return None