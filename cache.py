# cache.py
import redis
import pickle
import os
import logging

# Get logger for this module
logger = logging.getLogger("cache")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0)

def set_cached_data(key: str, data: object):
    """Serializes a Python object using pickle and stores it in Redis."""
    logger.info(f"Caching data for key: {key[:50]}...")
    
    try:
        start_time = __import__('time').time()
        
        # Use the highest pickle protocol for better performance
        serialized_object = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        serialized_size = len(serialized_object)
        
        # Store with TTL of 1 hour (3600 seconds)
        redis_client.setex(key, 3600, serialized_object)
        
        end_time = __import__('time').time()
        logger.info(f"Successfully cached {serialized_size/1024:.1f} KB in {end_time - start_time:.3f} seconds")
        
    except redis.RedisError as e:
        logger.error(f"Redis error while caching data: {str(e)}")
    except pickle.PickleError as e:
        logger.error(f"Pickle serialization error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during caching: {str(e)}")

def get_cached_data(key: str) -> object | None:
    """Retrieves and deserializes an object from Redis."""
    logger.info(f"Retrieving cached data for key: {key[:50]}...")
    
    try:
        start_time = __import__('time').time()
        
        # Check if key exists before retrieving
        if not redis_client.exists(key):
            logger.info(f"Cache miss: Key not found")
            return None
        
        # Get the cached object
        cached_object = redis_client.get(key)
        
        if cached_object:
            # Measure size of the cached data
            cached_size = len(cached_object)
            logger.debug(f"Found {cached_size/1024:.1f} KB of cached data")
            
            # Load the data
            data = pickle.loads(cached_object)
            
            end_time = __import__('time').time()
            logger.info(f"Cache hit: Data retrieved in {end_time - start_time:.3f} seconds")
            
            # Try to log some info about what was retrieved
            if isinstance(data, tuple) and len(data) == 2:
                logger.debug(f"Retrieved tuple with {len(data[1])} chunks")
                
            return data
            
        else:
            logger.warning("Empty cache result despite key existence check")
            return None
            
    except redis.RedisError as e:
        logger.error(f"Redis error during retrieval: {str(e)}")
    except pickle.PickleError as e:
        logger.error(f"Error deserializing cached object: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error retrieving from cache: {str(e)}")
        
    return None