import redis
import pickle
import logging
from config import REDIS_URL, REDIS_DISABLED

logger = logging.getLogger("cache")

if not REDIS_DISABLED:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=False)
        logger.info(f"Redis client initialized with URL: {REDIS_URL}")
    except Exception as e:
        logger.error(f"Failed to initialize Redis client: {str(e)}")
        redis_client = None
else:
    logger.info("Redis is disabled by configuration")
    redis_client = None

def set_cached_data(key: str, data: object):
    """Serializes a Python object using pickle and stores it in Redis."""
    if REDIS_DISABLED or redis_client is None:
        logger.info("Cache write skipped (Redis disabled)")
        return
        
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

def get_cached_data(key: str) -> object:
    """Retrieves and deserializes an object from Redis."""
    if REDIS_DISABLED or redis_client is None:
        logger.info("Cache read skipped (Redis disabled)")
        return None
        
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
            if isinstance(data, tuple) and len(data) >= 2:
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