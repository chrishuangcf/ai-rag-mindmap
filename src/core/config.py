"""
Configuration and environment setup for the LangChain RAG API.
Handles Redis connections, environment variables, and global settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- LLM Provider Configuration ---
# Choose between "openrouter" (default) and "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()

# --- OpenRouter/OpenAI Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-r1:free")
DEFAULT_REFERER = os.getenv("DEFAULT_REFERER", "https://localhost:8000")
DEFAULT_TITLE = os.getenv("DEFAULT_TITLE", "LangChain RAG API")

# --- Ollama Configuration ---
# Use host.docker.internal for containerized environments to connect to the host
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


# --- Redis Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}"

# Global flags for feature availability
GOOGLE_SEARCH_AVAILABLE = False
REDIS_AVAILABLE = False
REDIS_SEARCH_AVAILABLE = False
LANGCHAIN_REDIS_NEW = False

# Google search setup
try:
    from googlesearch import search as google_search
    GOOGLE_SEARCH_AVAILABLE = True
    print("Google search available for fallback")
except ImportError:
    GOOGLE_SEARCH_AVAILABLE = False
    print("Google search not available - install googlesearch-python for fallback functionality")

# Redis setup
redis_client = None
redis_client_str = None

try:
    import redis
    # Try the new langchain-redis package first
    try:
        from langchain_redis import RedisVectorStore, RedisConfig
        LANGCHAIN_REDIS_NEW = True
        print("Using new langchain-redis package")
    except ImportError:
        # Fallback to older community package
        from langchain_community.vectorstores import Redis as LangChainRedis
        LANGCHAIN_REDIS_NEW = False
        print("Using older langchain-community Redis")
    
    REDIS_AVAILABLE = True
    
    # Try to import Redis search modules
    try:
        from redis.commands.search.field import VectorField, TextField, NumericField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        REDIS_SEARCH_AVAILABLE = True
        print("Redis with search modules available")
    except ImportError as e:
        print(f"Redis search modules not available: {e}")
        REDIS_SEARCH_AVAILABLE = False
        
except ImportError as e:
    print(f"Redis not available: {e}")
    REDIS_AVAILABLE = False
    REDIS_SEARCH_AVAILABLE = False
    LANGCHAIN_REDIS_NEW = False

# Initialize Redis connections if available
if REDIS_AVAILABLE:
    try:
        redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
        redis_client_str = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        
        # Test connection
        redis_client.ping()
        print(f"Connected to Redis at {REDIS_URL}")
        
        # Check Redis modules
        modules = redis_client.module_list()
        search_loaded = any('search' in str(module).lower() for module in modules)
        json_loaded = any('json' in str(module).lower() for module in modules)
        
        print(f"Redis modules - Search: {search_loaded}, JSON: {json_loaded}")
        
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        redis_client = None
        redis_client_str = None
        REDIS_AVAILABLE = False

def get_redis_clients():
    """Get Redis clients (binary and string decoders)"""
    return redis_client, redis_client_str

def is_redis_available():
    """Check if Redis is available"""
    return REDIS_AVAILABLE

def is_google_search_available():
    """Check if Google search is available"""
    return GOOGLE_SEARCH_AVAILABLE
