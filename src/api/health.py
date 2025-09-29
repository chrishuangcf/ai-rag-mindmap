"""
Health check API endpoints.
"""

from fastapi import APIRouter
from src.core.config import REDIS_AVAILABLE, REDIS_SEARCH_AVAILABLE, GOOGLE_SEARCH_AVAILABLE
from src.services.cache import get_all_cache_hashes, get_memory_cache_info, get_memory_vector_stores

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint for the web UI."""
    cache_hashes = get_all_cache_hashes()
    memory_cache_info = get_memory_cache_info()
    memory_vector_stores = get_memory_vector_stores()
    
    # Count Redis vs in-memory sources
    redis_sources = 0
    in_memory_sources = len(memory_vector_stores)
    
    if REDIS_AVAILABLE:
        redis_sources = len(cache_hashes) - in_memory_sources
        if redis_sources < 0:
            redis_sources = 0
    
    return {
        "status": "healthy",
        "redis_available": REDIS_AVAILABLE,
        "redis_search_available": REDIS_SEARCH_AVAILABLE,
        "google_search_available": GOOGLE_SEARCH_AVAILABLE,
        "in_memory_cache_active": len(memory_cache_info) > 0,
        "cache_info": {
            "redis_sources": redis_sources,
            "in_memory_sources": in_memory_sources,
            "total_sources": len(cache_hashes)
        }
    }
