"""
Cache management API endpoints.
"""

import time
import json
from fastapi import APIRouter, HTTPException
from src.services.cache import (
    get_all_cache_hashes, load_cache_info, delete_cache,
    get_memory_cache_info, get_memory_vector_stores
)
from src.core.config import REDIS_AVAILABLE, redis_client, redis_client_str

router = APIRouter(prefix="/cache")

@router.get("/list")
async def list_cache_entries():
    """List all cache entries for the web UI."""
    try:
        cache_hashes = get_all_cache_hashes()
        cache_entries = []
        
        for cache_hash in cache_hashes:
            cache_entry = _build_cache_entry(cache_hash)
            if cache_entry:
                cache_entries.append(cache_entry)
        
        # Sort by last accessed time (most recent first)
        cache_entries.sort(key=lambda x: x.get("last_accessed", 0), reverse=True)
        
        return {
            "available_caches": cache_entries,
            "total_caches": len(cache_entries),
            "redis_available": REDIS_AVAILABLE
        }
    except Exception as e:
        print(f"Error listing cache entries: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing cache entries: {str(e)}")

def _build_cache_entry(cache_hash: str):
    """Build a cache entry dictionary for the given hash."""
    cache_info = load_cache_info(cache_hash)
    if not cache_info:
        return None
    
    # Determine storage type
    storage_type = _get_storage_locations(cache_hash)
    
    # Create URL preview (first few URLs)
    urls = cache_info.get("repo_urls", [])
    url_preview = urls[:3] if len(urls) > 3 else urls
    
    return {
        "hash": cache_hash,
        "urls": urls,
        "url_preview": url_preview,
        "total_chunks": cache_info.get("total_chunks", 0),
        "created_at": cache_info.get("created_at", 0),
        "last_accessed": cache_info.get("last_accessed", 0),
        "storage_locations": storage_type
    }

def _get_storage_locations(cache_hash: str):
    """Determine storage locations for a cache hash."""
    storage_type = []
    
    if REDIS_AVAILABLE and redis_client_str:
        if redis_client_str.sismember("cache_hashes", cache_hash):
            storage_type.append("Redis")
    
    memory_stores = get_memory_vector_stores()
    if cache_hash in memory_stores:
        storage_type.append("In-Memory")
    
    return storage_type if storage_type else ["Unknown"]

@router.delete("/{cache_hash}")
async def delete_cache_entry(cache_hash: str):
    """Delete a specific cache entry."""
    try:
        cache_info = load_cache_info(cache_hash)
        if not cache_info:
            raise HTTPException(status_code=404, detail=f"Cache entry {cache_hash} not found")
        
        await delete_cache(cache_hash)
        
        return {
            "message": f"Cache entry {cache_hash} deleted successfully",
            "deleted_urls": cache_info.get("repo_urls", [])
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error deleting cache entry {cache_hash}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting cache entry: {str(e)}")

@router.delete("")
async def clear_all_caches():
    """Clear all cache entries."""
    try:
        cache_hashes = get_all_cache_hashes()
        deleted_count = 0
        
        for cache_hash in cache_hashes:
            try:
                await delete_cache(cache_hash)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting cache {cache_hash}: {e}")
                continue
        
        return {
            "message": f"Successfully deleted {deleted_count} cache entries",
            "deleted_count": deleted_count
        }
    except Exception as e:
        print(f"Error clearing all caches: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing caches: {str(e)}")

@router.get("/{cache_hash}/vector_data")
async def get_cache_vector_data(cache_hash: str):
    """Get all vector data for a specific cache from Redis"""
    try:
        if not redis_client:
            raise HTTPException(status_code=503, detail="Redis client not available")

        # Get all document keys from Redis
        keys = redis_client.keys(f"{cache_hash}:*")
        if not keys:
            raise HTTPException(status_code=404, detail=f"No vector data found for cache {cache_hash}")

        # Fetch all documents
        documents = []
        for key in keys:
            data = redis_client.hgetall(key)
            documents.append({
                "id": key.decode("utf-8"),
                "content": data.get(b"content", b"").decode("utf-8"),
                "vector": data.get(b"vector", b"").decode("utf-8"),
                "metadata": json.loads(data.get(b"metadata", b"{}").decode("utf-8"))
            })

        return {
            "cache_hash": cache_hash,
            "documents": documents,
            "total_documents": len(documents)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving vector data: {str(e)}")
