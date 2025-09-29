"""
Debug and test API endpoints.
"""

import os
import time
import requests
from fastapi import APIRouter, HTTPException

from src.core.models import SimpleRAGRequest
from src.utils.url_utils import validate_and_convert_urls, generate_doc_hash
from src.services.cache import get_all_cache_hashes, load_cache_info, get_vectorizer_key, get_memory_vector_stores
from src.services.embeddings import SimpleTFIDFEmbeddings
from src.services.cache import load_persistent_vector_store
from src.services.google_search import perform_google_search, get_google_search_summary
from src.core.config import REDIS_AVAILABLE, redis_client

router = APIRouter()

@router.post("/test/url-conversion")
async def test_url_conversion(request: SimpleRAGRequest):
    """Test endpoint to check URL conversion."""
    try:
        repo_urls = validate_and_convert_urls(request.repo_urls)
        doc_hash = generate_doc_hash(repo_urls)
        
        return {
            "original_urls": request.repo_urls,
            "converted_urls": repo_urls,
            "doc_hash": doc_hash,
            "redis_available": REDIS_AVAILABLE,
            "message": "URL conversion successful"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/test/fetch-document")
async def test_fetch_document(request: SimpleRAGRequest):
    """Test endpoint to check document fetching."""
    try:
        repo_urls = validate_and_convert_urls(request.repo_urls)
        
        if not repo_urls:
            return {"error": "No URLs provided"}
        
        # Test fetching just the first URL
        test_url = repo_urls[0]
        response = requests.get(test_url)
        response.raise_for_status()
        
        content_preview = response.text[:500] + "..." if len(response.text) > 500 else response.text
        
        return {
            "url": test_url,
            "status_code": response.status_code,
            "content_length": len(response.text),
            "content_preview": content_preview,
            "message": "Document fetch successful"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/test/simple")
async def test_simple():
    """Simple test endpoint."""
    return {"message": "Simple test works", "timestamp": time.time()}

@router.post("/test/env-check")
async def test_env_check():
    """Test endpoint to check environment variables."""
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_base = os.getenv("OPENAI_API_BASE")
        
        return {
            "openai_api_key_set": bool(openai_api_key),
            "openai_api_base_set": bool(openai_api_base),
            "openai_api_key_length": len(openai_api_key) if openai_api_key else 0,
            "openai_api_base": openai_api_base,
            "message": "Environment check completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/debug/cache-status")
async def debug_cache_status():
    """Debug endpoint to check cache status."""
    try:
        cache_hashes = get_all_cache_hashes()
        memory_vector_stores = get_memory_vector_stores()
        
        cache_status = {
            "redis_available": REDIS_AVAILABLE,
            "total_memory_caches": len(memory_vector_stores),
            "memory_cache_hashes": list(cache_hashes) if not REDIS_AVAILABLE else [],
            "memory_vector_stores": list(memory_vector_stores.keys())
        }
        
        if REDIS_AVAILABLE and redis_client:
            try:
                cache_status["redis_cache_hashes"] = list(cache_hashes)
                cache_status["redis_connection"] = "active"
            except Exception as e:
                cache_status["redis_connection"] = f"error: {e}"
        else:
            cache_status["redis_connection"] = "unavailable"
        
        # Check memory cache details
        memory_cache_details = []
        for cache_hash in cache_hashes:
            cache_info = load_cache_info(cache_hash)
            if cache_info:
                memory_cache_details.append({
                    "hash": cache_hash,
                    "urls": cache_info.get("repo_urls", []),
                    "chunks": cache_info.get("total_chunks", 0),
                    "created": cache_info.get("created_at", 0)
                })
        
        cache_status["cache_details"] = memory_cache_details
        return cache_status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/debug/test-search")
async def debug_test_search(request: dict):
    """Debug endpoint to test search functionality on a specific cache."""
    try:
        cache_hash = request.get("cache_hash")
        query = request.get("query", "test query")
        
        if not cache_hash:
            return {"error": "cache_hash required"}
        
        cache_info = load_cache_info(cache_hash)
        if not cache_info:
            return {"error": f"Cache {cache_hash} not found"}
        
        try:
            embeddings = SimpleTFIDFEmbeddings.from_saved(redis_client, get_vectorizer_key(cache_hash))
            vectorstore, storage_type = load_persistent_vector_store(cache_hash, embeddings)
            
            if not vectorstore:
                return {"error": f"No vector store found for {cache_hash}"}
            
            results = vectorstore.similarity_search(query, k=3)
            
            return {
                "cache_hash": cache_hash,
                "query": query,
                "storage_type": storage_type,
                "results_count": len(results),
                "results": [
                    {
                        "content_preview": doc.page_content[:200],
                        "metadata": doc.metadata
                    }
                    for doc in results
                ],
                "message": "Search test completed"
            }
            
        except Exception as e:
            return {"error": f"Search test failed: {str(e)}"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/test/google-fallback")
async def test_google_fallback(request: dict):
    """Test endpoint to verify Google search fallback functionality."""
    try:
        query = request.get("query", "test query")
        num_results = request.get("num_results", 3)
        
        # Test Google search
        google_results = perform_google_search(query, num_results)
        
        if not google_results:
            return {
                "query": query,
                "google_results": [],
                "message": "Google search returned no results"
            }
        
        # Test AI summary generation
        try:
            summary = get_google_search_summary(query, google_results)
            return {
                "query": query,
                "google_results_count": len(google_results),
                "google_results": google_results,
                "ai_summary": summary,
                "message": "Google fallback test completed successfully"
            }
        except Exception as summary_error:
            return {
                "query": query,
                "google_results_count": len(google_results),
                "google_results": google_results,
                "ai_summary_error": str(summary_error),
                "message": "Google search worked but AI summary failed"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
