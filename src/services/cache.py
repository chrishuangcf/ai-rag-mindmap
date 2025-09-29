"""
Cache management utilities for the LangChain RAG API.
Handles Redis and in-memory cache operations.
"""

import json
import time
import traceback
from src.core.config import redis_client, redis_client_str, REDIS_AVAILABLE, LANGCHAIN_REDIS_NEW, REDIS_URL

# In-memory cache fallback
_memory_cache_info = {}
_memory_cache_hashes = set()
_memory_vector_stores = {}

def get_info_key(doc_hash: str) -> str:
    """Generate Redis key for cache info."""
    return f"cache:info:{doc_hash}"

def get_vectorizer_key(doc_hash: str) -> str:
    """Generate Redis key for vectorizer."""
    return f"cache:vectorizer:{doc_hash}"

def save_cache_info(doc_hash: str, repo_urls: list[str], total_chunks: int):
    """Save cache information to Redis or in-memory storage."""
    cache_info = {
        "doc_hash": doc_hash,
        "repo_urls": repo_urls,
        "total_chunks": total_chunks,
        "created_at": time.time(),
        "last_accessed": time.time()
    }
    
    if REDIS_AVAILABLE and redis_client_str:
        redis_client_str.sadd("cache_hashes", doc_hash)
        redis_client_str.set(get_info_key(doc_hash), json.dumps(cache_info))
    else:
        # Fallback to in-memory cache
        _memory_cache_hashes.add(doc_hash)
        _memory_cache_info[doc_hash] = cache_info

def load_cache_info(doc_hash: str) -> dict | None:
    """Load cache information from Redis or in-memory storage."""
    if REDIS_AVAILABLE and redis_client_str:
        info_json = redis_client_str.get(get_info_key(doc_hash))
        return json.loads(info_json) if info_json else None
    else:
        # Fallback to in-memory cache
        return _memory_cache_info.get(doc_hash)

def is_cache_valid(doc_hash: str, repo_urls: list[str]) -> bool:
    """Check if cache is valid for the given document hash and URLs."""
    if REDIS_AVAILABLE and redis_client_str:
        if not redis_client_str.sismember("cache_hashes", doc_hash):
            return False
        
        cache_info = load_cache_info(doc_hash)
        if not cache_info:
            return False

        if not redis_client.exists(get_vectorizer_key(doc_hash)):
            return False

        # Check if the index exists in Redis
        if not redis_client.keys(f"doc:{doc_hash}:*"):
             return False

        if set(cache_info.get('repo_urls', [])) != set(repo_urls):
            return False
            
        return True
    else:
        # Fallback to in-memory cache validation
        if doc_hash not in _memory_cache_hashes:
            return False
        
        cache_info = load_cache_info(doc_hash)
        if not cache_info:
            return False
        
        # Check if vectorizer exists in memory cache
        from src.services.embeddings import SimpleTFIDFEmbeddings
        vectorizer_key = get_vectorizer_key(doc_hash)
        if not hasattr(SimpleTFIDFEmbeddings, '_memory_cache') or vectorizer_key not in SimpleTFIDFEmbeddings._memory_cache:
            return False
        
        # Check if vector store exists in memory
        if doc_hash not in _memory_vector_stores:
            return False
        
        if set(cache_info.get('repo_urls', [])) != set(repo_urls):
            return False
            
        return True

def load_persistent_vector_store(doc_hash: str, embeddings):
    """Load a persistent vector store for the given cache hash."""
    print(f"Loading persistent vector store for hash: {doc_hash}")
    print(f"REDIS_AVAILABLE: {REDIS_AVAILABLE}, LANGCHAIN_REDIS_NEW: {LANGCHAIN_REDIS_NEW}")
    
    if REDIS_AVAILABLE:
        try:
            if LANGCHAIN_REDIS_NEW:
                # Use new langchain-redis package
                from langchain_redis import RedisVectorStore, RedisConfig
                print(f"Attempting to load Redis vector store using new langchain-redis package")
                config = RedisConfig(
                    index_name=doc_hash,
                    redis_url=REDIS_URL,
                    metadata_schema=[
                        {"name": "source_url", "type": "text"}
                    ]
                )
                vectorstore = RedisVectorStore(embeddings, config=config)
                print(f"Successfully created Redis vector store")
                
                # Try to get document count or verify the store has content
                try:
                    # Test search to see if documents exist
                    test_results = vectorstore.similarity_search("test", k=1)
                    print(f"Test search returned {len(test_results)} documents")
                    
                    # Try to get some basic info about the vector store
                    if hasattr(vectorstore, 'client'):
                        print(f"Vector store client available")
                    
                except Exception as verify_error:
                    print(f"Error verifying vector store content: {verify_error}")
                
                return vectorstore, "redis"
            else:
                # Use old langchain-community Redis package
                from langchain_community.vectorstores import Redis as LangChainRedis
                print(f"Attempting to load Redis vector store using old langchain-community package")
                vectorstore = LangChainRedis.from_existing_index(
                    embedding=embeddings,
                    index_name=doc_hash,
                    redis_url=REDIS_URL
                )
                print(f"Successfully loaded Redis vector store from existing index")
                return vectorstore, "redis"
        except Exception as e:
            print(f"Failed to load Redis vector store: {e}")
            print(f"Redis vector store error traceback: {traceback.format_exc()}")
    
    # Fallback to in-memory
    print(f"Checking in-memory vector stores. Available keys: {list(_memory_vector_stores.keys())}")
    if doc_hash in _memory_vector_stores:
        print(f"Found in-memory vector store for hash: {doc_hash}")
        vectorstore = _memory_vector_stores[doc_hash]
        print(f"In-memory vector store has {len(vectorstore.documents) if hasattr(vectorstore, 'documents') else 'unknown'} documents")
        return vectorstore, "memory"
    
    print(f"No vector store found for hash: {doc_hash}")
    return None, None

async def delete_cache(doc_hash: str):
    """Delete cache data for the given hash."""
    if REDIS_AVAILABLE and redis_client_str:
        try:
            # Delete using the correct key patterns
            redis_client_str.delete(get_info_key(doc_hash))
            redis_client.delete(get_vectorizer_key(doc_hash))
            redis_client_str.srem("cache_hashes", doc_hash)
            
        except Exception as e:
            print(f"Failed to delete Redis cache: {e}")
    
    # Remove from memory cache
    if doc_hash in _memory_vector_stores:
        del _memory_vector_stores[doc_hash]
    
    if doc_hash in _memory_cache_hashes:
        _memory_cache_hashes.remove(doc_hash)
    
    if doc_hash in _memory_cache_info:
        del _memory_cache_info[doc_hash]

def get_all_cache_hashes():
    """Get all available cache hashes."""
    if REDIS_AVAILABLE and redis_client_str:
        return redis_client_str.smembers("cache_hashes")
    else:
        return _memory_cache_hashes

def get_memory_vector_stores():
    """Get the memory vector stores dict."""
    return _memory_vector_stores

def get_memory_cache_info():
    """Get the memory cache info dict."""
    return _memory_cache_info

async def verify_and_recreate_redis_indices():
    """Verify Redis search indices exist and recreate if necessary."""
    if not REDIS_AVAILABLE or not redis_client:
        return
    
    try:
        cache_hashes = get_all_cache_hashes()
        if not cache_hashes:
            print("No cache hashes found, skipping index verification")
            return
        
        print(f"Verifying Redis search indices for {len(cache_hashes)} caches...")
        
        for doc_hash in cache_hashes:
            try:
                # Check if the search index exists
                index_exists = False
                try:
                    # Try to get index info
                    redis_client.ft(doc_hash).info()
                    index_exists = True
                    print(f"✅ Index {doc_hash} exists")
                except Exception:
                    print(f"❌ Index {doc_hash} missing - needs recreation")
                
                if not index_exists:
                    print(f"🔄 Index {doc_hash} will be recreated when documents are accessed")
                    # Note: We mark indices as needing recreation rather than recreating now
                    # because we need the original documents and embeddings to recreate properly
                    
            except Exception as e:
                print(f"Error checking index {doc_hash}: {e}")
        
        print("Redis index verification completed")
        
    except Exception as e:
        print(f"Error during Redis index verification: {e}")
