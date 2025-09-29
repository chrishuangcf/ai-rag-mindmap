"""
Storage Manager Component

Handles all storage operations including:
- Redis caching and persistence
- Vector data storage (Redis and in-memory)
- Graph data storage coordination
- Cache lifecycle management
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from src.core.config import REDIS_AVAILABLE, LANGCHAIN_REDIS_NEW, REDIS_URL, redis_client
from src.services.cache import (
    save_cache_info, load_cache_info, is_cache_valid, 
    load_persistent_vector_store, delete_cache,
    get_vectorizer_key, get_all_cache_hashes,
    get_memory_vector_stores
)
from src.services.embeddings import SimpleTFIDFEmbeddings, SimpleInMemoryVectorStore
from src.utils.url_utils import generate_doc_hash


class StorageManager:
    """
    Manages all storage operations for the RAG system including Redis caching,
    vector storage, and data persistence.
    """
    
    def __init__(self):
        self.redis_client = redis_client
        self.memory_vector_stores = get_memory_vector_stores()
    
    async def create_vector_store(self, doc_hash: str, splits: List, embeddings: SimpleTFIDFEmbeddings) -> Tuple[Any, str]:
        """
        Create a vector store for document splits, trying Redis first then falling back to in-memory.
        
        Args:
            doc_hash: Unique hash for the document set
            splits: List of document splits
            embeddings: Fitted embeddings object
            
        Returns:
            Tuple of (vectorstore, storage_type)
        """
        vectorstore = None
        storage_type = "unknown"
        
        # Clean up any existing cache for this hash
        if REDIS_AVAILABLE and self.redis_client and self.redis_client.sismember("cache_hashes", doc_hash):
            print(f"Deleting stale cache for hash: {doc_hash}")
            await delete_cache(doc_hash)
        
        # Try Redis first
        if REDIS_AVAILABLE:
            try:
                print(f"Creating new vector store in Redis with index: {doc_hash}")
                
                if LANGCHAIN_REDIS_NEW:
                    from langchain_redis import RedisVectorStore, RedisConfig
                    config = RedisConfig(
                        index_name=doc_hash,
                        redis_url=REDIS_URL,
                        metadata_schema=[
                            {"name": "source_url", "type": "text"}
                        ]
                    )
                    vectorstore = RedisVectorStore(embeddings, config=config)
                    vectorstore.add_documents(splits)
                    storage_type = "redis_new"
                else:
                    # Use old langchain-community Redis package  
                    from langchain_community.vectorstores import Redis as LangChainRedis
                    vectorstore = LangChainRedis.from_documents(
                        documents=splits,
                        embedding=embeddings,
                        redis_url=REDIS_URL,
                        index_name=doc_hash
                    )
                    storage_type = "redis_legacy"
                    
                print(f"Successfully created Redis vector store for {len(splits)} documents")
                
            except Exception as e:
                print(f"Redis vector store creation failed: {e}")
                print("Falling back to in-memory vector store")
                vectorstore = None
        
        # Fallback to in-memory storage
        if vectorstore is None:
            print(f"Creating new in-memory vector store for hash: {doc_hash}")
            vectorstore = SimpleInMemoryVectorStore(embeddings, splits)
            self.memory_vector_stores[doc_hash] = vectorstore
            storage_type = "memory"
        
        return vectorstore, storage_type
    
    def load_vector_store(self, cache_hash: str) -> Tuple[Any, str]:
        """
        Load an existing vector store from cache.
        
        Args:
            cache_hash: Hash of the cached documents
            
        Returns:
            Tuple of (vectorstore, storage_type)
        """
        # Load embeddings first
        embeddings = SimpleTFIDFEmbeddings.from_saved(
            self.redis_client, 
            get_vectorizer_key(cache_hash)
        )
        
        # Load vector store
        vectorstore, storage_type = load_persistent_vector_store(cache_hash, embeddings)
        return vectorstore, storage_type
    
    def save_cache_metadata(self, doc_hash: str, repo_urls: List[str], total_chunks: int) -> None:
        """
        Save cache metadata and embeddings.
        
        Args:
            doc_hash: Document hash
            repo_urls: List of source URLs
            total_chunks: Number of document chunks
        """
        save_cache_info(doc_hash, repo_urls, total_chunks)
    
    def save_embeddings(self, embeddings: SimpleTFIDFEmbeddings, doc_hash: str) -> None:
        """
        Save embeddings to persistent storage.
        
        Args:
            embeddings: Fitted embeddings object
            doc_hash: Document hash for storage key
        """
        embeddings.save_vectorizer(self.redis_client, get_vectorizer_key(doc_hash))
    
    def load_cache_info(self, cache_hash: str) -> Optional[Dict]:
        """
        Load cache information.
        
        Args:
            cache_hash: Hash of the cached documents
            
        Returns:
            Cache info dictionary or None if not found
        """
        return load_cache_info(cache_hash)
    
    def update_cache_access_time(self, cache_hash: str) -> None:
        """
        Update the last accessed time for a cache.
        
        Args:
            cache_hash: Hash of the cached documents
        """
        cache_info = load_cache_info(cache_hash)
        if cache_info:
            cache_info["last_accessed"] = time.time()
            save_cache_info(
                cache_hash, 
                cache_info.get("repo_urls", []), 
                cache_info.get("total_chunks", 0)
            )
    
    def get_all_cache_hashes(self) -> List[str]:
        """
        Get all cache hashes.
        
        Returns:
            List of cache hashes
        """
        return get_all_cache_hashes()
    
    async def delete_cache(self, cache_hash: str) -> None:
        """
        Delete a cache and all associated data.
        
        Args:
            cache_hash: Hash of the cache to delete
        """
        await delete_cache(cache_hash)
    
    def is_cache_valid(self, cache_hash: str) -> bool:
        """
        Check if a cache is valid.
        
        Args:
            cache_hash: Hash of the cache to check
            
        Returns:
            True if cache is valid, False otherwise
        """
        return is_cache_valid(cache_hash)
    
    def generate_document_hash(self, repo_urls: List[str]) -> str:
        """
        Generate a consistent hash for a set of document URLs.
        
        Args:
            repo_urls: List of document URLs
            
        Returns:
            Generated hash string
        """
        return generate_doc_hash(repo_urls)
    
    def get_storage_locations(self, cache_hash: str) -> List[str]:
        """
        Determine where a cache is stored.
        
        Args:
            cache_hash: Hash of the cache
            
        Returns:
            List of storage locations (e.g., ["redis", "memory"])
        """
        locations = []
        
        # Check Redis
        if REDIS_AVAILABLE and self.redis_client:
            if self.redis_client.sismember("cache_hashes", cache_hash):
                locations.append("redis")
        
        # Check memory
        if cache_hash in self.memory_vector_stores:
            locations.append("memory")
        
        return locations if locations else ["unknown"]
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        all_hashes = self.get_all_cache_hashes()
        total_chunks = 0
        redis_caches = 0
        memory_caches = 0
        
        for cache_hash in all_hashes:
            cache_info = self.load_cache_info(cache_hash)
            if cache_info:
                total_chunks += cache_info.get("total_chunks", 0)
            
            locations = self.get_storage_locations(cache_hash)
            if "redis" in locations:
                redis_caches += 1
            if "memory" in locations:
                memory_caches += 1
        
        return {
            "total_caches": len(all_hashes),
            "total_chunks": total_chunks,
            "redis_caches": redis_caches,
            "memory_caches": memory_caches,
            "redis_available": REDIS_AVAILABLE
        }
