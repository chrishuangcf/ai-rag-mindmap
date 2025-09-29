"""
Mind Map Integration API endpoints for the main RAG service.
These endpoints provide cache data to the mind map service.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
import json

from src.services.cache import load_cache_info, get_all_cache_hashes, get_vectorizer_key, load_persistent_vector_store
from src.services.embeddings import SimpleTFIDFEmbeddings
from src.core.config import redis_client


mindmap_integration_router = APIRouter()


@mindmap_integration_router.get("/cache/{cache_hash}/info")
async def get_cache_info(cache_hash: str):
    """Get detailed information about a cache including documents and metadata"""
    try:
        cache_info = load_cache_info(cache_hash)
        if not cache_info:
            raise HTTPException(status_code=404, detail=f"Cache {cache_hash} not found")
        
        return {
            "cache_hash": cache_hash,
            "repo_urls": cache_info.get("repo_urls", []),
            "total_chunks": cache_info.get("total_chunks", 0),
            "created_at": cache_info.get("created_at"),
            "last_accessed": cache_info.get("last_accessed"),
            "documents": cache_info.get("documents", []),
            "metadata": cache_info.get("metadata", [])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving cache info: {str(e)}")


@mindmap_integration_router.get("/cache/{cache_hash}/embeddings")
async def get_cache_embeddings(cache_hash: str):
    """Get document embeddings for a specific cache"""
    try:
        cache_info = load_cache_info(cache_hash)
        if not cache_info:
            raise HTTPException(status_code=404, detail=f"Cache {cache_hash} not found")
        
        # Load the vectorizer/embeddings
        try:
            embeddings = SimpleTFIDFEmbeddings.from_saved(redis_client, get_vectorizer_key(cache_hash))
            
            # Get the embeddings matrix if available
            if hasattr(embeddings, 'get_embeddings_matrix'):
                embeddings_matrix = embeddings.get_embeddings_matrix()
                embeddings_list = embeddings_matrix.tolist() if embeddings_matrix is not None else []
            else:
                # Fallback: get embeddings for each document
                documents = cache_info.get("documents", [])
                embeddings_list = []
                for doc in documents:
                    try:
                        doc_embedding = embeddings.embed_query(doc.get("content", ""))
                        embeddings_list.append(doc_embedding)
                    except Exception:
                        embeddings_list.append([])
            
            return {
                "cache_hash": cache_hash,
                "embeddings": embeddings_list,
                "embedding_dimension": len(embeddings_list[0]) if embeddings_list else 0,
                "total_documents": len(embeddings_list)
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading embeddings: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving cache embeddings: {str(e)}")


@mindmap_integration_router.get("/cache/{cache_hash}/documents")
async def get_cache_documents(cache_hash: str, include_content: bool = True, limit: Optional[int] = None):
    """Get documents from a specific cache with optional content"""
    try:
        print(f"🔍 DEBUG: Getting documents for cache {cache_hash}, limit={limit}")
        
        cache_info = load_cache_info(cache_hash)
        if not cache_info:
            print(f"❌ DEBUG: Cache {cache_hash} not found in cache info")
            raise HTTPException(status_code=404, detail=f"Cache {cache_hash} not found")
        
        print(f"✅ DEBUG: Cache info loaded: {cache_info.keys()}")
        
        # Load the vector store to get actual documents
        try:
            print(f"🔄 DEBUG: Loading embeddings for cache {cache_hash}")
            embeddings = SimpleTFIDFEmbeddings.from_saved(redis_client, get_vectorizer_key(cache_hash))
            print(f"✅ DEBUG: Embeddings loaded successfully")
            
            print(f"🔄 DEBUG: Loading persistent vector store")
            vectorstore, storage_type = load_persistent_vector_store(cache_hash, embeddings)
            print(f"✅ DEBUG: Vector store loaded, type: {storage_type}")
            
            if not vectorstore:
                print(f"❌ DEBUG: Vector store is None")
                raise HTTPException(status_code=404, detail=f"Vector store not found for cache {cache_hash}")
            
            print(f"🔄 DEBUG: Starting document extraction, storage_type: {storage_type}")
            # Extract documents from vector store
            documents = []
            
            if storage_type == "memory" and hasattr(vectorstore, 'documents'):
                # For in-memory vector stores
                docs = vectorstore.documents
                for i, doc in enumerate(docs):
                    if limit and i >= limit:
                        break
                    doc_data = {
                        "id": i,
                        "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                    }
                    if include_content:
                        doc_data["content"] = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    documents.append(doc_data)
            
            elif storage_type == "redis":
                # For Redis vector stores, do a broad search to get documents
                try:
                    print(f"Attempting to search Redis vector store for documents...")
                    # Try different search strategies to get documents
                    sample_docs = []
                    
                    # First try searching with common words
                    for search_term in ["the", "and", "is", "of", "to", "a"]:
                        try:
                            print(f"Searching with term: {search_term}")
                            docs = vectorstore.similarity_search(search_term, k=20)
                            print(f"Found {len(docs)} documents with term '{search_term}'")
                            sample_docs.extend(docs)
                            if len(sample_docs) >= (limit or 50):
                                break
                        except Exception as e:
                            print(f"Error searching with term '{search_term}': {e}")
                            continue
                    
                    print(f"Total sample docs collected: {len(sample_docs)}")
                    
                    # Remove duplicates based on content
                    unique_docs = []
                    seen_content = set()
                    for doc in sample_docs:
                        content_key = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)[:100]
                        if content_key not in seen_content:
                            seen_content.add(content_key)
                            unique_docs.append(doc)
                            if limit and len(unique_docs) >= limit:
                                break
                    
                    print(f"Unique documents after deduplication: {len(unique_docs)}")
                    
                    for i, doc in enumerate(unique_docs):
                        if limit and i >= limit:
                            break
                        doc_data = {
                            "id": i,
                            "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                        }
                        if include_content:
                            doc_data["content"] = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        documents.append(doc_data)
                        
                    print(f"Final documents to return: {len(documents)}")
                        
                except Exception as search_error:
                    print(f"Error searching Redis vector store: {search_error}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    # Fallback: return empty documents list but don't fail
                    documents = []
            
        except Exception as e:
            print(f"Error loading vector store for {cache_hash}: {e}")
            # Return basic info without documents
            documents = []
        
        return {
            "cache_hash": cache_hash,
            "documents": documents,
            "total_documents": len(documents),
            "repo_urls": cache_info.get("repo_urls", []),
            "storage_type": storage_type if 'storage_type' in locals() else "unknown"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving cache documents: {str(e)}")


@mindmap_integration_router.get("/caches")
async def list_all_caches():
    """List all available caches with summary information"""
    try:
        cache_hashes = get_all_cache_hashes()
        
        caches = []
        for cache_hash in cache_hashes:
            try:
                cache_info = load_cache_info(cache_hash)
                if cache_info:
                    caches.append({
                        "cache_hash": cache_hash,
                        "repo_urls": cache_info.get("repo_urls", []),
                        "total_chunks": cache_info.get("total_chunks", 0),
                        "created_at": cache_info.get("created_at"),
                        "last_accessed": cache_info.get("last_accessed")
                    })
            except Exception as e:
                print(f"Error loading cache info for {cache_hash}: {e}")
                continue
        
        return {
            "total_caches": len(caches),
            "caches": caches
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing caches: {str(e)}")


@mindmap_integration_router.get("/cache/{cache_hash}/similarity_search")
async def similarity_search_cache(
    cache_hash: str, 
    query: str, 
    k: int = 5,
    score_threshold: float = 0.0
):
    """Perform similarity search within a specific cache"""
    try:
        from src.services.cache import load_persistent_vector_store
        from src.services.embeddings import SimpleTFIDFEmbeddings
        
        cache_info = load_cache_info(cache_hash)
        if not cache_info:
            raise HTTPException(status_code=404, detail=f"Cache {cache_hash} not found")
        
        # Load embeddings and vector store
        embeddings = SimpleTFIDFEmbeddings.from_saved(redis_client, get_vectorizer_key(cache_hash))
        vectorstore, storage_type = load_persistent_vector_store(cache_hash, embeddings)
        
        if not vectorstore:
            raise HTTPException(status_code=404, detail=f"Vector store not found for cache {cache_hash}")
        
        # Perform similarity search
        if hasattr(vectorstore, 'similarity_search_with_score'):
            results = vectorstore.similarity_search_with_score(query, k=k)
            # Filter by score threshold
            filtered_results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in results
                if score >= score_threshold
            ]
        else:
            # Fallback without scores
            docs = vectorstore.similarity_search(query, k=k)
            filtered_results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": None
                }
                for doc in docs
            ]
        
        return {
            "cache_hash": cache_hash,
            "query": query,
            "results": filtered_results,
            "total_results": len(filtered_results),
            "storage_type": storage_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing similarity search: {str(e)}")


@mindmap_integration_router.post("/cache/{cache_hash}/extract_concepts")
async def extract_concepts_from_cache(
    cache_hash: str,
    max_concepts: int = 20,
    min_frequency: int = 2
):
    """Extract key concepts from documents in a cache"""
    try:
        cache_info = load_cache_info(cache_hash)
        if not cache_info:
            raise HTTPException(status_code=404, detail=f"Cache {cache_hash} not found")
        
        documents = cache_info.get("documents", [])
        
        # Simple concept extraction (in production, use proper NLP)
        concept_freq = {}
        all_text = ""
        
        for doc in documents:
            content = doc.get("content", "")
            all_text += " " + content
            
            # Extract capitalized words as concepts
            import re
            words = re.findall(r'\b[A-Z][a-z]+\b', content)
            for word in words:
                if len(word) > 3:  # Filter short words
                    concept_freq[word] = concept_freq.get(word, 0) + 1
        
        # Filter by frequency and limit
        concepts = [
            {"concept": concept, "frequency": freq}
            for concept, freq in concept_freq.items()
            if freq >= min_frequency
        ]
        
        # Sort by frequency and limit
        concepts = sorted(concepts, key=lambda x: x["frequency"], reverse=True)[:max_concepts]
        
        return {
            "cache_hash": cache_hash,
            "concepts": concepts,
            "total_concepts": len(concepts),
            "total_documents": len(documents),
            "text_length": len(all_text)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting concepts: {str(e)}")
