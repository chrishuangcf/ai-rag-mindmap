"""
Core API endpoints for RAG functionality.
"""

import os
import requests
import time
import math
import tempfile
from typing import List
from fastapi import APIRouter, HTTPException, Request, File, UploadFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langsmith import traceable

# Import component classes
from src.components import StorageManager, RAGProcessor, FileIOManager, GitLabManager

from src.core.models import SimpleRAGRequest, CachedRAGRequest, GlobalSearchRequest, ConceptSearchRequest, TestGitlabTokenRequest
from src.utils.url_utils import validate_and_convert_urls, generate_doc_hash
from src.services.embeddings import SimpleTFIDFEmbeddings, SimpleInMemoryVectorStore
from src.services.cache import (
    save_cache_info, load_cache_info, is_cache_valid, 
    load_persistent_vector_store, delete_cache,
    get_vectorizer_key, get_all_cache_hashes,
    get_memory_vector_stores
)
from src.services.rag_service import get_llm, get_retrieval_chain, get_enhanced_retrieval_chain
from src.services.google_search import is_dont_know_response, perform_google_search, get_google_search_summary
from src.services.unified_mindmap import unified_mindmap_service
from src.core.config import REDIS_AVAILABLE, LANGCHAIN_REDIS_NEW, REDIS_URL, GOOGLE_SEARCH_AVAILABLE, redis_client
from src.core.state import get_app_state
from src.config.prompts import PromptTemplates, AnalysisType, get_global_search_prompt

router = APIRouter()
app_state = get_app_state()

# Initialize component instances
storage_manager = StorageManager()
rag_processor = RAGProcessor()
file_io_manager = FileIOManager()
gitlab_manager = GitLabManager()

# Legacy functions - replaced by GitLabManager component
# These are kept for backward compatibility but delegate to the component

def is_gitlab_url(url):
    """Check if URL is from GitLab (gitlab.com or self-hosted GitLab)."""
    return gitlab_manager.is_gitlab_url(url)

async def fetch_gitlab_file(url, token):
    """
    Fetch GitLab file with authentication support.
    Uses GitLabManager component for all GitLab operations.
    """
    return await gitlab_manager.fetch_gitlab_file(url, token)

def safe_score(score):
    if score is None:
        return None
    if isinstance(score, float) and (math.isnan(score) or math.isinf(score)):
        return None
    return score

@router.post("/mindmap/documents_for_concept", summary="Get documents for a mind map concept")
async def get_documents_for_concept(request: ConceptSearchRequest):
    """
    For a given concept (node name) from a mind map, search across the relevant
    caches to find the source documents.
    """
    all_results = []
    for cache_hash in request.cache_hashes:
        try:
            embeddings = SimpleTFIDFEmbeddings.from_saved(redis_client, get_vectorizer_key(cache_hash))
            vectorstore, storage_type = load_persistent_vector_store(cache_hash, embeddings)

            if not vectorstore:
                continue

            # Perform similarity search
            if hasattr(vectorstore, 'similarity_search_with_score'):
                results = vectorstore.similarity_search_with_score(request.concept, k=request.top_k)
                for doc, score in results:
                    all_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": safe_score(score),  # <-- sanitize here
                        "cache_hash": cache_hash,
                        "storage_type": storage_type
                    })
            else:
                # Fallback for vectorstores without score
                docs = vectorstore.similarity_search(request.concept, k=request.top_k)
                for doc in docs:
                    all_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": None, # No score available
                        "cache_hash": cache_hash,
                        "storage_type": storage_type
                    })

        except Exception as e:
            print(f"Error searching in cache {cache_hash} for concept '{request.concept}': {e}")
            continue

    # Sort results by score (descending) if scores are available
    if all(r['score'] is not None for r in all_results):
        sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
    else:
        sorted_results = all_results

    top_results = sorted_results[:request.top_k]

    return {"documents": top_results}

# --- LLM Configuration Endpoints ---

@router.get("/config/llm", summary="Get current LLM configuration")
async def get_llm_config():
    """Returns the current LLM provider and model."""
    return {
        "provider": app_state.llm_provider,
        "model": app_state.llm_model
    }

@router.post("/config/llm", summary="Switch LLM provider")
async def set_llm_config(request: Request):
    """
    Switches the LLM provider on-the-fly.
    Expects a JSON body with a "provider" key ("openrouter" or "ollama").
    """
    try:
        data = await request.json()
        provider = data.get("provider")
        if not provider or provider not in ["openrouter", "ollama"]:
            raise HTTPException(status_code=400, detail="Invalid provider specified.")
        
        app_state.llm_provider = provider
        return await get_llm_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch provider: {e}")

@router.post("/test-gitlab-token", summary="Test GitLab access token")
async def test_gitlab_token(request: TestGitlabTokenRequest):
    """
    Test if a GitLab access token is valid by making a simple API call.
    """
    try:
        headers = {"PRIVATE-TOKEN": request.token}
        response = requests.get(request.test_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            user_info = response.json()
            return {
                "valid": True,
                "message": "GitLab token is valid",
                "user_info": {
                    "name": user_info.get("name"),
                    "username": user_info.get("username"),
                    "email": user_info.get("email")
                }
            }
        elif response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="Invalid GitLab token. Please check your token and ensure it has the required permissions."
            )
        elif response.status_code == 403:
            raise HTTPException(
                status_code=403,
                detail="GitLab token lacks sufficient permissions. Ensure it has 'read_api' or 'read_repository' scope."
            )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"GitLab API returned status {response.status_code}: {response.text}"
            )
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="GitLab API request timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to GitLab API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error testing GitLab token: {str(e)}")

@router.post("/rag/upload", summary="Upload and process multiple files for RAG")
async def rag_upload_endpoint(files: List[UploadFile] = File(...)):
    """
    Upload multiple PDF, DOCX, TXT, or Markdown files and process them for RAG.
    PDF/DOCX files are transcoded to Markdown, while TXT/Markdown files are processed directly.
    """
    try:
        # Use FileIOManager to process uploaded files
        documents, processed_files, skipped_files = file_io_manager.process_uploaded_files(files)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid content could be extracted from the uploaded files")

        # Process documents with RAGProcessor and StorageManager
        doc_hash = generate_doc_hash(processed_files)
        
        # Split documents using RAGProcessor
        splits = rag_processor.split_documents(documents)
        
        # Create embeddings using RAGProcessor
        embeddings = rag_processor.create_embeddings(splits)
        
        # Actually embed the documents to fit the vectorizer
        _ = embeddings.embed_documents(splits)

        # Store using StorageManager
        if REDIS_AVAILABLE and redis_client and redis_client.sismember("cache_hashes", doc_hash):
            await delete_cache(doc_hash)

        vectorstore, storage_type = await storage_manager.create_vector_store(doc_hash, splits, embeddings)
        
        # Save the embeddings/vectorizer for later retrieval
        storage_manager.save_embeddings(embeddings, doc_hash)
        
        # Save cache info
        save_cache_info(doc_hash, processed_files, len(splits))

        # Trigger unified mind map update
        try:
            await unified_mindmap_service.trigger_mindmap_update()
        except Exception as e:
            print(f"Failed to update mind map: {e}")
            # Don't fail the whole operation if mind map update fails

        return {
            "message": f"Successfully processed {len(processed_files)} files",
            "total_chunks": len(splits),
            "cache_hash": doc_hash,
            "processed_files": processed_files,
            "skipped_files": skipped_files,
            "storage_type": storage_type
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.post("/rag")
async def rag_create_endpoint(request: SimpleRAGRequest):
    """Create and cache RAG documents from URLs."""
    try:
        repo_urls = validate_and_convert_urls(request.repo_urls)
        doc_hash = generate_doc_hash(repo_urls)
        
        # --- 1. Fetch and process documents using FileIOManager ---
        all_docs = file_io_manager.process_urls(repo_urls, request.gitlab_token)
        
        if not all_docs:
            raise HTTPException(status_code=400, detail="Failed to fetch any documents.")

        # --- 2. Process documents using RAGProcessor ---
        splits = rag_processor.split_documents(all_docs)
        embeddings = rag_processor.create_embeddings(splits)
        
        # Actually embed the documents to fit the vectorizer
        _ = embeddings.embed_documents(splits)
        
        # --- 3. Store using StorageManager ---
        if REDIS_AVAILABLE and redis_client and redis_client.sismember("cache_hashes", doc_hash):
            print(f"Deleting stale cache for hash: {doc_hash}")
            await delete_cache(doc_hash)

        vectorstore, storage_type = await storage_manager.create_vector_store(doc_hash, splits, embeddings)
        
        # Save the embeddings/vectorizer for later retrieval
        storage_manager.save_embeddings(embeddings, doc_hash)
        
        # Save cache info
        save_cache_info(doc_hash, repo_urls, len(splits))
        print(f"Cache saved for hash: {doc_hash}")

        # Trigger unified mind map update
        try:
            await unified_mindmap_service.trigger_mindmap_update()
        except Exception as e:
            print(f"Failed to update mind map: {e}")
            # Don't fail the whole operation if mind map update fails

        # --- 4. Return document processing success info ---
        return {
            "message": "Documents processed successfully",
            "total_chunks": len(splits),
            "cached": False,
            "cache_hash": doc_hash,
            "processed_urls": repo_urls,
            "storage_type": storage_type
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in rag_create_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/rag/cached")
async def rag_cached_endpoint(request: CachedRAGRequest):
    """Query against cached RAG documents."""
    try:
        # Load cached data
        cache_info = load_cache_info(request.cache_hash)
        if not cache_info:
            raise HTTPException(status_code=404, detail=f"Cache with hash {request.cache_hash} not found")

        print(f"Loading cached vector store: {request.cache_hash}")
        
        # Load vector store
        vectorstore = storage_manager.load_vector_store(request.cache_hash)
        if not vectorstore:
            raise HTTPException(status_code=500, detail=f"Failed to load cached vector store for {request.cache_hash}")

        # Process RAG query using RAGProcessor
        rag_response = rag_processor.process_rag_query(request.query, vectorstore, use_enhanced=True)

        # Update last accessed time
        cache_info["last_accessed"] = time.time()
        save_cache_info(request.cache_hash, cache_info.get("repo_urls", []), cache_info.get("total_chunks", 0))

        # Process with Google fallback if needed using RAGProcessor
        enhanced_response = rag_processor.process_with_google_fallback(request.query, rag_response)
        
        # Format response
        result = {
            "answer": enhanced_response["answer"],
            "cache_hash": request.cache_hash,
            "total_chunks": cache_info.get('total_chunks', 0),
            "cached": True,
            "cached_urls": cache_info.get("repo_urls", []),
            "source_documents": enhanced_response.get("source_documents", [])
        }
        
        # Add Google fallback info if present
        if enhanced_response.get("enhanced_with_google"):
            result["google_fallback"] = True
            result["google_search"] = enhanced_response.get("google_search", {})
            result["original_rag_answer"] = rag_response["answer"]
        
        return result
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in rag_cached_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/search/global")
async def global_search_endpoint(request: dict):
    """Search across all cached documents with persistent storage."""
    try:
        query = request.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        max_results = request.get("max_results", 10)
        all_results = []
        
        # Get all available caches
        cache_hashes = get_all_cache_hashes()
        
        print(f"Global search: Found {len(cache_hashes)} cached document sets")
        print(f"Cache hashes: {list(cache_hashes)}")
        
        if not cache_hashes:
            return {
                "query": query,
                "total_results": 0,
                "searched_caches": 0,
                "results": [],
                "message": "No cached documents found. Please upload and process documents first using the main RAG form."
            }
        
        for doc_hash in cache_hashes:
            try:
                cache_info = load_cache_info(doc_hash)
                if not cache_info:
                    print(f"No cache info found for hash: {doc_hash}")
                    continue
                
                print(f"Processing cache {doc_hash} with {cache_info.get('total_chunks', 0)} chunks")
                
                # Load embeddings and vector store
                try:
                    embeddings = SimpleTFIDFEmbeddings.from_saved(redis_client, get_vectorizer_key(doc_hash))
                    vectorstore, storage_type = load_persistent_vector_store(doc_hash, embeddings)
                except Exception as e:
                    print(f"Could not load embeddings for cache {doc_hash}: {e}")
                    continue
                
                if not vectorstore:
                    print(f"No vector store available for cache {doc_hash}")
                    continue
                
                # Search in this cache
                try:
                    relevant_docs = vectorstore.similarity_search(query, k=max_results)
                    print(f"Found {len(relevant_docs)} relevant documents in cache {doc_hash}")
                    
                    # Add results with cache info
                    for doc in relevant_docs:
                        all_results.append({
                            "content": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                            "source_url": doc.metadata.get('source_url', 'Unknown'),
                            "cache_hash": doc_hash,
                            "cache_urls": cache_info.get("repo_urls", []),
                            "storage_type": storage_type,
                            "relevance_score": "high"
                        })
                        
                except Exception as search_error:
                    print(f"Search error in cache {doc_hash}: {search_error}")
                    continue
                    
            except Exception as e:
                print(f"Error processing cache {doc_hash}: {e}")
                continue
        
        # Sort and limit results
        all_results = all_results[:max_results]
        
        result = {
            "query": query,
            "total_results": len(all_results),
            "searched_caches": len(cache_hashes),
            "results": all_results
        }
        
        # If no results found in RAG documents, try Google search as fallback
        if len(all_results) == 0 and GOOGLE_SEARCH_AVAILABLE:
            print(f"No RAG results found, attempting Google search fallback for: '{query}'")
            google_results = perform_google_search(query, num_results=5)
            
            if google_results:
                result["google_fallback"] = True
                result["google_results"] = google_results
                result["total_results"] = len(google_results)
                result["message"] = f"No relevant documents found for query '{query}' in {len(cache_hashes)} cached document sets. Showing Google search results as fallback."
                
                # Convert Google results to the same format as RAG results
                for google_result in google_results:
                    all_results.append({
                        "content": google_result["content"],
                        "source_url": google_result["url"],
                        "cache_hash": "google_search",
                        "cache_urls": [google_result["url"]],
                        "storage_type": "google",
                        "relevance_score": "google_fallback"
                    })
                
                result["results"] = all_results
        
        return result
        
    except Exception as e:
        print(f"Error in global_search_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/rag/global")
async def global_rag_endpoint(request: dict):
    """Get AI-powered answers from all cached documents with persistent storage."""
    try:
        query = request.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        detailed_analysis = request.get("detailed_analysis", False)
        max_docs_per_cache = request.get("max_docs_per_cache", 8 if detailed_analysis else 5)
        
        # Collect relevant documents from all caches
        all_relevant_docs = []
        searched_caches = []
        
        cache_hashes = get_all_cache_hashes()
        
        print(f"Global RAG: Found {len(cache_hashes)} cached document sets, detailed_analysis={detailed_analysis}")
        
        if not cache_hashes:
            return {
                "answer": "No cached documents found. Please upload and process documents first using the main RAG form.",
                "query": query,
                "searched_caches": [],
                "total_documents_found": 0,
                "source_documents": [],
                "analysis_type": "deep" if detailed_analysis else "summary"
            }
        
        for doc_hash in cache_hashes:
            try:
                cache_info = load_cache_info(doc_hash)
                if not cache_info:
                    continue
                
                # Load embeddings and vector store
                try:
                    embeddings = SimpleTFIDFEmbeddings.from_saved(redis_client, get_vectorizer_key(doc_hash))
                    vectorstore, storage_type = load_persistent_vector_store(doc_hash, embeddings)
                except Exception as e:
                    print(f"Could not load embeddings for cache {doc_hash}: {e}")
                    continue
                
                if not vectorstore:
                    continue
                
                # Get relevant documents from this cache
                try:
                    relevant_docs = vectorstore.similarity_search(query, k=max_docs_per_cache)
                    
                    # Add metadata
                    docs_added = 0
                    for doc in relevant_docs:
                        if docs_added >= max_docs_per_cache:
                            break
                        doc.metadata['cache_hash'] = doc_hash
                        doc.metadata['cache_urls'] = cache_info.get("repo_urls", [])
                        doc.metadata['storage_type'] = storage_type
                        all_relevant_docs.append(doc)
                        docs_added += 1
                    
                    searched_caches.append({
                        "hash": doc_hash,
                        "urls": cache_info.get("repo_urls", []),
                        "docs_found": docs_added,
                        "storage_type": storage_type
                    })
                    
                except Exception as search_error:
                    print(f"Search error in cache {doc_hash}: {search_error}")
                    continue
                    
            except Exception as e:
                print(f"Error processing cache {doc_hash}: {e}")
                continue
        
        if not all_relevant_docs:
            # Try Google search as fallback when no RAG documents found
            if GOOGLE_SEARCH_AVAILABLE:
                print(f"No RAG documents found, attempting Google search fallback for: '{query}'")
                google_results = perform_google_search(query, num_results=3)
                
                if google_results:
                    # Generate AI summary from Google results
                    google_summary = get_google_search_summary(query, google_results)
                    
                    return {
                        "answer": google_summary,
                        "query": query,
                        "searched_caches": searched_caches,
                        "total_documents_found": len(google_results),
                        "analysis_type": "deep" if detailed_analysis else "summary",
                        "google_fallback": True,
                        "source_documents": [
                            {
                                "content": result["content"],
                                "source_url": result["url"],
                                "cache_hash": "google_search",
                                "cache_urls": [result["url"]],
                                "storage_type": "google"
                            }
                            for result in google_results
                        ]
                    }
            
            return {
                "answer": f"No relevant documents found for query '{query}' in {len(cache_hashes)} cached document sets.",
                "query": query,
                "searched_caches": searched_caches,
                "total_documents_found": 0,
                "analysis_type": "deep" if detailed_analysis else "summary",
                "source_documents": []
            }
        
        # Create AI response with different prompts based on analysis type
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.chains.combine_documents import create_stuff_documents_chain
        
        llm = get_llm()
        
        # Use centralized prompt system
        if detailed_analysis:
            analysis_type = AnalysisType.DEEP_THINKING
        else:
            analysis_type = AnalysisType.SUMMARY
        
        # Get prompt template from centralized configuration
        prompt_template = PromptTemplates.get_user_template(analysis_type)
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        response = document_chain.invoke({
            "context": all_relevant_docs,
            "input": query
        })
        
        # Check if AI doesn't know the answer and trigger Google search fallback
        if is_dont_know_response(response) and GOOGLE_SEARCH_AVAILABLE:
            print(f"AI doesn't know the answer, attempting Google search fallback for: '{query}'")
            google_results = perform_google_search(query, num_results=3)
            
            if google_results:
                # Generate AI summary from Google results
                google_summary = get_google_search_summary(query, google_results)
                
                return {
                    "answer": google_summary,
                    "query": query,
                    "searched_caches": searched_caches,
                    "total_documents_found": len(google_results),
                    "analysis_type": "deep" if detailed_analysis else "summary",
                    "google_fallback": True,
                    "original_rag_answer": response,
                    "source_documents": [
                        {
                            "content": result["content"],
                            "source_url": result["url"],
                            "cache_hash": "google_search",
                            "cache_urls": [result["url"]],
                            "storage_type": "google"
                        }
                        for result in google_results
                    ]
                }
        
        # Return normal RAG response if AI knows the answer
        return {
            "answer": response,
            "query": query,
            "searched_caches": searched_caches,
            "total_documents_found": len(all_relevant_docs),
            "analysis_type": "deep" if detailed_analysis else "summary",
            "source_documents": [
                {
                    "content": doc.page_content,
                    "source_url": doc.metadata.get('source_url'),
                    "cache_hash": doc.metadata.get('cache_hash'),
                    "cache_urls": doc.metadata.get('cache_urls', []),
                    "storage_type": doc.metadata.get('storage_type')
                }
                for doc in all_relevant_docs
            ]
        }
        
    except Exception as e:
        print(f"Error in global_rag_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/rag/detailed", summary="Detailed RAG analysis with comprehensive insights")
async def detailed_rag_endpoint(request: CachedRAGRequest):
    """
    Perform detailed analysis with comprehensive insights from cached RAG documents.
    This endpoint provides deeper, more thorough responses than the standard cached endpoint.
    """
    try:
        cache_info = load_cache_info(request.cache_hash)
        if not cache_info:
            raise HTTPException(status_code=404, detail=f"Cache with hash {request.cache_hash} not found")

        print(f"Loading cached vector store for detailed analysis: {request.cache_hash}")
        
        try:
            embeddings = SimpleTFIDFEmbeddings.from_saved(redis_client, get_vectorizer_key(request.cache_hash))
            vectorstore, _ = load_persistent_vector_store(request.cache_hash, embeddings)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load cached vector store: {e}")

        llm = get_llm()
        # Retrieve more documents for comprehensive analysis
        retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
        retrieval_chain = get_enhanced_retrieval_chain(llm, retriever)
        
        # Use enhanced prompt for detailed analysis
        rag_response = retrieval_chain.invoke({"input": request.query})

        # Update last accessed time
        cache_info["last_accessed"] = time.time()
        save_cache_info(request.cache_hash, cache_info.get("repo_urls", []), cache_info.get("total_chunks", 0))

        # Check if AI doesn't know the answer and trigger Google search fallback
        if is_dont_know_response(rag_response["answer"]) and GOOGLE_SEARCH_AVAILABLE:
            print(f"AI doesn't know the answer, attempting Google search fallback for: '{request.query}'")
            google_results = perform_google_search(request.query, num_results=5)  # More results for detailed analysis
            
            if google_results:
                # Generate comprehensive AI summary from Google results
                google_summary = get_google_search_summary(request.query, google_results)
                
                return {
                    "answer": google_summary,
                    "cache_hash": request.cache_hash,
                    "total_chunks": cache_info.get('total_chunks', 0),
                    "cached": True,
                    "cached_urls": cache_info.get("repo_urls", []),
                    "google_fallback": True,
                    "analysis_type": "detailed",
                    "original_rag_answer": rag_response["answer"],
                    "source_documents": [
                        {
                            "content": result["content"],
                            "source_url": result["url"],
                            "cache_hash": "google_search",
                            "cache_urls": [result["url"]],
                            "storage_type": "google"
                        }
                        for result in google_results
                    ]
                }

        # Return enhanced RAG response
        return {
            "answer": rag_response["answer"],
            "cache_hash": request.cache_hash,
            "total_chunks": cache_info.get('total_chunks', 0),
            "cached": True,
            "cached_urls": cache_info.get("repo_urls", []),
            "analysis_type": "detailed",
            "documents_analyzed": len(rag_response["context"]),
            "source_documents": [
                {
                    "content": doc.page_content, 
                    "source_url": doc.metadata.get('source_url'),
                    "relevance_score": "high"
                }
                for doc in rag_response["context"]
            ]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in detailed_rag_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/rag/url", summary="Direct RAG query from URL with enhanced analysis")
@traceable(run_type="chain", name="URL RAG Endpoint")
async def rag_url_endpoint(request: dict):
    """
    Fetch content from URL(s), process it, and provide detailed analysis directly without caching.
    Useful for one-time comprehensive analysis of web content.
    """
    try:
        urls = request.get("urls", [])
        query = request.get("query", "").strip()
        detailed = request.get("detailed", True)  # Default to detailed analysis
        
        if not urls:
            raise HTTPException(status_code=400, detail="At least one URL is required")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Validate and convert URLs
        validated_urls = validate_and_convert_urls(urls)
        
        # --- 1. Fetch and process documents ---
        all_docs = []
        successful_urls = []
        
        for i, url in enumerate(validated_urls):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Determine content type and process accordingly
                content_type = response.headers.get('content-type', '').lower()
                url_lower = url.lower()
                
                # Create document directly for text-based files
                if (any(ext in url_lower for ext in ['.md', '.markdown']) or 'text/markdown' in content_type or
                    any(ext in url_lower for ext in ['.txt']) or 'text/plain' in content_type):
                    content = response.text
                    if content.strip():
                        doc = Document(page_content=content, metadata={"source_url": url})
                        all_docs.append(doc)
                        successful_urls.append(url)
                else:
                    # For other file types, use TextLoader as fallback
                    
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp_file:
                        tmp_file.write(response.text)
                        tmp_file_path = tmp_file.name
                    
                    try:
                        loader = TextLoader(tmp_file_path, encoding='utf-8')
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata["source_url"] = url
                        all_docs.extend(docs)
                        successful_urls.append(url)
                    finally:
                        os.unlink(tmp_file_path)
                        
            except requests.exceptions.RequestException as e:
                print(f"Warning: Failed to fetch {url}: {e}")
                continue
            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue
        
        if not all_docs:
            raise HTTPException(status_code=400, detail="Failed to fetch any documents from the provided URLs.")

        # --- 2. Split documents ---
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        
        # --- 3. Create temporary embeddings and vector store (no caching) ---
        embeddings = SimpleTFIDFEmbeddings(max_features=500)
        _ = embeddings.embed_documents(splits)
        
        # Create temporary in-memory vector store
        vectorstore = SimpleInMemoryVectorStore(embeddings, splits)
        
        # --- 4. Perform enhanced RAG query ---
        llm = get_llm()
        # Use more documents for detailed analysis
        k_value = 10 if detailed else 5
        retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})
        
        if detailed:
            retrieval_chain = get_enhanced_retrieval_chain(llm, retriever)
        else:
            retrieval_chain = get_retrieval_chain(llm, retriever)
            
        rag_response = retrieval_chain.invoke({"input": query})
        
        # Check if AI doesn't know the answer and trigger Google search fallback
        if is_dont_know_response(rag_response["answer"]) and GOOGLE_SEARCH_AVAILABLE:
            print(f"AI doesn't know the answer, attempting Google search fallback for: '{query}'")
            google_results = perform_google_search(query, num_results=5)
            
            if google_results:
                google_summary = get_google_search_summary(query, google_results)
                
                return {
                    "answer": google_summary,
                    "query": query,
                    "processed_urls": successful_urls,
                    "total_chunks": len(splits),
                    "cached": False,
                    "analysis_type": "detailed" if detailed else "standard",
                    "google_fallback": True,
                    "original_rag_answer": rag_response["answer"],
                    "source_documents": [
                        {
                            "content": result["content"],
                            "source_url": result["url"],
                            "cache_hash": "google_search",
                            "cache_urls": [result["url"]],
                            "storage_type": "google"
                        }
                        for result in google_results
                    ]
                }
        
        # Return enhanced RAG response
        return {
            "answer": rag_response["answer"],
            "query": query,
            "processed_urls": successful_urls,
            "total_chunks": len(splits),
            "cached": False,
            "analysis_type": "detailed" if detailed else "standard",
            "documents_analyzed": len(rag_response["context"]),
            "source_documents": [
                {
                    "content": doc.page_content, 
                    "source_url": doc.metadata.get('source_url'),
                    "relevance_score": "high"
                }
                for doc in rag_response["context"]
            ]
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in rag_url_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/mindmap/unified", summary="Get unified mind map data")
async def get_unified_mindmap():
    """Get the unified mind map data for web UI visualization."""
    try:
        mindmap_data = unified_mindmap_service.get_mindmap_data()
        
        if not mindmap_data:
            # Try to create the mind map if it doesn't exist
            await unified_mindmap_service.trigger_mindmap_update()
            mindmap_data = unified_mindmap_service.get_mindmap_data()
        
        if not mindmap_data:
            return {"status": "no_data", "message": "No cached documents available for mind map"}
        
        return {
            "status": "success", 
            "mindmap": mindmap_data.get("mindmap", {}),
            "stats": mindmap_data.get("stats", {})
        }
        
    except Exception as e:
        print(f"Error getting unified mind map: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/mindmap/refresh", summary="Refresh unified mind map")
async def refresh_unified_mindmap():
    """Manually refresh the unified mind map with current data."""
    try:
        await unified_mindmap_service.trigger_mindmap_update()
        return {"status": "success", "message": "Mind map refresh triggered"}
        
    except Exception as e:
        print(f"Error refreshing mind map: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/cache/{cache_hash}/documents", summary="Get documents for a cache hash")
async def get_cache_documents(cache_hash: str):
    """Get all documents for a specific cache hash"""
    try:
        print(f"Getting documents for cache: {cache_hash}")
        
        # Load cache info to verify it exists
        cache_info = load_cache_info(cache_hash)
        if not cache_info:
            print(f"Cache info not found for {cache_hash}")
            raise HTTPException(status_code=404, detail=f"Cache {cache_hash} not found")
        
        print(f"Cache info loaded: {cache_info}")
        
        # Load the actual documents from the vector store
        documents = []
        
        try:
            print(f"Loading embeddings for cache {cache_hash}")
            # Create embeddings instance 
            embeddings = SimpleTFIDFEmbeddings.from_saved(redis_client, get_vectorizer_key(cache_hash))
            print(f"Embeddings loaded successfully")
            
            # Load the vector store
            print(f"Loading vector store for cache {cache_hash}")
            vectorstore, storage_type = load_persistent_vector_store(cache_hash, embeddings)
            print(f"Vector store loaded, type: {storage_type}")
            
            if not vectorstore:
                print(f"Vector store is None for cache {cache_hash}")
                raise HTTPException(status_code=404, detail=f"Vector store not found for cache {cache_hash}")
            
            # Extract documents from vector store
            if storage_type == "memory" and hasattr(vectorstore, 'documents'):
                print(f"Using memory vector store with {len(vectorstore.documents)} documents")
                # For in-memory vector stores, get the actual documents
                docs = vectorstore.documents
                for i, doc in enumerate(docs):
                    documents.append({
                        "content": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                        "metadata": doc.metadata if hasattr(doc, 'metadata') else {
                            "source_url": cache_info.get("repo_urls", [f"document_{i}"])[0] if cache_info.get("repo_urls") else f"document_{i}",
                            "cache_hash": cache_hash,
                            "document_index": i
                        }
                    })
            else:
                print(f"Using Redis vector store, attempting to search for documents")
                # For Redis vector stores, search to get documents
                try:
                    # Search with common terms to retrieve documents
                    docs_found = []
                    search_terms = ["the", "and", "is", "a", "to", "machine", "learning", "system", "data"]
                    
                    for search_term in search_terms:
                        try:
                            docs = vectorstore.similarity_search(search_term, k=20)
                            docs_found.extend(docs)
                            if len(docs_found) > 50:  # Limit to prevent too many duplicates
                                break
                        except Exception as search_error:
                            print(f"Search failed for term '{search_term}': {search_error}")
                            continue
                    
                    print(f"Found {len(docs_found)} documents from search")
                    
                    # Remove duplicates
                    unique_docs = []
                    seen_content = set()
                    for doc in docs_found:
                        content_key = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)[:100]
                        if content_key not in seen_content:
                            seen_content.add(content_key)
                            unique_docs.append(doc)
                    
                    print(f"After deduplication: {len(unique_docs)} unique documents")
                    
                    for i, doc in enumerate(unique_docs):
                        documents.append({
                            "content": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                            "metadata": doc.metadata if hasattr(doc, 'metadata') else {
                                "source_url": cache_info.get("repo_urls", [f"document_{i}"])[0] if cache_info.get("repo_urls") else f"document_{i}",
                                "cache_hash": cache_hash,
                                "document_index": i
                            }
                        })
                        
                except Exception as redis_error:
                    print(f"Error retrieving documents from Redis: {redis_error}")
                    # Fallback to empty documents
                    documents = []
                    
        except Exception as vector_error:
            print(f"Error loading vector store for cache {cache_hash}: {vector_error}")
            print(f"Vector error type: {type(vector_error)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            
            # Fallback to placeholder content
            documents = []
            repo_urls = cache_info.get("repo_urls", [])
            
            for i, url in enumerate(repo_urls):
                documents.append({
                    "content": f"Document from {url} (vector store unavailable - {str(vector_error)})",
                    "metadata": {
                        "source_url": url,
                        "cache_hash": cache_hash,
                        "document_index": i
                    }
                })
        
        print(f"Returning {len(documents)} documents for cache {cache_hash}")
        return {"documents": documents, "total_documents": len(documents)}
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in get_cache_documents: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@router.get("/cache/{cache_hash}/embeddings", summary="Get embeddings for a cache hash")
async def get_cache_embeddings(cache_hash: str):
    """Get all embeddings for a cache hash"""
    try:
        # Load cache info to verify it exists
        cache_info = load_cache_info(cache_hash)
        if not cache_info:
            raise HTTPException(status_code=404, detail=f"Cache {cache_hash} not found")
        
        # For now, return empty embeddings (the mind map service will handle this)
        total_chunks = cache_info.get("total_chunks", 0)
        embeddings_list = [[] for _ in range(total_chunks)]  # Empty embeddings for each chunk
        
        return {"embeddings": embeddings_list, "total_embeddings": len(embeddings_list)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving embeddings: {str(e)}")

@router.get("/cache/{cache_hash}/info", summary="Get cache information")
async def get_cache_info(cache_hash: str):
    """Get information about a specific cache"""
    try:
        cache_info = load_cache_info(cache_hash)
        if not cache_info:
            raise HTTPException(status_code=404, detail=f"Cache {cache_hash} not found")
        
        return cache_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving cache info: {str(e)}")



@router.post("/test-gitlab-url", summary="Test GitLab URL access with token")
async def test_gitlab_url(request: dict):
    """
    Test GitLab URL access with a token to help debug authentication issues.
    """
    url = request.get("url")
    token = request.get("token")
    
    if not url or not token:
        raise HTTPException(status_code=400, detail="Both 'url' and 'token' are required")
    
    if not is_gitlab_url(url):
        raise HTTPException(status_code=400, detail="URL must be a GitLab URL")
    
    try:
        content = await fetch_gitlab_file(url, token)
        return {
            "success": True,
            "message": "Successfully fetched GitLab file",
            "content_length": len(content) if content else 0,
            "content_preview": content[:200] + "..." if content and len(content) > 200 else (content or "")
        }
    except HTTPException as e:
        return {
            "success": False,
            "error": e.detail,
            "status_code": e.status_code
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "status_code": 500
        }
