"""
Core RAG service functionality for the LangChain RAG API.
Handles LLM integration and retrieval chains.
"""

import os
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from src.core.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_BASE,
    OPENROUTER_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    DEFAULT_REFERER,
    DEFAULT_TITLE
)
from src.core.state import get_app_state
from src.config.prompts import PromptTemplates, AnalysisType

def get_llm():
    """Initialize and return the LLM instance based on the application's current state."""
    app_state = get_app_state()
    provider = app_state.llm_provider

    if provider == "openrouter":
        if not OPENROUTER_API_KEY or not OPENROUTER_API_BASE:
            raise HTTPException(status_code=500, detail="OpenRouter environment variables not set")
        
        return ChatOpenAI(
            model=OPENROUTER_MODEL,
            openai_api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_API_BASE,
            temperature=0.0,
            extra_headers={"HTTP-Referer": DEFAULT_REFERER, "X-Title": DEFAULT_TITLE}
        )
    elif provider == "ollama":
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0.0,
            # Explicitly disable unsupported sampling parameters to avoid warnings
            mirostat=None,
            mirostat_eta=None,
            mirostat_tau=None,
            tfs_z=None,
        )
    else:
        # This case should ideally not be reached if state validation is working
        raise HTTPException(
            status_code=500,
            detail=f"Invalid LLM_PROVIDER in app state: {provider}. Choose 'openrouter' or 'ollama'."
        )

def get_retrieval_chain(llm, retriever, query_type="standard"):
    """Create a retrieval chain with centralized prompts."""
    
    # Map query types to analysis types
    if query_type == "detailed":
        analysis_type = AnalysisType.DETAILED
    elif query_type == "summary":
        analysis_type = AnalysisType.SUMMARY
    elif query_type == "deep_thinking":
        analysis_type = AnalysisType.DEEP_THINKING
    else:
        analysis_type = AnalysisType.STANDARD
    
    # Get prompt template from centralized configuration
    prompt_template = PromptTemplates.get_user_template(analysis_type)
    prompt = ChatPromptTemplate.from_template(prompt_template)
        
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

def get_enhanced_retrieval_chain(llm, retriever):
    """Create an enhanced retrieval chain for deeper analysis."""
    return get_retrieval_chain(llm, retriever, query_type="detailed")
