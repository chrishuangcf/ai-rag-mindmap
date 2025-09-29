"""
Google search fallback functionality for the LangChain RAG API.
"""

import re
import requests
from src.core.config import GOOGLE_SEARCH_AVAILABLE

def is_dont_know_response(response: str) -> bool:
    """Check if the AI response indicates it doesn't know the answer."""
    response_lower = response.lower().strip()
    
    # Common patterns for "I don't know" responses
    dont_know_patterns = [
        "i don't know",
        "i do not know", 
        "don't know",
        "do not know",
        "i'm not sure",
        "i am not sure",
        "i cannot answer",
        "i can't answer",
        "no information",
        "not enough information",
        "insufficient information",
        "no relevant information",
        "cannot find",
        "can't find",
        "not available in",
        "not found in",
        "not provided in",
        "not mentioned in",
        "based on the provided context, i",
        "the provided context does not",
        "the context does not contain",
        "no information about this",
        "i don't have information",
        "i do not have information"
    ]
    
    return any(pattern in response_lower for pattern in dont_know_patterns)

def perform_google_search(query: str, num_results: int = 5):
    """Perform Google search as fallback when RAG documents don't have relevant results."""
    if not GOOGLE_SEARCH_AVAILABLE:
        return []
    
    try:
        from googlesearch import search as google_search
        print(f"Performing Google search fallback for query: '{query}'")
        results = []
        
        # Use googlesearch-python to get search results
        search_results = google_search(query, num_results=num_results, sleep_interval=1)
        
        for i, url in enumerate(search_results):
            try:
                # Try to fetch a brief snippet from each URL
                response = requests.get(url, timeout=5, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code == 200:
                    # Extract first 500 characters as preview
                    content_preview = response.text[:500] + "..." if len(response.text) > 500 else response.text
                    
                    # Clean up HTML tags for basic preview
                    content_preview = re.sub(r'<[^>]+>', ' ', content_preview)
                    content_preview = re.sub(r'\s+', ' ', content_preview).strip()
                    
                    results.append({
                        "title": f"Google Result {i+1}",
                        "url": url,
                        "content": content_preview[:300] + "..." if len(content_preview) > 300 else content_preview,
                        "source": "Google Search"
                    })
                else:
                    # Fallback with just the URL
                    results.append({
                        "title": f"Google Result {i+1}",
                        "url": url,
                        "content": f"Found relevant result at {url}",
                        "source": "Google Search"
                    })
            except Exception as e:
                print(f"Error fetching content from {url}: {e}")
                # Still include the URL even if we can't fetch content
                results.append({
                    "title": f"Google Result {i+1}",
                    "url": url,
                    "content": f"Found relevant result at {url}",
                    "source": "Google Search"
                })
        
        print(f"Google search returned {len(results)} results")
        return results
        
    except Exception as e:
        print(f"Google search failed: {e}")
        return []

def get_google_search_summary(query: str, search_results: list):
    """Generate an AI summary based on Google search results."""
    if not search_results:
        return "No relevant information found in Google search."
    
    try:
        from src.services.rag_service import get_llm
        from src.config.prompts import get_google_search_prompt
        
        llm = get_llm()
        
        # Create context string from search results
        context_parts = []
        for result in search_results:
            source = result.get('url', 'Unknown source')
            content = result.get('content', '')
            context_parts.append(f"Source: {source}\nContent: {content}\n")
        
        context = "\n".join(context_parts)
        
        # Use centralized prompt system
        prompt = get_google_search_prompt(query, context)
        
        # Get response from LLM
        response = llm.invoke(prompt)
        
        # Extract content from response if it's an AIMessage
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
        
    except Exception as e:
        print(f"Error generating Google search summary: {e}")
        return f"Found {len(search_results)} relevant results from Google search, but couldn't generate summary."
