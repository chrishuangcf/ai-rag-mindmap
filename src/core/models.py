"""
Pydantic models and data structures for the LangChain RAG API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class SimpleRAGRequest(BaseModel):
    repo_urls: list[str]
    gitlab_token: Optional[str] = None

class CachedRAGRequest(BaseModel):
    cache_hash: str
    query: str

class GlobalSearchRequest(BaseModel):
    query: str

class ConceptSearchRequest(BaseModel):
    """Request model for searching documents related to a mind map concept."""
    concept: str
    cache_hashes: List[str]
    top_k: int = Field(default=5, ge=1, le=20)

class TestGitlabTokenRequest(BaseModel):
    """Request model for testing GitLab access token."""
    token: str
    test_url: str = "https://gitlab.com/api/v4/user"
