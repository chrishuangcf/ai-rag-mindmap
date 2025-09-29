"""
Components Module

Modular components for the RAG system:
- StorageManager: Handles Redis caching, vector storage, and data persistence
- RAGProcessor: Manages LangChain operations, embeddings, and RAG processing
- FileIOManager: Handles file uploads and URL loading operations
- GitLabManager: Manages GitLab authentication and API operations
"""

from .storage_manager import StorageManager
from .rag_processor import RAGProcessor
from .file_io_manager import FileIOManager
from .gitlab_manager import GitLabManager

__all__ = [
    "StorageManager",
    "RAGProcessor", 
    "FileIOManager",
    "GitLabManager"
]

from .storage_manager import StorageManager
from .rag_processor import RAGProcessor
from .file_io_manager import FileIOManager
from .gitlab_manager import GitLabManager

__all__ = [
    "StorageManager",
    "RAGProcessor", 
    "FileIOManager",
    "GitLabManager"
]
