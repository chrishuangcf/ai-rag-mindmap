"""
RAG Processor Component

Handles all RAG operations including:
- LangChain document splitting
- Embedding generation and processing
- RAG query processing and retrieval
- LLM interactions
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from src.services.embeddings import SimpleTFIDFEmbeddings
from src.services.rag_service import get_llm, get_retrieval_chain, get_enhanced_retrieval_chain
from src.services.google_search import is_dont_know_response, perform_google_search, get_google_search_summary
from src.core.config import GOOGLE_SEARCH_AVAILABLE


class RAGProcessor:
    """
    Handles all RAG processing operations including document splitting,
    embedding generation, and query processing.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, max_features: int = 500):
        """
        Initialize the RAG processor.
        
        Args:
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            max_features: Maximum features for embeddings
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_features = max_features
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for processing.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents
        """
        if not documents:
            return []
        
        try:
            split_docs = self.text_splitter.split_documents(documents)
            print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            print(f"Error splitting documents: {e}")
            return documents
    
    def create_embeddings(self, documents: List[Document]) -> SimpleTFIDFEmbeddings:
        """
        Create embeddings from documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            SimpleTFIDFEmbeddings object
        """
        if not documents:
            raise ValueError("No documents provided for embedding creation")
        
        embeddings = SimpleTFIDFEmbeddings(max_features=self.max_features)
        print(f"Created embeddings with {self.max_features} features from {len(documents)} documents")
        return embeddings
    
    @traceable
    def process_rag_query(self, query: str, vector_store, use_enhanced: bool = False) -> Dict[str, Any]:
        """
        Process a RAG query using the vector store.
        
        Args:
            query: Query string
            vector_store: Vector store for retrieval
            use_enhanced: Whether to use enhanced retrieval chain
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            llm = get_llm()
            
            if use_enhanced:
                chain = get_enhanced_retrieval_chain(vector_store, llm)
            else:
                chain = get_retrieval_chain(vector_store, llm)
            
            response = chain.invoke({"input": query})
            
            result = {
                "answer": response.get("answer", ""),
                "query": query,
                "use_enhanced": use_enhanced,
                "source_documents": []
            }
            
            if "context" in response:
                source_docs = response["context"]
                if isinstance(source_docs, list):
                    result["source_documents"] = [
                        {
                            "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in source_docs
                    ]
            
            return result
            
        except Exception as e:
            print(f"Error processing RAG query: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "query": query,
                "error": True
            }
    
    def process_with_google_fallback(self, query: str, rag_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query with Google search fallback if RAG response is insufficient.
        
        Args:
            query: Query string
            rag_result: Result from RAG processing
            
        Returns:
            Enhanced result with Google search if needed
        """
        if not GOOGLE_SEARCH_AVAILABLE:
            return rag_result
        
        answer = rag_result.get("answer", "")
        
        if is_dont_know_response(answer):
            try:
                search_results = perform_google_search(query)
                
                if search_results:
                    google_summary = get_google_search_summary(query, search_results)
                    
                    rag_result["google_search"] = {
                        "summary": google_summary,
                        "results": search_results[:3]
                    }
                    rag_result["answer"] = f"{answer}\n\nAdditional information from web search:\n{google_summary}"
                    rag_result["enhanced_with_google"] = True
                
            except Exception as e:
                print(f"Error in Google search fallback: {e}")
        
        return rag_result
    
    def create_document_from_content(self, content: str, source: str) -> Document:
        """
        Create a Document object from content and metadata.
        
        Args:
            content: Document content
            source: Source identifier
            
        Returns:
            Document object
        """
        return Document(
            page_content=content, 
            metadata={"source_url": source}
        )
    
    def validate_documents_for_processing(self, documents: List[Document]) -> List[Document]:
        """
        Validate and filter documents before processing.
        
        Args:
            documents: List of documents to validate
            
        Returns:
            List of valid documents
        """
        valid_docs = []
        
        for doc in documents:
            if not doc.page_content or not doc.page_content.strip():
                continue
            
            if len(doc.page_content.strip()) < 10:
                continue
            
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {"source_url": "unknown"}
            
            valid_docs.append(doc)
        
        print(f"Validated {len(valid_docs)} out of {len(documents)} documents for processing")
        return valid_docs
