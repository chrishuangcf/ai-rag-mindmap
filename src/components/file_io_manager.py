"""
File I/O Manager Component

Handles all file and URL operations including:
- File uploads and processing
- URL content fetching
- Document format conversion
- Content validation and preprocessing
"""

import os
import requests
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from fastapi import UploadFile, HTTPException
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

from src.utils.url_utils import validate_and_convert_urls


class FileIOManager:
    """
    Manages all file I/O operations including uploads, URL fetching,
    and document conversion.
    """
    
    def __init__(self):
        self.allowed_extensions = {"pdf", "docx", "md", "markdown", "txt"}
        self.transcoder_url = "http://transcoder-app:5001/transcode"
        self.timeout = 30
    
    def validate_file_extension(self, filename: str) -> bool:
        """
        Validate if file extension is supported.
        
        Args:
            filename: Name of the file
            
        Returns:
            True if extension is supported, False otherwise
        """
        if not filename:
            return False
        
        file_extension = filename.split(".")[-1].lower()
        return file_extension in self.allowed_extensions
    
    def get_file_extension(self, filename: str) -> str:
        """
        Get the file extension from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            File extension in lowercase
        """
        if not filename:
            return ""
        return filename.split(".")[-1].lower()
    
    def is_text_based_file(self, filename: str) -> bool:
        """
        Check if file is text-based (can be processed directly).
        
        Args:
            filename: Name of the file
            
        Returns:
            True if file is text-based, False otherwise
        """
        extension = self.get_file_extension(filename)
        return extension in {"md", "markdown", "txt"}
    
    def is_binary_file(self, filename: str) -> bool:
        """
        Check if file is binary (needs transcoding).
        
        Args:
            filename: Name of the file
            
        Returns:
            True if file is binary, False otherwise
        """
        extension = self.get_file_extension(filename)
        return extension in {"pdf", "docx"}
    
    def process_text_file(self, file: UploadFile) -> Optional[Document]:
        """
        Process a text-based file directly.
        
        Args:
            file: Uploaded file object
            
        Returns:
            Document object or None if processing failed
        """
        try:
            content = file.file.read()
            file_content = content.decode("utf-8")
            
            if not file_content.strip():
                print(f"Warning: File '{file.filename}' is empty, skipping")
                return None
            
            return Document(
                page_content=file_content, 
                metadata={"source_url": file.filename}
            )
            
        except Exception as e:
            print(f"Error processing text file {file.filename}: {e}")
            return None
    
    def process_binary_file(self, file: UploadFile) -> Optional[Document]:
        """
        Process a binary file through the transcoder service.
        
        Args:
            file: Uploaded file object
            
        Returns:
            Document object or None if processing failed
        """
        try:
            # Prepare for transcoding
            files_data = {'file': (file.filename, file.file, file.content_type)}
            
            # Send to transcoder service
            response = requests.post(self.transcoder_url, files=files_data, timeout=self.timeout)
            response.raise_for_status()
            
            transcoded_data = response.json()
            markdown_content = transcoded_data.get("markdown_content")
            
            if not markdown_content or not markdown_content.strip():
                print(f"Warning: File '{file.filename}' produced no content after transcoding, skipping")
                return None
            
            return Document(
                page_content=markdown_content, 
                metadata={
                    "source_url": file.filename,
                    "original_format": self.get_file_extension(file.filename),
                    "transcoded": True
                }
            )
            
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to transcoder service for {file.filename}: {e}")
            return None
        except Exception as e:
            print(f"Error processing binary file {file.filename}: {e}")
            return None
    
    def process_uploaded_files(self, files: List[UploadFile]) -> Tuple[List[Document], List[str], List[str]]:
        """
        Process multiple uploaded files.
        
        Args:
            files: List of uploaded files
            
        Returns:
            Tuple of (documents, processed_filenames, skipped_filenames)
        """
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Validate all files first
        for file in files:
            if not file.filename:
                raise HTTPException(status_code=400, detail="One or more files have no filename")
            
            if not self.validate_file_extension(file.filename):
                file_extension = self.get_file_extension(file.filename)
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: '{file_extension}' in file '{file.filename}'. "
                           f"Please upload PDF, DOCX, TXT, or Markdown files."
                )
        
        documents = []
        processed_files = []
        skipped_files = []
        
        for file in files:
            try:
                doc = None
                
                if self.is_text_based_file(file.filename or ""):
                    doc = self.process_text_file(file)
                elif self.is_binary_file(file.filename or ""):
                    doc = self.process_binary_file(file)
                
                if doc:
                    documents.append(doc)
                    processed_files.append(file.filename)
                else:
                    skipped_files.append(file.filename)
                    
            except Exception as e:
                print(f"Error processing file {file.filename}: {e}")
                skipped_files.append(file.filename)
        
        if not documents:
            raise HTTPException(
                status_code=400, 
                detail="No valid content could be extracted from the uploaded files"
            )
        
        return documents, processed_files, skipped_files
    
    def fetch_url_content(self, url: str, gitlab_token: Optional[str] = None) -> Optional[str]:
        """
        Fetch content from a URL.
        
        Args:
            url: URL to fetch
            gitlab_token: Optional GitLab token for authentication
            
        Returns:
            Content string or None if fetch failed
        """
        try:
            # Note: For GitLab URLs with token, this would need async handling
            # For now, use synchronous requests
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
            
        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to fetch {url}: {e}")
            return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def is_gitlab_url(self, url: str) -> bool:
        """
        Check if URL is from GitLab.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL is from GitLab, False otherwise
        """
        return 'gitlab' in url.lower()
    
    def create_document_from_url(self, url: str, content: str) -> Document:
        """
        Create a Document from URL content.
        
        Args:
            url: Source URL
            content: Content string
            
        Returns:
            Document object
        """
        # Determine file type from URL
        url_lower = url.lower()
        file_type = "unknown"
        
        if any(ext in url_lower for ext in ['.md', '.markdown']):
            file_type = "markdown"
        elif '.txt' in url_lower:
            file_type = "text"
        elif '.py' in url_lower:
            file_type = "python"
        elif '.js' in url_lower:
            file_type = "javascript"
        elif '.json' in url_lower:
            file_type = "json"
        elif '.yaml' in url_lower or '.yml' in url_lower:
            file_type = "yaml"
        
        return Document(
            page_content=content,
            metadata={
                "source_url": url,
                "file_type": file_type,
                "fetched_from_url": True
            }
        )
    
    def process_url_with_text_loader(self, url: str, content: str) -> List[Document]:
        """
        Process URL content using TextLoader as fallback.
        
        Args:
            url: Source URL
            content: Content string
            
        Returns:
            List of Document objects
        """
        try:
            # Save content to temporary file
            with tempfile.NamedTemporaryFile(
                mode='w', delete=False, suffix='.txt', encoding='utf-8'
            ) as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                loader = TextLoader(tmp_file_path, encoding='utf-8')
                docs = loader.load()
                
                # Update metadata with source URL
                for doc in docs:
                    doc.metadata["source_url"] = url
                    doc.metadata["processed_with"] = "text_loader"
                
                return docs
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
        except Exception as e:
            print(f"Error processing URL {url} with TextLoader: {e}")
            return []
    
    def process_urls(self, repo_urls: List[str], gitlab_token: Optional[str] = None) -> List[Document]:
        """
        Process multiple URLs and return documents.
        
        Args:
            repo_urls: List of URLs to process
            gitlab_token: Optional GitLab token
            
        Returns:
            List of Document objects
        """
        # Validate and convert URLs
        validated_urls = validate_and_convert_urls(repo_urls)
        
        documents = []
        
        for url in validated_urls:
            content = self.fetch_url_content(url, gitlab_token)
            
            if not content:
                continue
            
            # Determine processing method based on URL
            url_lower = url.lower()
            
            if (any(ext in url_lower for ext in ['.md', '.markdown']) or
                any(ext in url_lower for ext in ['.txt'])):
                # Handle Markdown and TXT files directly
                if content.strip():
                    doc = self.create_document_from_url(url, content)
                    documents.append(doc)
            else:
                # Use TextLoader for other file types
                docs = self.process_url_with_text_loader(url, content)
                documents.extend(docs)
        
        return documents
    
    def validate_documents(self, documents: List[Document]) -> List[Document]:
        """
        Validate documents and filter out empty or invalid ones.
        
        Args:
            documents: List of documents to validate
            
        Returns:
            List of valid documents
        """
        valid_docs = []
        
        for doc in documents:
            # Check for empty content
            if not doc.page_content or not doc.page_content.strip():
                source = doc.metadata.get('source_url', 'unknown')
                print(f"Skipping empty document from {source}")
                continue
            
            # Check minimum content length
            if len(doc.page_content.strip()) < 10:
                source = doc.metadata.get('source_url', 'unknown')
                print(f"Skipping very short document from {source}")
                continue
            
            valid_docs.append(doc)
        
        print(f"Validated {len(valid_docs)} out of {len(documents)} documents")
        return valid_docs
