"""
URL utilities and hashing functions for the LangChain RAG API.
"""

import hashlib

def convert_github_url_to_raw(url: str) -> str:
    """Convert GitHub blob URLs to raw URLs for direct content access."""
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url

def convert_gitlab_url_to_raw(url: str) -> str:
    """Convert GitLab blob URLs to raw URLs for direct content access."""
    if "gitlab" in url.lower() and "/-/blob/" in url:
        return url.replace("/-/blob/", "/-/raw/")
    return url

def convert_url_to_raw(url: str) -> str:
    """Convert repository URLs to their raw content format for both GitHub and GitLab."""
    # Try GitHub conversion first
    converted_url = convert_github_url_to_raw(url)
    if converted_url != url:
        return converted_url
    
    # Try GitLab conversion
    converted_url = convert_gitlab_url_to_raw(url)
    if converted_url != url:
        return converted_url
    
    # If already in raw format or unknown format, return as-is
    return url

def validate_and_convert_urls(repo_urls: list[str]) -> list[str]:
    """Validate and convert a list of URLs to their raw content format."""
    return [convert_url_to_raw(url) for url in repo_urls]

def generate_doc_hash(repo_urls: list[str]) -> str:
    """Generate a consistent hash for a set of document URLs."""
    sorted_urls = sorted(repo_urls)
    urls_string = '|'.join(sorted_urls)
    return hashlib.md5(urls_string.encode()).hexdigest()[:12]
