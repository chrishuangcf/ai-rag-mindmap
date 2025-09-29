#!/usr/bin/env python3
"""
Simple test to verify the in-memory RAG system works without Redis
"""

import requests
import json
import time

def test_in_memory_rag():
    """Test the in-memory RAG system"""
    
    base_url = "http://localhost:8000"
    
    # Test data
    test_request = {
        "repo_urls": [
            "https://github.com/chrishuangcf/wiim-mini-ui/blob/main/README.md"
        ],
        "query": "What dependencies do these projects have?"
    }
    
    print("Testing in-memory RAG system...")
    print(f"Request: {json.dumps(test_request, indent=2)}")
    
    try:
        # Test health first
        health_response = requests.get(f"{base_url}/health")
        print(f"Health: {health_response.json()}")
        
        # Test RAG auto endpoint
        print("\nTesting /rag/auto endpoint...")
        response = requests.post(f"{base_url}/rag/auto", json=test_request, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("Success!")
            print(f"Answer: {result.get('answer', 'No answer')[:200]}...")
            print(f"Cache Hash: {result.get('cache_hash', 'No hash')}")
            print(f"Cached: {result.get('cached', False)}")
            print(f"Total Chunks: {result.get('total_chunks', 0)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

if __name__ == "__main__":
    success = test_in_memory_rag()
    if success:
        print("\n✅ Test passed! In-memory RAG system is working.")
    else:
        print("\n❌ Test failed! Check the error messages above.")
