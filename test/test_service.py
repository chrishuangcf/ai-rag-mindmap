import requests
import json
import time

# Test the service
base_url = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_rag_with_github_blob_url():
    """Test RAG endpoint with GitHub blob URL (should auto-convert)"""
    print("\n=== Testing RAG with GitHub Blob URL ===")
    
    rag_data = {
        "repo_urls": [
            "https://github.com/chrishuangcf/wiim-mini-ui/blob/main/README.md"
        ],
        "query": "What dependencies do these projects have?"
    }
    
    try:
        print(f"Sending request to: {base_url}/rag")
        print(f"Data: {json.dumps(rag_data, indent=2)}")
        
        response = requests.post(f"{base_url}/rag", json=rag_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Cache hash: {result.get('cache_hash', 'N/A')}")
            print(f"Answer preview: {result.get('answer', 'No answer')[:200]}...")
            
            # Show URL conversion info
            url_info = result.get('url_info', {})
            if url_info:
                print(f"URL Conversions: {url_info.get('conversions_applied', {})}")
            
            return result.get('cache_hash')
        else:
            print(f"Error response: {response.text}")
            return None
            
    except Exception as e:
        print(f"RAG test failed: {e}")
        return None

def test_rag_auto():
    """Test auto RAG endpoint"""
    print("\n=== Testing Auto RAG Endpoint ===")
    
    rag_data = {
        "repo_urls": [
            "https://github.com/chrishuangcf/wiim-mini-ui/blob/main/README.md"
        ],
        "query": "What is this project about?"
    }
    
    try:
        response = requests.post(f"{base_url}/rag/auto", json=rag_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Used cache: {result.get('cached', False)}")
            print(f"Answer preview: {result.get('answer', 'No answer')[:200]}...")
            return True
        else:
            print(f"Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Auto RAG test failed: {e}")
        return False

def test_cache_list():
    """Test cache listing"""
    print("\n=== Testing Cache List ===")
    
    try:
        response = requests.get(f"{base_url}/cache/list")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Cache count: {result.get('cached_sets', 0)}")
            caches = result.get('available_caches', [])
            for cache in caches:
                print(f"  Hash: {cache.get('hash')}")
                print(f"  URLs: {cache.get('url_preview', [])}")
            return True
        else:
            print(f"Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Cache list test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting service tests...\n")
    
    # Test health first
    if not test_health():
        print("Server is not running! Start it with: uvicorn main:app --reload")
        return
    
    # Test RAG with GitHub blob URL
    cache_hash = test_rag_with_github_blob_url()
    
    # Wait a moment for processing
    time.sleep(2)
    
    # Test auto endpoint (should use cache)
    test_rag_auto()
    
    # Test cache list
    test_cache_list()
    
    print("\n=== All Tests Complete ===")

if __name__ == "__main__":
    main()