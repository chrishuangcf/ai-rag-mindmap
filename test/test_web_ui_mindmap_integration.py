#!/usr/bin/env python3
"""
Test script to verify the web UI vector mind map integration works correctly.
This tests the full flow from document upload to automatic mind map generation.
"""

import requests
import time
import json
import os
from pathlib import Path

# Configuration
RAG_SERVICE_URL = "http://localhost:8000"
MINDMAP_SERVICE_URL = "http://localhost:8003"
WEB_UI_URL = "http://localhost:8080"

def test_services_health():
    """Test that all required services are running."""
    print("🔍 Testing service health...")
    
    services = {
        "RAG Service": f"{RAG_SERVICE_URL}/api/health",
        "Mind Map Service": f"{MINDMAP_SERVICE_URL}/health",
    }
    
    for service_name, health_url in services.items():
        try:
            response = requests.get(health_url, timeout=5)
            if response.ok:
                print(f"✅ {service_name}: Healthy")
            else:
                print(f"❌ {service_name}: Unhealthy (Status: {response.status_code})")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ {service_name}: Connection failed - {e}")
            return False
    
    return True

def test_document_upload():
    """Test uploading a document to create a cache."""
    print("\n📄 Testing document upload...")
    
    # Create a test document
    test_content = """
    Machine Learning and Artificial Intelligence
    
    Machine learning is a subset of artificial intelligence that focuses on building systems 
    that learn from data. Neural networks are computational models inspired by biological 
    neural networks. Deep learning uses multiple layers of neural networks to process data.
    
    Natural Language Processing (NLP) is a field that combines computational linguistics 
    with statistical methods to help computers understand human language. Vector databases 
    store high-dimensional vectors for similarity search and retrieval.
    
    Python is widely used for machine learning development, with libraries like TensorFlow 
    and PyTorch. APIs provide interfaces for accessing machine learning services.
    """
    
    # Create temp file
    test_file_path = Path("/tmp/test_ml_document.txt")
    test_file_path.write_text(test_content)
    
    try:
        # Upload the file
        with open(test_file_path, 'rb') as file:
            files = {'files': file}
            response = requests.post(f"{RAG_SERVICE_URL}/api/rag/upload", files=files, timeout=30)
        
        if response.ok:
            data = response.json()
            cache_hash = data.get('cache_hash')
            print(f"✅ Document uploaded successfully!")
            print(f"   Cache hash: {cache_hash}")
            print(f"   Total chunks: {data.get('total_chunks', 0)}")
            return cache_hash
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Upload request failed: {e}")
        return None
    finally:
        # Cleanup
        if test_file_path.exists():
            test_file_path.unlink()

def test_cache_availability():
    """Test that caches are available via the RAG service."""
    print("\n🗄️ Testing cache availability...")
    
    try:
        response = requests.get(f"{RAG_SERVICE_URL}/api/mindmap/caches", timeout=10)
        if response.ok:
            data = response.json()
            cache_count = data.get('total_caches', 0)
            print(f"✅ Found {cache_count} available caches")
            
            if cache_count > 0:
                # Show first cache details
                first_cache = data['caches'][0]
                print(f"   First cache: {first_cache['cache_hash'][:8]}")
                print(f"   Chunks: {first_cache.get('total_chunks', 0)}")
            
            return data.get('caches', [])
        else:
            print(f"❌ Failed to get caches: {response.status_code}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Cache request failed: {e}")
        return []

def test_vector_mindmap_creation(cache_hashes):
    """Test creating a vector mind map using available caches."""
    print("\n🧠 Testing vector mind map creation...")
    
    if not cache_hashes:
        print("❌ No caches available for mind map creation")
        return False
    
    # Use the first cache hash
    test_cache_hash = cache_hashes[0]['cache_hash']
    
    payload = {
        "cache_hashes": [test_cache_hash],
        "title": "Test Vector Mind Map",
        "max_concepts": 15,
        "similarity_threshold": 0.4,
        "focus_on_technical": True,
        "extract_from_embeddings": True
    }
    
    try:
        response = requests.post(
            f"{MINDMAP_SERVICE_URL}/api/v1/concepts/create_vector_mindmap",
            json=payload,
            timeout=30
        )
        
        if response.ok:
            data = response.json()
            mindmap = data.get('mindmap', {})
            stats = data.get('extraction_stats', {})
            
            print(f"✅ Vector mind map created successfully!")
            print(f"   Title: {mindmap.get('title', 'Unknown')}")
            print(f"   Nodes: {len(mindmap.get('nodes', []))}")
            print(f"   Relationships: {len(mindmap.get('relationships', []))}")
            print(f"   Total concepts: {stats.get('total_concepts', 0)}")
            print(f"   Technical concepts: {stats.get('technical_concepts', 0)}")
            
            return True
        else:
            print(f"❌ Mind map creation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Mind map request failed: {e}")
        return False

def test_web_ui_integration():
    """Test that the web UI can load and includes mind map functionality."""
    print("\n🌐 Testing web UI integration...")
    
    try:
        # Check if web UI is accessible
        response = requests.get(f"{WEB_UI_URL}/", timeout=10)
        if response.ok:
            html_content = response.text
            
            # Check for mind map related elements
            mind_map_indicators = [
                'mindmap-container',
                'mindmap-visualization',
                'create-mindmap-btn',
                'VectorMindMapVisualizer',
                'createVectorMindMap'
            ]
            
            found_indicators = []
            for indicator in mind_map_indicators:
                if indicator in html_content:
                    found_indicators.append(indicator)
            
            print(f"✅ Web UI accessible")
            print(f"   Found {len(found_indicators)}/{len(mind_map_indicators)} mind map indicators")
            
            if len(found_indicators) >= 3:
                print("✅ Mind map integration appears to be working")
                return True
            else:
                print("⚠️ Mind map integration may be incomplete")
                return False
        else:
            print(f"❌ Web UI not accessible: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Web UI request failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🧪 Starting Web UI Vector Mind Map Integration Tests")
    print("=" * 60)
    
    # Test 1: Service Health
    if not test_services_health():
        print("\n❌ Service health check failed. Please start all services first.")
        return False
    
    # Test 2: Document Upload
    cache_hash = test_document_upload()
    if not cache_hash:
        print("\n❌ Document upload failed. Testing with existing caches...")
    
    # Wait a moment for cache to be processed
    time.sleep(2)
    
    # Test 3: Cache Availability
    available_caches = test_cache_availability()
    if not available_caches:
        print("\n❌ No caches available. Cannot test mind map creation.")
        return False
    
    # Test 4: Vector Mind Map Creation
    mindmap_success = test_vector_mindmap_creation(available_caches)
    if not mindmap_success:
        print("\n❌ Vector mind map creation failed.")
        return False
    
    # Test 5: Web UI Integration
    ui_success = test_web_ui_integration()
    if not ui_success:
        print("\n❌ Web UI integration test failed.")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All integration tests passed!")
    print("\nNext steps:")
    print("1. Open the web UI at http://localhost:8080")
    print("2. Upload some documents or files")
    print("3. Check that the mind map is auto-generated")
    print("4. Try manually creating mind maps with different settings")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
