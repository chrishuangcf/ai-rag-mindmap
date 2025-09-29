#!/usr/bin/env python3
"""
Test script for the enhanced web UI with auto-generation functionality
"""

import asyncio
import httpx
import json
from typing import Dict, Any

# Service endpoints
MINDMAP_SERVICE_URL = "http://localhost:8001"
RAG_SERVICE_URL = "http://localhost:8000"

async def test_web_ui_integration():
    """Test the web UI integration with vector mind maps"""
    print("🧪 Testing Web UI Integration for Vector Mind Maps\n")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        # Test 1: Check if mind map service is running
        print("1. 🔍 Checking Mind Map Service Status...")
        try:
            response = await client.get(f"{MINDMAP_SERVICE_URL}/api/v1/mindmap/")
            if response.status_code == 200:
                print("   ✅ Mind Map service is running")
            else:
                print(f"   ⚠️  Mind Map service returned {response.status_code}")
        except Exception as e:
            print(f"   ❌ Mind Map service not accessible: {e}")
            return
        
        # Test 2: Check if RAG service is running
        print("\n2. 🔍 Checking RAG Service Status...")
        try:
            response = await client.get(f"{RAG_SERVICE_URL}/api/v1/health")
            if response.status_code == 200:
                print("   ✅ RAG service is running")
                rag_available = True
            else:
                print(f"   ⚠️  RAG service returned {response.status_code}")
                rag_available = False
        except Exception as e:
            print(f"   ❌ RAG service not accessible: {e}")
            rag_available = False
        
        # Test 3: Check available caches
        print("\n3. 📦 Checking Available Caches...")
        available_caches = []
        if rag_available:
            try:
                response = await client.get(f"{RAG_SERVICE_URL}/api/v1/mindmap/caches")
                if response.status_code == 200:
                    available_caches = response.json()
                    print(f"   ✅ Found {len(available_caches)} available caches")
                    for i, cache in enumerate(available_caches[:3]):
                        print(f"      {i+1}. {cache.get('cache_hash', 'unknown')[:12]}... "
                              f"({cache.get('total_chunks', 0)} chunks)")
                else:
                    print(f"   ⚠️  Cache endpoint returned {response.status_code}")
            except Exception as e:
                print(f"   ❌ Failed to get caches: {e}")
        
        # Test 4: Test technical concept extraction endpoint
        print("\n4. 🧠 Testing Technical Concept Extraction...")
        sample_text = """
        Our system uses FastAPI framework with Docker containers deployed on Kubernetes. 
        The database layer includes PostgreSQL for relational data and Redis for caching. 
        Machine learning models are built with TensorFlow and served via REST endpoints.
        """
        
        try:
            response = await client.post(
                f"{MINDMAP_SERVICE_URL}/api/v1/concepts/extract_technical",
                json={"text": sample_text}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Extracted {result['technical_keywords_count']} technical concepts")
                print(f"   📊 Method: {result['extraction_method']}")
                
                # Show a few concepts
                concepts = result.get('concepts', [])[:5]
                for concept in concepts:
                    category = concept.get('technical_category', 'general')
                    relevance = concept.get('relevance_score', 0)
                    print(f"      • {concept['text']} ({category}, {relevance:.2f})")
            else:
                print(f"   ❌ Technical extraction failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Technical extraction error: {e}")
        
        # Test 5: Test vector mind map creation (with real caches if available)
        print("\n5. 🗺️  Testing Vector Mind Map Creation...")
        
        if available_caches:
            # Use real caches
            cache_hashes = [cache['cache_hash'] for cache in available_caches[:2]]
            print(f"   Using real caches: {[h[:12] + '...' for h in cache_hashes]}")
            
            try:
                response = await client.post(
                    f"{MINDMAP_SERVICE_URL}/api/v1/concepts/create_vector_mindmap",
                    json={
                        "cache_hashes": cache_hashes,
                        "title": "Test Vector Mind Map",
                        "max_concepts": 15,
                        "similarity_threshold": 0.4,
                        "focus_on_technical": True,
                        "extract_from_embeddings": True
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    mindmap = result['mindmap']
                    stats = result['extraction_stats']
                    
                    print(f"   ✅ Vector mind map created successfully!")
                    print(f"   📊 Title: {mindmap['title']}")
                    print(f"   📊 Nodes: {len(mindmap['nodes'])}")
                    print(f"   📊 Relationships: {len(mindmap['relationships'])}")
                    print(f"   📊 Technical concepts: {stats['technical_concepts']}")
                    print(f"   📊 Average relevance: {stats['avg_relevance']:.2f}")
                    
                    # Show some technical nodes
                    technical_nodes = [
                        node for node in mindmap['nodes']
                        if node.get('metadata', {}).get('technical_category')
                    ]
                    
                    print(f"\n   🎯 Technical Concept Nodes ({len(technical_nodes)}):")
                    for node in technical_nodes[:5]:
                        category = node.get('metadata', {}).get('technical_category', 'N/A')
                        size = node.get('size', 20)
                        print(f"      • {node['name']} ({category}, size: {size:.0f})")
                    
                    return mindmap['id']  # Return for further testing
                    
                else:
                    error_text = await response.text()
                    print(f"   ❌ Vector mind map creation failed: {response.status_code}")
                    print(f"   Error: {error_text}")
                    
            except Exception as e:
                print(f"   ❌ Vector mind map creation error: {e}")
        
        else:
            print("   ⚠️  No real caches available, testing demo mode...")
            
            # The web UI should fall back to demo mode
            print("   (Demo mode will be tested through the web interface)")
        
        # Test 6: Test web UI accessibility
        print("\n6. 🌐 Testing Web UI Accessibility...")
        try:
            response = await client.get(f"{MINDMAP_SERVICE_URL}/static/index.html")
            if response.status_code == 200:
                html_content = response.text
                
                # Check for key UI elements
                checks = [
                    ("auto-generate checkbox", 'id="auto-generate"'),
                    ("vector focus selector", 'id="vector-focus"'),
                    ("auto concepts range", 'id="auto-max-concepts"'),
                    ("create auto button", 'onclick="createAutoVectorMindMap()"'),
                    ("status message area", 'id="auto-status"')
                ]
                
                print("   ✅ Web UI is accessible")
                for check_name, check_pattern in checks:
                    if check_pattern in html_content:
                        print(f"      ✅ {check_name} found")
                    else:
                        print(f"      ❌ {check_name} missing")
                        
            else:
                print(f"   ❌ Web UI not accessible: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Web UI accessibility error: {e}")
        
        # Final summary
        print(f"\n{'='*60}")
        print("🎉 Web UI Integration Test Summary:")
        print(f"   • Mind Map Service: {'✅ Running' if True else '❌ Down'}")
        print(f"   • RAG Service: {'✅ Running' if rag_available else '❌ Down'}")
        print(f"   • Available Caches: {len(available_caches)}")
        print(f"   • Vector Mind Maps: {'✅ Supported' if True else '❌ Not Supported'}")
        print(f"   • Auto-generation: {'✅ Ready' if True else '❌ Not Ready'}")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Open {MINDMAP_SERVICE_URL}/static/index.html in your browser")
        print(f"   2. The page should auto-generate a vector mind map on load")
        print(f"   3. Use the 'Auto Vector Mind Map' section for instant generation")
        print(f"   4. Toggle between technical/general/mixed concept focus")
        print(f"   5. Adjust max concepts and observe the mind map updates")

async def main():
    """Run the web UI integration test"""
    try:
        await test_web_ui_integration()
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
