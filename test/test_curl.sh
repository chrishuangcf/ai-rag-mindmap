#!/bin/bash
# Complete cURL Testing Guide for LangChain RAG API
# Make sure to start the server first: uvicorn main:app --reload

echo "=== LangChain RAG API cURL Testing Guide ==="
echo

# Set the base URL
BASE_URL="http://localhost:8000"

echo "1. Health Check"
echo "==============="
echo "curl -X GET \"$BASE_URL/health\""
echo
curl -X GET "$BASE_URL/health"
echo
echo

echo "2. Test RAG with GitHub Blob URL (Auto-converts to raw URL)"
echo "==========================================================="
echo "This tests the exact same URL you had issues with:"
echo
CURL_CMD='curl -X POST "'$BASE_URL'/rag" \
  -H "Content-Type: application/json" \
  -d '"'"'{\
    "repo_urls": [\
      "https://github.com/chrishuangcf/wiim-mini-ui/blob/main/README.md"\
    ],\
    "query": "What dependencies do these projects have?"\
  }'"'"''

echo "$CURL_CMD"
echo
echo "Executing..."
eval "$CURL_CMD"
echo
echo

echo "3. Test Auto RAG (Smart caching)"
echo "==============================="
echo "This should use the cache from the previous request:"
echo
CURL_CMD2='curl -X POST "'$BASE_URL'/rag/auto" \
  -H "Content-Type: application/json" \
  -d '"'"'{\
    "repo_urls": [\
      "https://github.com/chrishuangcf/wiim-mini-ui/blob/main/README.md"\
    ],\
    "query": "What is this project about?"\
  }'"'"''

echo "$CURL_CMD2"
echo
echo "Executing..."
eval "$CURL_CMD2"
echo
echo

echo "4. List Cached Documents"
echo "======================="
echo "curl -X GET \"$BASE_URL/cache/list\""
echo
curl -X GET "$BASE_URL/cache/list"
echo
echo

echo "5. Test with a different GitHub URL"
echo "===================================================="
CURL_CMD3='curl -X POST "'$BASE_URL'/rag" \
  -H "Content-Type: application/json" \
  -d '"'"'{\
    "repo_urls": [\
      "https://github.com/langchain-ai/langchain/blob/master/README.md"\
    ],\
    "query": "What is LangChain?"\
  }'"'"''

echo "$CURL_CMD3"
echo
echo "Executing..."
eval "$CURL_CMD3"
echo
echo

echo "6. Search Caches"
echo "==============="
echo "Search for caches containing 'wiim':"
echo "curl -X GET \"$BASE_URL/cache/search?url_pattern=wiim\""
echo
curl -X GET "$BASE_URL/cache/search?url_pattern=wiim"
echo
echo

echo "=== Testing Complete ==="
echo
echo "Key points about GitHub URL support:"
echo "- Blob URLs are automatically converted to raw URLs"
echo "- Mixed blob/raw URLs work seamlessly"  
echo "- URL conversion info is shown in responses"
echo "- Caching works with both URL formats"
echo
echo "If you got errors, check:"
echo "1. Server is running: uvicorn main:app --reload"
echo "2. Environment variables are set (OPENAI_API_KEY, OPENAI_API_BASE)"
echo "3. Dependencies are installed: pip install -r requirements.txt"
