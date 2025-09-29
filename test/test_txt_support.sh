#!/bin/bash

# Test script for TXT file support and multiple file uploads

echo "Testing LangChain Multi-Document RAG API - TXT Support & Multiple Files"
echo "======================================================================"

API_BASE_URL="http://localhost:8000/api"

echo ""
echo "1. Testing multiple file upload (TXT + Markdown)..."
echo "---------------------------------------------------"

# Test multiple file upload
curl -X POST "$API_BASE_URL/rag/upload" \
  -F "files=@documents/sample_ml_text.txt" \
  -F "files=@documents/deep_learning_guide.md"

echo ""
echo "2. Testing TXT file from URL..."
echo "-------------------------------"

# Create a test TXT URL endpoint (you can replace with a real URL)
echo "Note: Replace with actual TXT file URL for testing"
echo "Example URL format: https://raw.githubusercontent.com/user/repo/main/file.txt"

curl -X POST "$API_BASE_URL/rag" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_urls": ["https://raw.githubusercontent.com/microsoft/vscode/main/README.md"]
  }'

echo ""
echo "3. Testing health check..."
echo "-------------------------"

curl -X GET "$API_BASE_URL/health"

echo ""
echo "4. Testing cache list (should show new entries)..."
echo "-------------------------------------------------"

curl -X GET "$API_BASE_URL/cache/list"

echo ""
echo "5. Testing global search with detailed analysis..."
echo "------------------------------------------------"

curl -X POST "$API_BASE_URL/rag/global" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key concepts and applications mentioned in the documents? Provide a comprehensive analysis.",
    "detailed_analysis": true,
    "max_docs_per_cache": 8
  }'

echo ""
echo "6. Testing global search with summary analysis..."
echo "-----------------------------------------------"

curl -X POST "$API_BASE_URL/rag/global" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main topics covered in the uploaded documents?",
    "detailed_analysis": false,
    "max_docs_per_cache": 5
  }'

echo ""
echo "7. Testing URL analysis with detailed mode..."
echo "---------------------------------------------"

curl -X POST "$API_BASE_URL/rag/url" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://raw.githubusercontent.com/microsoft/vscode/main/README.md"],
    "query": "What is VS Code and what are its main features and capabilities?",
    "detailed": true
  }'

echo ""
echo "Testing completed!"
echo "=================="
echo ""
echo "To test via web UI:"
echo "1. Open http://localhost:8000 in your browser"
echo "2. Try uploading multiple files at once:"
echo "   - Select both documents/sample_ml_text.txt and documents/deep_learning_guide.md"
echo "   - Click 'Process from Files'"
echo "3. Try the new 'Direct URL Analysis' section with detailed analysis enabled"
echo "4. Use the 'AI-Powered Document Search' with 'Deep Thinking' radio button selected"
echo "5. Check the cache management section for new entries"

echo ""
echo "Sample TXT URLs you can test:"
echo "- https://raw.githubusercontent.com/microsoft/vscode/main/README.md"
echo "- https://raw.githubusercontent.com/python/cpython/main/README.rst (will be processed as text)"
