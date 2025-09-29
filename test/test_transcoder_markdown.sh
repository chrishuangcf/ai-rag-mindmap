#!/bin/bash

# Test script for transcoder service with Markdown URL support
# This script tests the new Markdown URL functionality

echo "🧪 Testing Transcoder Service - Markdown URL Support"
echo "=================================================="

BASE_URL="http://localhost:8002"

# Test 1: Health check
echo -e "\n1️⃣ Testing health endpoint..."
curl -s "$BASE_URL/health" | python3 -m json.tool

# Test 2: Markdown URL via optimized endpoint
echo -e "\n2️⃣ Testing optimized Markdown URL endpoint..."
curl -s -X POST "$BASE_URL/transcode/markdown" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://raw.githubusercontent.com/fastapi/fastapi/master/README.md"
  }' | python3 -m json.tool

# Test 3: Markdown URL via general transcode endpoint
echo -e "\n3️⃣ Testing Markdown URL via general transcode endpoint..."
curl -s -X POST "$BASE_URL/transcode" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md"
  }' | python3 -m json.tool

# Test 4: Test with GitHub blob URL (should work)
echo -e "\n4️⃣ Testing GitHub blob URL (should convert to raw URL)..."
curl -s -X POST "$BASE_URL/transcode/markdown" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://github.com/microsoft/vscode/blob/main/README.md"
  }' | python3 -m json.tool

# Test 5: Test with non-Markdown URL (should fail on optimized endpoint)
echo -e "\n5️⃣ Testing non-Markdown URL on optimized endpoint (should fail)..."
curl -s -X POST "$BASE_URL/transcode/markdown" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/document.pdf"
  }' | python3 -m json.tool

echo -e "\n✅ Testing complete!"
echo "📝 Note: Make sure the transcoder service is running on port 8002"
echo "🚀 Start with: docker-compose up transcoder"
