#!/bin/bash

# Test script for AI-powered search with Google search fallback functionality
# Make sure your server is running on localhost:8000

echo "Testing AI-Powered Search with Google Search Fallback..."
echo "======================================================="

# Test the Google fallback functionality
echo "1. Testing Google fallback detection and search..."
curl -X POST "http://localhost:8000/test/google-fallback" \
  -H "Content-Type: application/json" \
  -d '{"query": "latest artificial intelligence trends 2025"}' \
  | jq '.'

echo ""
echo "2. Testing AI-powered search with a question that might trigger fallback..."
curl -X POST "http://localhost:8000/rag/global" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest developments in quantum computing in 2025?"}' \
  | jq '.'

echo ""
echo "3. Testing health check to verify Google search availability..."
curl -X GET "http://localhost:8000/health" | jq '.'

echo ""
echo "4. Testing AI-powered search with a programming question..."
curl -X POST "http://localhost:8000/rag/global" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I implement a REST API in Python?"}' \
  | jq '.'

echo ""
echo "AI-powered search test completed!"
