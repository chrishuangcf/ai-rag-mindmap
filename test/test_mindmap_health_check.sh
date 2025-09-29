#!/bin/bash

# Test script for Mind Map Service Health Check

echo "🧪 Testing Mind Map Service Health Check Implementation"
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test main service health check
echo -e "\n${BLUE}1. Testing Main Service Health Check...${NC}"
curl -s http://localhost:8000/api/health || echo -e "${RED}❌ Main service not responding${NC}"

# Test mindmap service health check
echo -e "\n${BLUE}2. Testing Mind Map Service Health Check...${NC}"
curl -s http://localhost:8003/health || echo -e "${RED}❌ Mind map service not responding${NC}"

# Test mindmap service route health check
echo -e "\n${BLUE}3. Testing Mind Map Service Route Health Check...${NC}"
curl -s http://localhost:8003/api/v1/mindmap/health || echo -e "${RED}❌ Mind map service routes not responding${NC}"

# Test batch queue status
echo -e "\n${BLUE}4. Testing Batch Queue Status...${NC}"
curl -s http://localhost:8003/api/v1/batch/ || echo -e "${RED}❌ Batch queue status not available${NC}"

# Test web UI access
echo -e "\n${BLUE}5. Testing Web UI Access...${NC}"
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/static/index.html | grep -q "200" && echo -e "${GREEN}✅ Web UI accessible${NC}" || echo -e "${RED}❌ Web UI not accessible${NC}"

echo -e "\n${YELLOW}💡 To test the health check UI:${NC}"
echo "1. Open http://localhost:8000/static/index.html in your browser"
echo "2. Check the 'API Health' section at the top"
echo "3. Toggle auto-refresh on/off to see real-time updates"
echo "4. Look for Mind Map Service status with processing information"

echo -e "\n${YELLOW}🔧 To test processing status:${NC}"
echo "1. Create a large mindmap to trigger batch processing:"
echo "   curl -X POST http://localhost:8003/api/v1/mindmap/ \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"title\": \"Test Large Map\", \"cache_hashes\": [\"hash1\", \"hash2\", \"hash3\", \"hash4\", \"hash5\", \"hash6\"], \"max_nodes\": 300}'"
echo "2. Observe the health check showing 'processing' status"
echo "3. Watch the auto-refresh update the status in real-time"

echo -e "\n${GREEN}✅ Health check test script completed!${NC}"
