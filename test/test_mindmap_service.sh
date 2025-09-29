#!/bin/bash

# Mind Map Service Test Script
# This script tests the basic functionality of the mind map service

set -e

echo "🧪 Testing Mind Map Service"
echo "=========================="

# Configuration
BASE_URL="http://localhost:8003"
API_BASE="$BASE_URL/api/v1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Test service health
test_health() {
    log_info "Testing service health..."
    
    # Test main health endpoint
    if curl -s -f "$BASE_URL/health" > /dev/null; then
        log_success "Main health endpoint responding"
    else
        log_error "Main health endpoint failed"
        return 1
    fi
    
    # Test API documentation
    if curl -s -f "$BASE_URL/docs" > /dev/null; then
        log_success "API documentation accessible"
    else
        log_warning "API documentation not accessible"
    fi
    
    log_success "Health checks passed"
}

# Test Neo4j connectivity
test_neo4j() {
    log_info "Testing Neo4j connectivity..."
    
    # This would require a test endpoint in the service
    # For now, just check if Neo4j container is running
    if docker ps | grep -q "neo4j-mindmap"; then
        log_success "Neo4j container is running"
    else
        log_error "Neo4j container not found"
        return 1
    fi
}

# Test batch queue
test_batch_queue() {
    log_info "Testing batch queue..."
    
    response=$(curl -s "$API_BASE/batch/" || echo "failed")
    
    if [[ "$response" == "failed" ]]; then
        log_error "Failed to get batch queue status"
        return 1
    fi
    
    log_success "Batch queue accessible"
    echo "Queue status: $response"
}

# Test mind map creation (requires actual cache data)
test_mindmap_creation() {
    log_info "Testing mind map creation..."
    
    # This is a mock test - in reality you'd need valid cache hashes
    log_warning "Mind map creation test requires valid cache hashes from RAG system"
    log_info "To test manually:"
    echo "  1. Ensure RAG system has cached data"
    echo "  2. Get cache hashes from: curl http://localhost:8000/api/caches"
    echo "  3. Use cache hashes to create mind map via UI or API"
}

# Test static files
test_static_files() {
    log_info "Testing static file serving..."
    
    if curl -s -f "$BASE_URL/static/index.html" > /dev/null; then
        log_success "Static HTML accessible"
    else
        log_error "Static HTML not accessible"
        return 1
    fi
    
    if curl -s -f "$BASE_URL/static/mindmap.js" > /dev/null; then
        log_success "Static JavaScript accessible"
    else
        log_error "Static JavaScript not accessible"
        return 1
    fi
}

# Test RAG service integration
test_rag_integration() {
    log_info "Testing RAG service integration..."
    
    # Check if main RAG service is accessible
    if curl -s -f "http://localhost:8000/api/caches" > /dev/null; then
        log_success "RAG service is accessible"
        
        # Get available caches
        caches=$(curl -s "http://localhost:8000/api/caches" | head -c 200)
        log_info "Available caches (first 200 chars): $caches"
    else
        log_warning "RAG service not accessible - mind map service will have limited functionality"
    fi
}

# Main test execution
main() {
    echo "Starting Mind Map Service tests..."
    echo "Waiting for services to be ready..."
    sleep 5
    
    # Run tests
    test_health || exit 1
    test_neo4j || exit 1
    test_batch_queue || exit 1
    test_static_files || exit 1
    test_rag_integration
    test_mindmap_creation
    
    echo ""
    log_success "🎉 Basic tests completed!"
    echo ""
    echo "Next steps:"
    echo "1. Open the Mind Map UI: $BASE_URL/static/index.html"
    echo "2. Check API docs: $BASE_URL/docs"
    echo "3. Monitor logs: docker-compose logs -f mindmap"
    echo "4. Neo4j browser: http://localhost:7474 (neo4j/mindmapneo4j)"
    echo ""
}

# Handle script arguments
case "${1:-}" in
    --health)
        test_health
        ;;
    --neo4j)
        test_neo4j
        ;;
    --batch)
        test_batch_queue
        ;;
    --static)
        test_static_files
        ;;
    --rag)
        test_rag_integration
        ;;
    --help)
        echo "Usage: $0 [--health|--neo4j|--batch|--static|--rag|--help]"
        echo "  --health   Test service health endpoints"
        echo "  --neo4j    Test Neo4j connectivity"
        echo "  --batch    Test batch queue functionality"
        echo "  --static   Test static file serving"  
        echo "  --rag      Test RAG service integration"
        echo "  --help     Show this help message"
        echo ""
        echo "Run without arguments to execute all tests"
        ;;
    *)
        main
        ;;
esac
