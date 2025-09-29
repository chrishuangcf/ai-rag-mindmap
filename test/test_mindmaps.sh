#!/bin/bash

# Mind Map Service Cleanup CLI
# Usage: ./cleanup_mindmaps.sh [option]

set -e

# Configuration
MINDMAP_SERVICE_URL="http://localhost:8003"
API_BASE="$MINDMAP_SERVICE_URL/api/v1"

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

# Function to delete all mindmaps
delete_all_mindmaps() {
    log_info "Deleting all mindmaps from the service..."
    
    response=$(curl -s -X DELETE "$API_BASE/mindmap/cleanup/all" || echo "failed")
    
    if [[ "$response" == "failed" ]]; then
        log_error "Failed to connect to mindmap service"
        return 1
    fi
    
    echo "$response" | grep -q "successfully" && {
        log_success "All mindmaps deleted successfully"
        echo "Response: $response"
    } || {
        log_error "Failed to delete mindmaps"
        echo "Response: $response"
        return 1
    }
}

# Function to delete and recreate
recreate_mindmap() {
    log_info "Deleting all mindmaps and recreating from available caches..."
    
    response=$(curl -s -X POST "$API_BASE/mindmap/recreate" || echo "failed")
    
    if [[ "$response" == "failed" ]]; then
        log_error "Failed to connect to mindmap service"
        return 1
    fi
    
    echo "$response" | grep -q "Unified Mind Map" && {
        log_success "Mindmaps recreated successfully"
        echo "Response: $response"
    } || {
        log_error "Failed to recreate mindmaps"
        echo "Response: $response"
        return 1
    }
}

# Function to check service status
check_status() {
    log_info "Checking mindmap service status..."
    
    response=$(curl -s "$MINDMAP_SERVICE_URL/health" || echo "failed")
    
    if [[ "$response" == "failed" ]]; then
        log_error "Mindmap service is not responding"
        return 1
    fi
    
    echo "$response" | grep -q '"service":"healthy"' && {
        log_success "Mindmap service is healthy"
    } || {
        log_warning "Mindmap service may have issues"
    }
    
    echo "Health status: $response"
}

# Function to list current mindmaps
list_mindmaps() {
    log_info "Listing current mindmaps..."
    
    response=$(curl -s "$API_BASE/mindmap/" || echo "failed")
    
    if [[ "$response" == "failed" ]]; then
        log_error "Failed to get mindmap list"
        return 1
    fi
    
    echo "Current mindmaps: $response"
}

# Function to clear batch queue
clear_queue() {
    log_info "Checking batch queue status..."
    
    response=$(curl -s "$API_BASE/batch/" || echo "failed")
    
    if [[ "$response" == "failed" ]]; then
        log_error "Failed to get batch queue status"
        return 1
    fi
    
    echo "Queue status: $response"
    
    # Check if there are pending jobs
    if echo "$response" | grep -q '"pending_jobs":[1-9]'; then
        log_warning "There are pending jobs in the queue"
        log_info "Use the delete_all option to clear the queue as well"
    else
        log_success "No pending jobs in queue"
    fi
}

# Main function
show_help() {
    echo "Mind Map Service Cleanup CLI"
    echo "============================"
    echo ""
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  delete_all    Delete all mindmaps (clean slate)"
    echo "  recreate      Delete all mindmaps and recreate from available caches"
    echo "  status        Check service health status"
    echo "  list          List current mindmaps"
    echo "  queue         Check batch queue status"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 delete_all   # Clean removal of all mindmaps"
    echo "  $0 recreate     # Delete and rebuild from current caches"
    echo "  $0 status       # Check if service is running properly"
}

# Main execution
case "${1:-help}" in
    delete_all)
        check_status && delete_all_mindmaps
        ;;
    recreate)
        check_status && recreate_mindmap
        ;;
    status)
        check_status
        ;;
    list)
        list_mindmaps
        ;;
    queue)
        clear_queue
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
