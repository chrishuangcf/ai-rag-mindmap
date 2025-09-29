#!/bin/bash

# Redis Manager Script
# Provides utilities to view and manage Redis data

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Redis connection settings
REDIS_CONTAINER="langchain_multi_docs_rag-redis-1"
REDIS_HOST="localhost"
REDIS_PORT="6379"

# Function to detect Redis container
detect_redis_container() {
    # Try to find Redis container automatically
    local container=$(docker ps --format "table {{.Names}}" | grep -E "(redis|Redis)" | head -1)
    if [ -n "$container" ]; then
        REDIS_CONTAINER="$container"
    fi
}

# Auto-detect Redis container
detect_redis_container

# Use Docker to connect to Redis container
REDIS_CLI="docker exec -i $REDIS_CONTAINER redis-cli"

# Alternative: Connect via host port (if redis-cli is installed locally)
# REDIS_CLI="redis-cli -h $REDIS_HOST -p $REDIS_PORT"

# Function to check if Redis is running
check_redis_connection() {
    echo -e "${BLUE}Checking Redis connection...${NC}"
    if ! $REDIS_CLI ping > /dev/null 2>&1; then
        echo -e "${RED}Error: Cannot connect to Redis container${NC}"
        echo -e "${YELLOW}Make sure Redis is running with: docker-compose up redis -d${NC}"
        echo -e "${YELLOW}Container name: $REDIS_CONTAINER${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Redis connection successful${NC}"
}

# Function to display Redis info
show_redis_info() {
    echo -e "\n${CYAN}=== Redis Server Information ===${NC}"
    $REDIS_CLI info server | grep -E "(redis_version|process_id|uptime_in_seconds|tcp_port)"
    
    echo -e "\n${CYAN}=== Memory Usage ===${NC}"
    $REDIS_CLI info memory | grep -E "(used_memory_human|used_memory_peak_human|maxmemory_human)"
    
    echo -e "\n${CYAN}=== Keyspace Information ===${NC}"
    $REDIS_CLI info keyspace
    
    echo -e "\n${CYAN}=== Connected Clients ===${NC}"
    $REDIS_CLI info clients | grep -E "(connected_clients|client_recent_max_input_buffer|client_recent_max_output_buffer)"
}

# Function to show all keys with their types and sizes
show_keys_summary() {
    echo -e "\n${CYAN}=== Keys Summary ===${NC}"
    
    # Get total key count
    key_count=$($REDIS_CLI dbsize)
    echo -e "Total keys: ${GREEN}$key_count${NC}"
    
    if [ "$key_count" -eq 0 ]; then
        echo -e "${YELLOW}No keys found in Redis${NC}"
        return
    fi
    
    # Create a temporary file to store keys
    temp_file=$(mktemp)
    $REDIS_CLI keys "*" > "$temp_file"
    
    echo -e "\n${PURPLE}All Keys with Details:${NC}"
    # Process each key
    while IFS= read -r key; do
        if [ -n "$key" ]; then
            key_type=$($REDIS_CLI type "$key")
            case $key_type in
                "string")
                    size=$($REDIS_CLI strlen "$key")
                    echo -e "${BLUE}$key${NC} (string, ${size} chars)"
                    ;;
                "hash")
                    size=$($REDIS_CLI hlen "$key")
                    echo -e "${BLUE}$key${NC} (hash, ${size} fields)"
                    ;;
                "list")
                    size=$($REDIS_CLI llen "$key")
                    echo -e "${BLUE}$key${NC} (list, ${size} items)"
                    ;;
                "set")
                    size=$($REDIS_CLI scard "$key")
                    echo -e "${BLUE}$key${NC} (set, ${size} members)"
                    ;;
                "zset")
                    size=$($REDIS_CLI zcard "$key")
                    echo -e "${BLUE}$key${NC} (sorted set, ${size} members)"
                    ;;
                *)
                    echo -e "${BLUE}$key${NC} ($key_type)"
                    ;;
            esac
        fi
    done < "$temp_file"
    
    echo -e "\n${PURPLE}Key Types Summary:${NC}"
    # Show type distribution
    while IFS= read -r key; do
        if [ -n "$key" ]; then
            $REDIS_CLI type "$key"
        fi
    done < "$temp_file" | sort | uniq -c | sort -nr | while read count type; do
        echo -e "  ${GREEN}$count${NC} $type"
    done
    
    # Clean up
    rm -f "$temp_file"
}

# Function to show detailed key information
show_key_details() {
    local pattern="$1"
    if [ -z "$pattern" ]; then
        pattern="*"
    fi
    
    echo -e "\n${CYAN}=== Detailed Key Information (pattern: $pattern) ===${NC}"
    
    $REDIS_CLI --scan --pattern "$pattern" | while read key; do
        if [ -n "$key" ]; then
            key_type=$($REDIS_CLI type "$key")
            ttl=$($REDIS_CLI ttl "$key")
            
            # TTL display
            if [ "$ttl" -eq -1 ]; then
                ttl_display="no expiry"
            elif [ "$ttl" -eq -2 ]; then
                ttl_display="expired/not found"
            else
                ttl_display="${ttl}s"
            fi
            
            echo -e "\n${GREEN}Key:${NC} $key"
            echo -e "${GREEN}Type:${NC} $key_type"
            echo -e "${GREEN}TTL:${NC} $ttl_display"
            
            # Show content preview based on type
            case $key_type in
                "string")
                    content=$($REDIS_CLI get "$key" | head -c 100)
                    echo -e "${GREEN}Content preview:${NC} $content..."
                    ;;
                "hash")
                    echo -e "${GREEN}Hash fields:${NC}"
                    $REDIS_CLI hgetall "$key" | head -10
                    ;;
                "list")
                    echo -e "${GREEN}List items (first 5):${NC}"
                    $REDIS_CLI lrange "$key" 0 4
                    ;;
                "set")
                    echo -e "${GREEN}Set members (first 5):${NC}"
                    $REDIS_CLI smembers "$key" | head -5
                    ;;
                "zset")
                    echo -e "${GREEN}Sorted set (first 5):${NC}"
                    $REDIS_CLI zrange "$key" 0 4 withscores
                    ;;
            esac
        fi
    done
}

# Function to clean specific keys by pattern
clean_keys_by_pattern() {
    local pattern="$1"
    if [ -z "$pattern" ]; then
        echo -e "${RED}Error: No pattern specified${NC}"
        return 1
    fi
    
    echo -e "\n${YELLOW}Finding keys matching pattern: $pattern${NC}"
    
    # Count matching keys first
    matching_keys=$($REDIS_CLI --scan --pattern "$pattern")
    if [ -z "$matching_keys" ]; then
        echo -e "${YELLOW}No keys found matching pattern: $pattern${NC}"
        return 0
    fi
    
    key_count=$(echo "$matching_keys" | wc -l | tr -d ' ')
    
    echo -e "${YELLOW}Found $key_count keys matching pattern${NC}"
    echo -e "${PURPLE}Sample keys:${NC}"
    echo "$matching_keys" | head -10
    
    if [ "$key_count" -gt 10 ]; then
        echo "... and $(($key_count - 10)) more"
    fi
    
    echo -e "\n${RED}Are you sure you want to delete these $key_count keys? (y/N)${NC}"
    read -r confirmation
    
    if [[ $confirmation =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Deleting keys...${NC}"
        deleted_count=$(echo "$matching_keys" | xargs -I {} $REDIS_CLI del {} | wc -l)
        echo -e "${GREEN}✓ Deleted $deleted_count keys${NC}"
    else
        echo -e "${YELLOW}Operation cancelled${NC}"
    fi
}

# Function to clean all Redis data
clean_all_data() {
    echo -e "\n${RED}⚠️  WARNING: This will delete ALL data in Redis! ⚠️${NC}"
    echo -e "${RED}This action cannot be undone.${NC}"
    echo -e "\nType 'DELETE ALL' to confirm: "
    read -r confirmation
    
    if [ "$confirmation" = "DELETE ALL" ]; then
        echo -e "${YELLOW}Flushing all Redis data...${NC}"
        $REDIS_CLI flushall
        echo -e "${GREEN}✓ All Redis data has been deleted${NC}"
    else
        echo -e "${YELLOW}Operation cancelled${NC}"
    fi
}

# Function to show help
show_help() {
    echo -e "\n${CYAN}Redis Manager - Usage:${NC}"
    echo -e "  $0 info              - Show Redis server information"
    echo -e "  $0 keys              - Show keys summary"
    echo -e "  $0 details [pattern] - Show detailed key information (optional pattern)"
    echo -e "  $0 clean <pattern>   - Clean keys matching pattern"
    echo -e "  $0 clean-all         - Clean ALL Redis data (dangerous!)"
    echo -e "  $0 monitor           - Monitor Redis commands in real-time"
    echo -e "  $0 help              - Show this help message"
    echo -e "\n${PURPLE}Examples:${NC}"
    echo -e "  $0 details 'cache:*' - Show all cache keys"
    echo -e "  $0 clean 'temp:*'    - Delete all temporary keys"
    echo -e "  $0 details           - Show all keys details"
}

# Function to monitor Redis commands
monitor_redis() {
    echo -e "\n${CYAN}=== Monitoring Redis Commands (Press Ctrl+C to stop) ===${NC}"
    $REDIS_CLI monitor
}

# Main script logic
main() {
    case "${1:-help}" in
        "info")
            check_redis_connection
            show_redis_info
            ;;
        "keys")
            check_redis_connection
            show_keys_summary
            ;;
        "details")
            check_redis_connection
            show_key_details "$2"
            ;;
        "clean")
            if [ -z "$2" ]; then
                echo -e "${RED}Error: Pattern required for clean command${NC}"
                echo -e "Usage: $0 clean <pattern>"
                exit 1
            fi
            check_redis_connection
            clean_keys_by_pattern "$2"
            ;;
        "clean-all")
            check_redis_connection
            clean_all_data
            ;;
        "monitor")
            check_redis_connection
            monitor_redis
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"
