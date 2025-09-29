# Redis Manager Script

A comprehensive shell script for managing Redis data in your Docker-based development environment.

## Prerequisites

- Docker and Docker Compose installed
- Redis running in Docker container

## Setup

1. Make sure the script is executable:
   ```bash
   chmod +x test/redis-manager.sh
   ```

2. Start Redis with Docker Compose:
   ```bash
   docker-compose up redis -d
   ```

3. The script automatically detects your Redis container and connects via Docker exec.

## How It Works

This script connects to Redis running in a Docker container using `docker exec` commands. It automatically detects Redis containers and doesn't require `redis-cli` to be installed on your host machine.

## Usage

### Show Redis Information
```bash
./test/redis-manager.sh info
```
Displays server information, memory usage, keyspace info, and connected clients.

### Show Keys Summary
```bash
./test/redis-manager.sh keys
```
Shows total key count, key type distribution, and sample keys with their types and sizes.

### Show Detailed Key Information
```bash
# Show details for all keys
./test/redis-manager.sh details

# Show details for specific pattern
./test/redis-manager.sh details 'cache:*'
./test/redis-manager.sh details 'session:*'
./test/redis-manager.sh details 'user:*'
```
Displays detailed information about keys including type, TTL, and content preview.

### Clean Keys by Pattern
```bash
# Clean temporary keys
./test/redis-manager.sh clean 'temp:*'

# Clean cache keys
./test/redis-manager.sh clean 'cache:*'

# Clean session keys
./test/redis-manager.sh clean 'session:*'
```
Safely deletes keys matching the specified pattern after confirmation.

### Clean All Data (Dangerous!)
```bash
./test/redis-manager.sh clean-all
```
⚠️ **WARNING**: This deletes ALL data in Redis! You must type 'DELETE ALL' to confirm.

### Monitor Redis Commands
```bash
./test/redis-manager.sh monitor
```
Shows Redis commands in real-time (press Ctrl+C to stop).

### Help
```bash
./test/redis-manager.sh help
```
Shows usage information and examples.

## Features

- **Safe Operations**: Confirmation prompts for destructive operations
- **Pattern Matching**: Use Redis glob patterns (*, ?, [abc], etc.)
- **Colorized Output**: Easy to read colored terminal output
- **Connection Check**: Automatically verifies Redis connectivity
- **Key Type Support**: Handles strings, hashes, lists, sets, and sorted sets
- **Memory Information**: Shows Redis memory usage and configuration
- **Real-time Monitoring**: Monitor Redis commands as they happen

## Common Patterns

- `*` - All keys
- `user:*` - All user-related keys
- `cache:*` - All cache keys
- `session:*` - All session keys
- `temp:*` - All temporary keys
- `*:expired` - All keys ending with ":expired"

## Examples

```bash
# Check Redis status and memory usage
./test/redis-manager.sh info

# See what keys exist
./test/redis-manager.sh keys

# Look at cache-related keys
./test/redis-manager.sh details 'cache:*'

# Clean up old session data
./test/redis-manager.sh clean 'session:old:*'

# Monitor what's happening in Redis
./test/redis-manager.sh monitor

# Nuclear option - clean everything (be careful!)
./test/redis-manager.sh clean-all
```

## Configuration

The script automatically detects Redis containers by looking for containers with "redis" in their name. If you need to specify a different container name, you can edit these variables at the top of the script:

```bash
REDIS_CONTAINER="your-redis-container-name"
```

## Troubleshooting

- **Connection Error**: 
  - Make sure Redis is running with `docker-compose up redis -d`
  - Check if Redis container is running: `docker ps | grep redis`
  - Verify container name matches the detected name in error message

- **Permission Denied**: 
  - Make sure the script is executable with `chmod +x test/redis-manager.sh`
  - Ensure Docker daemon is running and accessible

- **Container Not Found**: 
  - The script auto-detects Redis containers
  - If detection fails, manually set `REDIS_CONTAINER` variable in the script
  - Container name format is typically `<project>-redis-1`

- **Command Not Found**: 
  - Make sure you're running the script from the project root directory
  - Ensure Docker is installed and accessible from command line
