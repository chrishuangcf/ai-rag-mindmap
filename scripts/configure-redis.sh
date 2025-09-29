#!/bin/bash

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
until redis-cli -h redis -p 6379 ping; do
  echo "Redis is unavailable - sleeping"
  sleep 1
done

echo "Redis is ready - configuring memory optimization..."

# Apply Redis memory optimization commands
redis-cli -h redis -p 6379 CONFIG SET maxmemory-policy allkeys-lru
redis-cli -h redis -p 6379 CONFIG SET maxmemory 10gb
redis-cli -h redis -p 6379 CONFIG SET save ""
redis-cli -h redis -p 6379 CONFIG SET stop-writes-on-bgsave-error no

echo "Redis memory optimization applied successfully!"

# Keep the container running
exec "$@"
