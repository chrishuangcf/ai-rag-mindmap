#!/bin/bash

echo "🧪 Testing Redis Persistence Configuration"
echo "========================================="

# Check if Redis config file exists
if [ -f "redis.conf" ]; then
    echo "✅ Redis configuration file found"
    
    # Check persistence settings
    if grep -q "appendonly yes" redis.conf; then
        echo "✅ AOF (Append Only File) enabled"
    else
        echo "❌ AOF not enabled"
    fi
    
    if grep -q "save [0-9]" redis.conf; then
        echo "✅ RDB snapshots configured"
        echo "📋 RDB save intervals:"
        grep "^save " redis.conf | sed 's/^/   /'
    else
        echo "❌ RDB snapshots not configured"
    fi
else
    echo "❌ Redis configuration file not found"
fi

echo ""
echo "📂 Data directories:"
ls -la data/ 2>/dev/null || echo "❌ Data directories not found"

echo ""
echo "🐳 Docker Compose Redis configuration:"
if grep -A 10 "redis:" docker-compose.yaml | grep -q "redis.conf"; then
    echo "✅ Redis config file mounted in Docker"
else
    echo "❌ Redis config file not mounted in Docker"
fi

if grep -A 10 "redis:" docker-compose.yaml | grep -q "redis_data:/data"; then
    echo "✅ Redis data volume configured"
else
    echo "❌ Redis data volume not configured"
fi

echo ""
echo "🔧 To test persistence:"
echo "1. Start services: docker-compose up -d"
echo "2. Add some data to Redis"
echo "3. Stop services: docker-compose down"
echo "4. Start services again: docker-compose up -d"
echo "5. Check if data persists"
