#!/bin/bash

# Restore script for Redis and Neo4j persistent data
if [ -z "$1" ]; then
    echo "❌ Usage: $0 <backup_directory>"
    echo "📂 Available backups:"
    ls -la backups/ 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_DIR="$1"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "❌ Backup directory not found: $BACKUP_DIR"
    exit 1
fi

echo "⚠️  WARNING: This will replace current data with backup from $BACKUP_DIR"
echo "🛑 Make sure to stop services first: docker-compose down"
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Restore cancelled"
    exit 1
fi

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo "🛑 Services are still running. Stopping them..."
    docker-compose down
fi

# Restore Redis data
if [ -d "$BACKUP_DIR/redis" ]; then
    echo "📊 Restoring Redis data..."
    rm -rf data/redis/*
    cp -r "$BACKUP_DIR/redis/"* data/redis/
    chown -R 999:999 data/redis 2>/dev/null || echo "⚠️  Warning: Could not set Redis ownership"
    echo "✅ Redis data restored"
fi

# Restore Neo4j data
if [ -d "$BACKUP_DIR/neo4j" ]; then
    echo "🗄️  Restoring Neo4j data..."
    rm -rf data/neo4j/*
    cp -r "$BACKUP_DIR/neo4j/"* data/neo4j/
    chown -R 7474:7474 data/neo4j data/neo4j_logs 2>/dev/null || echo "⚠️  Warning: Could not set Neo4j ownership"
    echo "✅ Neo4j data restored"
fi

echo "🔄 Restore completed successfully!"
echo "🚀 Start services with: docker-compose up -d"

# Show backup info if available
if [ -f "$BACKUP_DIR/backup_info.txt" ]; then
    echo "📋 Backup information:"
    cat "$BACKUP_DIR/backup_info.txt"
fi
