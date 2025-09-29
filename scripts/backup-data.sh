#!/bin/bash

# Backup script for Redis and Neo4j persistent data
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "🔄 Creating backup in $BACKUP_DIR..."

# Backup Redis data
if [ -d "data/redis" ]; then
    echo "📊 Backing up Redis data..."
    cp -r data/redis "$BACKUP_DIR/redis"
    echo "✅ Redis backup completed"
fi

# Backup Neo4j data
if [ -d "data/neo4j" ]; then
    echo "🗄️  Backing up Neo4j data..."
    cp -r data/neo4j "$BACKUP_DIR/neo4j"
    echo "✅ Neo4j backup completed"
fi

# Create backup info file
cat > "$BACKUP_DIR/backup_info.txt" << EOF
Backup created: $(date)
Redis data size: $(du -sh data/redis 2>/dev/null | cut -f1 || echo "N/A")
Neo4j data size: $(du -sh data/neo4j 2>/dev/null | cut -f1 || echo "N/A")
Services running: $(docker-compose ps --services --filter status=running 2>/dev/null || echo "Docker Compose not available")
EOF

echo "📦 Backup completed successfully!"
echo "📂 Backup location: $BACKUP_DIR"
echo "💾 To restore from this backup:"
echo "   ./scripts/restore-backup.sh $BACKUP_DIR"
