#!/bin/bash

# Initialize data directories for persistent storage
echo "🔧 Initializing persistent data directories..."

# Create data directories if they don't exist
mkdir -p data/{redis,neo4j,neo4j_logs}

# Set proper permissions
echo "📁 Setting directory permissions..."
chmod 755 data/{redis,neo4j,neo4j_logs}

# Note about ownership for Docker containers
echo "� Directory ownership notes:"
echo "   - Redis container runs as UID 999:999"
echo "   - Neo4j container runs as UID 7474:7474"
echo "   - Current directories are created with your user ownership"
echo ""
echo "� If you encounter permission issues with Docker, run:"
echo "   sudo chown -R 999:999 data/redis"
echo "   sudo chown -R 7474:7474 data/neo4j data/neo4j_logs"
echo ""
echo "✅ Data directories initialized successfully!"
echo "📂 Redis data: ./data/redis"
echo "📂 Neo4j data: ./data/neo4j"
echo "📂 Neo4j logs: ./data/neo4j_logs"
