#!/bin/bash

# Simple Mind Map Cleanup using Neo4j directly
echo "🧹 Cleaning up all mindmaps..."

# Check if Neo4j container is running
if ! docker ps | grep -q neo4j; then
    echo "❌ Neo4j container is not running"
    exit 1
fi

# Get Neo4j container ID
NEO4J_CONTAINER=$(docker ps | grep neo4j | awk '{print $1}')

echo "📊 Current node count:"
docker exec -it $NEO4J_CONTAINER cypher-shell -u neo4j -p mindmapneo4j "MATCH (n) RETURN count(n) as total_nodes"

echo "🗑️  Deleting all nodes and relationships..."
docker exec -it $NEO4J_CONTAINER cypher-shell -u neo4j -p mindmapneo4j "MATCH (n) DETACH DELETE n"

echo "✅ All mindmaps have been deleted!"

echo "📊 Verification - remaining node count:"
docker exec -it $NEO4J_CONTAINER cypher-shell -u neo4j -p mindmapneo4j "MATCH (n) RETURN count(n) as total_nodes"
