"""
Database connections and initialization
"""

import redis.asyncio as aioredis
from neo4j import AsyncGraphDatabase
from typing import Optional
import asyncio

from .config import settings


class Neo4jConnection:
    """Neo4j database connection manager"""
    
    def __init__(self):
        self.driver = None
        self.session = None
    
    async def connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            await self.driver.verify_connectivity()
            print(f"✅ Connected to Neo4j at {settings.NEO4J_URI}")
            return self.driver
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            raise
    
    async def verify_connectivity(self):
        """Verify Neo4j connection"""
        if not self.driver:
            await self.connect()
        return await self.driver.verify_connectivity()
    
    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            print("🔌 Neo4j connection closed")
    
    async def get_session(self):
        """Get a new Neo4j session"""
        if not self.driver:
            await self.connect()
        return self.driver.session(database=settings.NEO4J_DATABASE)


class RedisConnection:
    """Redis database connection manager"""
    
    def __init__(self):
        self.client = None
    
    async def connect(self):
        """Establish connection to Redis"""
        try:
            self.client = aioredis.from_url(
                settings.REDIS_URL,
                db=settings.REDIS_DB,
                decode_responses=True
            )
            await self.client.ping()
            print(f"✅ Connected to Redis at {settings.REDIS_URL}")
            return self.client
        except Exception as e:
            print(f"❌ Failed to connect to Redis: {e}")
            raise
    
    async def ping(self):
        """Ping Redis to verify connection"""
        if not self.client:
            await self.connect()
        return await self.client.ping()
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            print("🔌 Redis connection closed")


# Global database instances
neo4j_connection = Neo4jConnection()
redis_connection = RedisConnection()

# Initialize connections
neo4j_driver = None
redis_client = None


async def init_databases():
    """Initialize database connections"""
    global neo4j_driver, redis_client
    
    neo4j_driver = await neo4j_connection.connect()
    redis_client = await redis_connection.connect()
    
    # Initialize Neo4j constraints and indexes
    await init_neo4j_schema()
    
    return neo4j_driver, redis_client


async def init_neo4j_schema():
    """Initialize Neo4j schema with constraints and indexes"""
    if not neo4j_driver:
        return
    
    schema_queries = [
        # Create constraints
        "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT document_hash IF NOT EXISTS FOR (d:Document) REQUIRE d.hash IS UNIQUE",
        "CREATE CONSTRAINT mindmap_id IF NOT EXISTS FOR (m:MindMap) REQUIRE m.id IS UNIQUE",
        
        # Create indexes for better performance
        "CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
        "CREATE INDEX concept_embedding IF NOT EXISTS FOR (c:Concept) ON (c.embedding)",
        "CREATE INDEX document_source IF NOT EXISTS FOR (d:Document) ON (d.source_url)",
        "CREATE INDEX relationship_weight IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.weight)",
        "CREATE INDEX similarity_score IF NOT EXISTS FOR ()-[r:SIMILAR_TO]-() ON (r.similarity)",
    ]
    
    async with neo4j_driver.session() as session:
        for query in schema_queries:
            try:
                await session.run(query)
                print(f"✅ Executed: {query[:50]}...")
            except Exception as e:
                print(f"⚠️  Schema query failed (might already exist): {e}")


# Convenience functions for getting database connections
async def get_neo4j_session():
    """Get Neo4j session"""
    return await neo4j_connection.get_session()


async def get_redis_client():
    """Get Redis client"""
    if not redis_client:
        await redis_connection.connect()
    return redis_client


# Initialize databases when module is imported
try:
    # Run initialization in the background
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If we're already in an event loop, schedule the initialization
        asyncio.create_task(init_databases())
    else:
        # If no event loop is running, run it synchronously
        asyncio.run(init_databases())
except Exception as e:
    print(f"⚠️  Could not initialize databases immediately: {e}")
    print("🔄 Databases will be initialized when needed")

# Export for easy importing
neo4j_driver = neo4j_connection
redis_client = redis_connection
