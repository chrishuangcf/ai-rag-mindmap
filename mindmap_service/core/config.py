"""
Configuration settings for the Mind Map Service
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    SERVICE_NAME: str = "RAG Mind Map Service"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # API configuration
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8003
    
    # Neo4j configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "mindmapneo4j")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # Redis configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "1"))  # Use different DB than main app
    
    # RAG Service configuration
    RAG_SERVICE_URL: str = os.getenv("RAG_SERVICE_URL", "http://app:8000") + "/api"
    
    # Batch processing configuration
    BATCH_QUEUE_NAME: str = "mindmap_batch_queue"
    BATCH_RESULT_TTL: int = 3600  # 1 hour
    MAX_BATCH_SIZE: int = 100
    MAX_CONCURRENT_JOBS: int = 5
    
    # Mind map configuration
    MAX_NODES_PER_MAP: int = 1000
    MAX_RELATIONSHIPS_PER_NODE: int = 50
    MIN_SIMILARITY_THRESHOLD: float = 0.3
    
    # Graph layout configuration
    DEFAULT_LAYOUT: str = "force"  # force, circular, hierarchical
    NODE_SIZE_RANGE: tuple = (10, 50)
    EDGE_WIDTH_RANGE: tuple = (1, 5)
    
    model_config = {"env_file": ".env", "case_sensitive": True}


# Global settings instance
settings = Settings()
