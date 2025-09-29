"""
Mind Map Service - Main FastAPI Application

This service creates interactive mind maps from RAG vector data,
integrating with Neo4j graph database and batch processing orchestrator.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging
from contextlib import asynccontextmanager

from core.config import settings
from core.database import neo4j_driver, redis_client
from api.routes import mindmap_router, batch_router
from api.concept_extraction_router import router as concept_extraction_router
from services.batch_orchestrator import BatchOrchestrator
from services.graph_service import GraphService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("🚀 Starting Mind Map Service...")
    
    # Initialize services
    app.state.graph_service = GraphService()
    app.state.batch_orchestrator = BatchOrchestrator()
    
    # Verify database connections
    try:
        await neo4j_driver.verify_connectivity()
        print("✅ Neo4j connection verified")
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
    
    try:
        await redis_client.ping()
        print("✅ Redis connection verified")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
    
    # Start the batch orchestrator
    try:
        await app.state.batch_orchestrator.start()
        print("✅ Batch Orchestrator started")
    except Exception as e:
        print(f"❌ Batch Orchestrator startup failed: {e}")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Mind Map Service...")
    
    # Stop batch orchestrator
    try:
        await app.state.batch_orchestrator.stop()
        print("✅ Batch Orchestrator stopped")
    except Exception as e:
        print(f"❌ Error stopping Batch Orchestrator: {e}")
    
    await neo4j_driver.close()
    await redis_client.close()


# Create FastAPI app
app = FastAPI(
    title="RAG Mind Map Service",
    description="Interactive mind map visualization for RAG vector data",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(mindmap_router, prefix="/api/v1/mindmap", tags=["Mind Map"])
app.include_router(batch_router, prefix="/api/v1/batch", tags=["Batch Processing"])
app.include_router(concept_extraction_router, prefix="/api/v1/concepts", tags=["Concept Extraction"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "RAG Mind Map Service",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "mindmap": "/api/v1/mindmap",
            "batch": "/api/v1/batch",
            "concepts": "/api/v1/concepts"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with detailed processing status."""
    health_status = {
        "service": "healthy",
        "neo4j": "unknown",
        "redis": "unknown",
        "processing_status": "unknown",
        "queue_info": {},
        "service_info": {
            "name": "RAG Mind Map Service",
            "version": "1.0.0",
            "uptime": "N/A"
        }
    }

    # Check database connections
    try:
        await neo4j_driver.verify_connectivity()
        health_status["neo4j"] = "healthy"
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        health_status["neo4j"] = "unhealthy"
        health_status["service"] = "unhealthy"

    try:
        await redis_client.ping()
        health_status["redis"] = "healthy"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health_status["redis"] = "unhealthy"
        health_status["service"] = "unhealthy"

    # Check processing status from BatchOrchestrator with detailed info
    try:
        orchestrator = BatchOrchestrator()
        queue_status = await orchestrator.get_queue_status()
        
        health_status["queue_info"] = {
            "is_running": queue_status.get('is_running', False),
            "pending_jobs": queue_status.get('pending_jobs', 0),
            "running_jobs": queue_status.get('running_jobs', 0),
            "max_concurrent": queue_status.get('max_concurrent', 0),
            "total_jobs": queue_status.get('total_jobs', 0),
            "completed_jobs": queue_status.get('completed_jobs', 0),
            "failed_jobs": queue_status.get('failed_jobs', 0),
            "success_rate": round(queue_status.get('success_rate', 0), 1)
        }
        
        # Determine processing status
        if queue_status.get('running_jobs', 0) > 0:
            health_status["processing_status"] = "processing"
        elif queue_status.get('pending_jobs', 0) > 0:
            health_status["processing_status"] = "pending"
        elif queue_status.get('is_running', False):
            health_status["processing_status"] = "idle"
        else:
            health_status["processing_status"] = "stopped"
            
    except Exception as e:
        logger.error(f"Could not get batch queue status for health check: {e}")
        health_status["processing_status"] = "error"
        health_status["service"] = "degraded"  # Use degraded instead of unhealthy for processing issues

    return health_status


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=True if os.getenv("DEBUG", "false").lower() == "true" else False
    )
