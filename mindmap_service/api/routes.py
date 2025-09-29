"""
API routes for the Mind Map Service
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from typing import List, Optional, Dict, Any
import logging

from core.models import (
    MindMap, BatchJob, CreateMindMapRequest, UpdateMindMapRequest,
    QueryMindMapRequest, BatchJobRequest, MindMapResponse, BatchJobResponse,
    MindMapStats, GraphAnalysisResult
)
from services.graph_service import GraphService
from services.batch_orchestrator import BatchOrchestrator


logger = logging.getLogger(__name__)

# Create routers
mindmap_router = APIRouter()
batch_router = APIRouter()

# Dependency injection
def get_graph_service(request: Request) -> GraphService:
    """Get graph service instance from app state"""
    return getattr(request.app.state, 'graph_service', GraphService())

def get_batch_orchestrator(request: Request) -> BatchOrchestrator:
    """Get batch orchestrator instance from app state"""
    return getattr(request.app.state, 'batch_orchestrator', BatchOrchestrator())


# Mind Map Routes

@mindmap_router.post("/", response_model=MindMapResponse)
async def create_mindmap(
    request: CreateMindMapRequest,
    background_tasks: BackgroundTasks,
    req: Request,
    graph_service: GraphService = Depends(get_graph_service)
):
    """Create a new mind map from RAG data"""
    try:
        print(f"🔍 DEBUG: Creating mindmap with request: {request}")
        
        # For large requests, use batch processing
        if len(request.cache_hashes) > 5 or request.max_nodes > 200:
            print(f"🔄 DEBUG: Using batch processing for large request")
            # Submit as batch job
            batch_orchestrator = get_batch_orchestrator(req)
            job = await batch_orchestrator.submit_job(
                job_type="create_mindmap",
                parameters=request.dict(),
                priority=5
            )
            
            return MindMapResponse(
                mindmap=MindMap(
                    id=job.id,
                    title=request.title,
                    description=f"Mind map creation in progress (Job ID: {job.id})",
                    cache_hashes=request.cache_hashes
                ),
                stats={"status": "processing", "job_id": job.id}
            )
        
        print(f"🔄 DEBUG: Processing immediately for smaller request")
        # Process immediately for smaller requests
        mindmap = await graph_service.create_mindmap(
            title=request.title,
            cache_hashes=request.cache_hashes,
            description=request.description,
            max_nodes=request.max_nodes,
            similarity_threshold=request.similarity_threshold
        )
        
        print(f"✅ DEBUG: Mind map created successfully with ID: {mindmap.id}")
        
        # Get stats
        print(f"🔄 DEBUG: Getting mindmap stats")
        stats = await graph_service.get_mindmap_stats(mindmap.id)
        
        print(f"✅ DEBUG: Stats retrieved, creating response")
        return MindMapResponse(
            mindmap=mindmap,
            stats=stats.dict()
        )
        
    except Exception as e:
        print(f"❌ DEBUG: Exception details: {type(e).__name__}: {e}")
        import traceback
        print(f"❌ DEBUG: Full traceback: {traceback.format_exc()}")
        logger.error(f"Error creating mind map: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mindmap_router.get("/{mindmap_id}", response_model=MindMapResponse)
async def get_mindmap(
    mindmap_id: str,
    req: Request,
    graph_service: GraphService = Depends(get_graph_service)
):
    """Get a mind map by ID"""
    try:
        mindmap = await graph_service.get_mindmap(mindmap_id)
        if not mindmap:
            raise HTTPException(status_code=404, detail="Mind map not found")
        
        # Get stats
        stats = await graph_service.get_mindmap_stats(mindmap_id)
        
        return MindMapResponse(
            mindmap=mindmap,
            stats=stats.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving mind map {mindmap_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mindmap_router.put("/{mindmap_id}", response_model=MindMapResponse)
async def update_mindmap(
    mindmap_id: str,
    request: UpdateMindMapRequest,
    req: Request,
    graph_service: GraphService = Depends(get_graph_service)
):
    """Update an existing mind map"""
    try:
        mindmap = await graph_service.update_mindmap(
            mindmap_id=mindmap_id,
            add_cache_hashes=request.add_cache_hashes
        )
        
        # Get updated stats
        stats = await graph_service.get_mindmap_stats(mindmap_id)
        
        return MindMapResponse(
            mindmap=mindmap,
            stats=stats.dict()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating mind map {mindmap_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))





@mindmap_router.get("/{mindmap_id}/stats", response_model=MindMapStats)
async def get_mindmap_stats(
    mindmap_id: str,
    req: Request,
    graph_service: GraphService = Depends(get_graph_service)
):
    """Get statistics for a mind map"""
    try:
        stats = await graph_service.get_mindmap_stats(mindmap_id)
        return stats
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting stats for mind map {mindmap_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mindmap_router.get("/{mindmap_id}/analysis", response_model=GraphAnalysisResult)
async def analyze_mindmap(
    mindmap_id: str,
    req: Request,
    graph_service: GraphService = Depends(get_graph_service)
):
    """Perform graph analysis on a mind map"""
    try:
        analysis = await graph_service.analyze_graph(mindmap_id)
        return analysis
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing mind map {mindmap_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mindmap_router.get("/", response_model=List[MindMap])
async def list_mindmaps(
    req: Request,
    limit: int = 10,
    offset: int = 0,
    graph_service: GraphService = Depends(get_graph_service)
):
    """List all mind maps"""
    try:
        from core.database import get_neo4j_session
        
        async with await get_neo4j_session() as session:
            result = await session.run(
                """
                MATCH (m:MindMap)
                RETURN m
                ORDER BY m.created_at DESC
                SKIP $offset
                LIMIT $limit
                """,
                offset=offset,
                limit=limit
            )
            
            mindmaps = []
            async for record in result:
                mindmap_data = dict(record["m"])
                
                # Handle Neo4j datetime conversion
                if mindmap_data.get('created_at') and hasattr(mindmap_data['created_at'], 'to_native'):
                    mindmap_data['created_at'] = mindmap_data['created_at'].to_native()
                if mindmap_data.get('updated_at') and hasattr(mindmap_data['updated_at'], 'to_native'):
                    mindmap_data['updated_at'] = mindmap_data['updated_at'].to_native()
                
                # Note: This returns mindmaps without nodes/relationships for performance
                # Use get_mindmap endpoint to get full data
                mindmaps.append(MindMap(**mindmap_data, nodes=[], relationships=[]))
            
            return mindmaps
            
    except Exception as e:
        logger.error(f"Error listing mind maps: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch Processing Routes

@batch_router.post("/", response_model=BatchJobResponse)
async def submit_batch_job(
    request: BatchJobRequest,
    req: Request,
    batch_orchestrator: BatchOrchestrator = Depends(get_batch_orchestrator)
):
    """Submit a new batch job"""
    try:
        job = await batch_orchestrator.submit_job(
            job_type=request.type,
            parameters=request.parameters,
            priority=request.priority
        )
        
        return BatchJobResponse(
            job=job,
            message=f"Batch job {job.id} submitted successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting batch job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@batch_router.get("/{job_id}", response_model=BatchJob)
async def get_batch_job(
    job_id: str,
    req: Request,
    batch_orchestrator: BatchOrchestrator = Depends(get_batch_orchestrator)
):
    """Get batch job status"""
    try:
        job = await batch_orchestrator.get_job_status(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return job
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@batch_router.delete("/{job_id}")
async def cancel_batch_job(
    job_id: str,
    req: Request,
    batch_orchestrator: BatchOrchestrator = Depends(get_batch_orchestrator)
):
    """Cancel a batch job"""
    try:
        success = await batch_orchestrator.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
        
        return {"message": f"Job {job_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling batch job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@batch_router.get("/")
async def get_queue_status(
    req: Request,
    batch_orchestrator: BatchOrchestrator = Depends(get_batch_orchestrator)
):
    """Get batch queue status"""
    try:
        status = await batch_orchestrator.get_queue_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))








# Utility Routes

@mindmap_router.get("/health")
async def mindmap_service_health(req: Request):
    """Health check endpoint for mindmap service"""
    try:
        from core.database import get_neo4j_session, get_redis_client
        
        health_status = {
            "service": "healthy",
            "neo4j": "unknown",
            "redis": "unknown",
            "processing_status": "unknown",
            "endpoints": "available"
        }

        # Check Neo4j connection
        try:
            async with await get_neo4j_session() as session:
                await session.run("RETURN 1")
            health_status["neo4j"] = "healthy"
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            health_status["neo4j"] = "unhealthy"
            health_status["service"] = "unhealthy"

        # Check Redis connection
        try:
            redis_client = await get_redis_client()
            await redis_client.ping()
            health_status["redis"] = "healthy"
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            health_status["redis"] = "unhealthy"
            health_status["service"] = "unhealthy"

        # Check processing status
        try:
            batch_orchestrator = get_batch_orchestrator(req)
            queue_status = await batch_orchestrator.get_queue_status()
            
            if queue_status.get('running_jobs', 0) > 0:
                health_status["processing_status"] = "processing"
            elif queue_status.get('pending_jobs', 0) > 0:
                health_status["processing_status"] = "pending"
            else:
                health_status["processing_status"] = "idle"
                
        except Exception as e:
            logger.error(f"Processing status check failed: {e}")
            health_status["processing_status"] = "error"

        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Vector Mindmap Endpoint

@mindmap_router.post("/concepts/create_vector_mindmap", response_model=MindMapResponse)
async def create_vector_mindmap(
    request: CreateMindMapRequest,
    background_tasks: BackgroundTasks,
    req: Request,
    graph_service: GraphService = Depends(get_graph_service)
):
    """Create a vector-based concept mindmap from RAG data"""
    try:
        print(f"🔍 DEBUG: Creating vector mindmap with request: {request}")
        
        # Use the standard create_mindmap method
        mindmap = await graph_service.create_mindmap(
            title=request.title,
            cache_hashes=request.cache_hashes,
            description=request.description,
            max_nodes=request.max_nodes,
            similarity_threshold=request.similarity_threshold
        )
        
        print(f"✅ DEBUG: Vector mind map created successfully with ID: {mindmap.id}")
        
        # Extract technical concepts from the created nodes
        technical_concepts = []
        technical_count = 0
        total_concepts = len(mindmap.nodes)
        
        for node in mindmap.nodes:
            if node.metadata and node.metadata.get("is_technical", False):
                technical_concepts.append(node.name)
                technical_count += 1
        
        print(f"🔍 DEBUG: Found {technical_count} technical concepts out of {total_concepts} total")
        
        # Get stats
        stats = await graph_service.get_mindmap_stats(mindmap.id)
        
        # Prepare extraction stats
        extraction_stats = {
            "total_concepts": total_concepts,
            "technical_concepts": technical_count,
            "avg_relevance": sum(node.metadata.get("relevance_score", 0.5) for node in mindmap.nodes if node.metadata) / max(total_concepts, 1),
            "extraction_method": "vector_analysis",
            "technical_focus": technical_count > 0
        }
        
        return MindMapResponse(
            mindmap=mindmap,
            stats=stats.dict() if hasattr(stats, 'dict') else stats,
            technical_concepts=technical_concepts,
            extraction_stats=extraction_stats
        )
    except Exception as e:
        print(f"❌ DEBUG: Vector mindmap creation failed: {e}")
        import traceback
        print(f"❌ DEBUG: Full traceback: {traceback.format_exc()}")
        logger.error(f"Vector mindmap creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating vector mind map: {str(e)}")

# Utility Routes

@mindmap_router.post("/query")
async def query_mindmap_data(
    request: QueryMindMapRequest,
    req: Request,
    graph_service: GraphService = Depends(get_graph_service)
):
    """Query mind map data with flexible filters"""
    try:
        from core.database import get_neo4j_session
        
        # Build Cypher query based on request
        where_clauses = []
        params = {}
        
        if request.mindmap_id:
            where_clauses.append("m.id = $mindmap_id")
            params["mindmap_id"] = request.mindmap_id
        
        if request.cache_hashes:
            where_clauses.append("ANY(hash IN m.cache_hashes WHERE hash IN $cache_hashes)")
            params["cache_hashes"] = request.cache_hashes
        
        if request.node_types:
            node_type_filters = " OR ".join([f"n.type = '{nt.value}'" for nt in request.node_types])
            where_clauses.append(f"({node_type_filters})")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "TRUE"
        
        query = f"""
        MATCH (m:MindMap)-[:CONTAINS]->(n)
        WHERE {where_clause}
        RETURN n
        LIMIT $limit
        """
        
        params["limit"] = request.limit
        
        async with await get_neo4j_session() as session:
            result = await session.run(query, **params)
            
            nodes = []
            async for record in result:
                node_data = dict(record["n"])
                nodes.append(node_data)
            
            return {
                "nodes": nodes,
                "total_found": len(nodes),
                "query_params": request.dict()
            }
            
    except Exception as e:
        logger.error(f"Error querying mind map data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
