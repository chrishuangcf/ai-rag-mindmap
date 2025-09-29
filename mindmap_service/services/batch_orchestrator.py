"""
Batch Orchestrator - Handles background processing and job queue management
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import uuid
import logging

from core.models import BatchJob, MindMap
from core.database import get_redis_client
from core.config import settings
from services.graph_service import GraphService


logger = logging.getLogger(__name__)


class BatchOrchestrator:
    """Orchestrates batch processing of mind map operations"""
    
    def __init__(self):
        self.graph_service = GraphService()
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.job_handlers: Dict[str, Callable] = {
            "create_mindmap": self._handle_create_mindmap,
            "update_mindmap": self._handle_update_mindmap,
            "analyze_mindmap": self._handle_analyze_mindmap,
            "bulk_process": self._handle_bulk_process,
        }
        self.max_concurrent_jobs = settings.MAX_CONCURRENT_JOBS
        self.is_running = False
    
    async def start(self):
        """Start the batch orchestrator"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("🚀 Starting Batch Orchestrator")
        
        # Start the main processing loop
        asyncio.create_task(self._process_queue())
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_completed_jobs())
    
    async def stop(self):
        """Stop the batch orchestrator"""
        self.is_running = False
        
        # Cancel all running jobs
        for job_id, task in self.running_jobs.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled job {job_id}")
        
        self.running_jobs.clear()
        logger.info("🛑 Batch Orchestrator stopped")
    
    async def submit_job(
        self, 
        job_type: str, 
        parameters: Dict[str, Any], 
        priority: int = 1
    ) -> BatchJob:
        """Submit a new batch job"""
        
        if job_type not in self.job_handlers:
            raise ValueError(f"Unknown job type: {job_type}")
        
        job = BatchJob(
            type=job_type,
            priority=priority,
            parameters=parameters,
            status="pending"
        )
        
        # Store job in Redis
        await self._store_job(job)
        
        # Add to queue
        redis_client = await get_redis_client()
        await redis_client.zadd(
            settings.BATCH_QUEUE_NAME,
            {job.id: priority}
        )
        
        logger.info(f"Submitted job {job.id} of type {job_type}")
        return job
    
    async def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get the status of a specific job"""
        return await self._get_job(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a specific job"""
        # If job is running, cancel the task
        if job_id in self.running_jobs:
            task = self.running_jobs[job_id]
            if not task.done():
                task.cancel()
                await self._update_job_status(job_id, "cancelled")
                return True
        
        # If job is pending, remove from queue
        redis_client = await get_redis_client()
        removed = await redis_client.zrem(settings.BATCH_QUEUE_NAME, job_id)
        
        if removed:
            await self._update_job_status(job_id, "cancelled")
            return True
        
        return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get the current status of the job queue"""
        redis_client = await get_redis_client()
        
        # Get pending jobs count
        pending_count = await redis_client.zcard(settings.BATCH_QUEUE_NAME)
        
        # Get running jobs count
        running_count = len([task for task in self.running_jobs.values() if not task.done()])
        
        # Get recent job statistics
        job_keys = await redis_client.keys("batch_job:*")
        total_jobs = len(job_keys)
        
        completed_jobs = 0
        failed_jobs = 0
        
        for key in job_keys[-100:]:  # Check last 100 jobs
            job_data = await redis_client.get(key)
            if job_data:
                job = BatchJob.model_validate_json(job_data)
                if job.status == "completed":
                    completed_jobs += 1
                elif job.status == "failed":
                    failed_jobs += 1
        
        return {
            "is_running": self.is_running,
            "pending_jobs": pending_count,
            "running_jobs": running_count,
            "max_concurrent": self.max_concurrent_jobs,
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "success_rate": completed_jobs / max(completed_jobs + failed_jobs, 1) * 100
        }
    
    async def _process_queue(self):
        """Main queue processing loop"""
        redis_client = await get_redis_client()
        
        while self.is_running:
            try:
                # Check if we can start more jobs
                running_count = len([task for task in self.running_jobs.values() if not task.done()])
                
                if running_count >= self.max_concurrent_jobs:
                    await asyncio.sleep(1)
                    continue
                
                # Get next job from queue (highest priority first)
                job_data = await redis_client.zpopmax(settings.BATCH_QUEUE_NAME, 1)
                
                if not job_data:
                    await asyncio.sleep(1)
                    continue
                
                job_id, priority = job_data[0]
                job = await self._get_job(job_id)
                
                if not job:
                    logger.warning(f"Job {job_id} not found in storage")
                    continue
                
                # Start processing the job
                await self._start_job(job)
                
            except Exception as e:
                logger.error(f"Error in queue processing: {e}")
                await asyncio.sleep(5)
    
    async def _start_job(self, job: BatchJob):
        """Start processing a specific job"""
        logger.info(f"Starting job {job.id} of type {job.type}")
        
        # Update job status
        job.status = "running"
        job.started_at = datetime.utcnow()
        await self._store_job(job)
        
        # Get handler
        handler = self.job_handlers.get(job.type)
        if not handler:
            await self._fail_job(job.id, f"No handler for job type: {job.type}")
            return
        
        # Create and start task
        task = asyncio.create_task(self._run_job(handler, job))
        self.running_jobs[job.id] = task
        
        # Handle task completion
        task.add_done_callback(lambda t: self._on_job_complete(job.id, t))
    
    async def _run_job(self, handler: Callable, job: BatchJob):
        """Run a specific job handler"""
        try:
            result = await handler(job)
            
            # Update job with result
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.result = result
            job.progress = 100.0
            
            await self._store_job(job)
            logger.info(f"Job {job.id} completed successfully")
            
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.completed_at = datetime.utcnow()
            await self._store_job(job)
            logger.info(f"Job {job.id} was cancelled")
            
        except Exception as e:
            await self._fail_job(job.id, str(e))
            logger.error(f"Job {job.id} failed: {e}")
    
    def _on_job_complete(self, job_id: str, task: asyncio.Task):
        """Callback for when a job completes"""
        if job_id in self.running_jobs:
            del self.running_jobs[job_id]
    
    async def _fail_job(self, job_id: str, error_message: str):
        """Mark a job as failed"""
        job = await self._get_job(job_id)
        if job:
            job.status = "failed"
            job.error_message = error_message
            job.completed_at = datetime.utcnow()
            await self._store_job(job)
    
    async def _store_job(self, job: BatchJob):
        """Store job in Redis"""
        redis_client = await get_redis_client()
        key = f"batch_job:{job.id}"
        await redis_client.setex(
            key, 
            settings.BATCH_RESULT_TTL, 
            job.json()
        )
    
    async def _get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get job from Redis"""
        redis_client = await get_redis_client()
        key = f"batch_job:{job_id}"
        job_data = await redis_client.get(key)
        
        if job_data:
            return BatchJob.model_validate_json(job_data)
        return None
    
    async def _update_job_status(self, job_id: str, status: str):
        """Update job status"""
        job = await self._get_job(job_id)
        if job:
            job.status = status
            await self._store_job(job)
    
    async def _cleanup_completed_jobs(self):
        """Cleanup old completed jobs"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                redis_client = await get_redis_client()
                job_keys = await redis_client.keys("batch_job:*")
                
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                for key in job_keys:
                    job_data = await redis_client.get(key)
                    if job_data:
                        job = BatchJob.parse_raw(job_data)
                        if (job.completed_at and 
                            job.completed_at < cutoff_time and 
                            job.status in ["completed", "failed", "cancelled"]):
                            await redis_client.delete(key)
                            logger.info(f"Cleaned up old job {job.id}")
                
            except Exception as e:
                logger.error(f"Error in job cleanup: {e}")
    
    # Job Handlers
    
    async def _handle_create_mindmap(self, job: BatchJob) -> Dict[str, Any]:
        """Handle mind map creation"""
        params = job.parameters
        
        mindmap = await self.graph_service.create_mindmap(
            title=params["title"],
            cache_hashes=params["cache_hashes"],
            description=params.get("description"),
            max_nodes=params.get("max_nodes", 100),
            similarity_threshold=params.get("similarity_threshold", 0.3)
        )
        
        return {
            "mindmap_id": mindmap.id,
            "nodes_created": len(mindmap.nodes),
            "relationships_created": len(mindmap.relationships)
        }
    
    async def _handle_update_mindmap(self, job: BatchJob) -> Dict[str, Any]:
        """Handle mind map update"""
        params = job.parameters
        
        mindmap = await self.graph_service.update_mindmap(
            mindmap_id=params["mindmap_id"],
            add_cache_hashes=params.get("add_cache_hashes", []),
            remove_cache_hashes=params.get("remove_cache_hashes", [])
        )
        
        return {
            "mindmap_id": mindmap.id,
            "nodes_updated": len(mindmap.nodes),
            "relationships_updated": len(mindmap.relationships)
        }
    
    async def _handle_analyze_mindmap(self, job: BatchJob) -> Dict[str, Any]:
        """Handle mind map analysis"""
        params = job.parameters
        
        analysis = await self.graph_service.analyze_graph(params["mindmap_id"])
        stats = await self.graph_service.get_mindmap_stats(params["mindmap_id"])
        
        return {
            "analysis": analysis.dict(),
            "stats": stats.dict()
        }
    
    async def _handle_bulk_process(self, job: BatchJob) -> Dict[str, Any]:
        """Handle bulk processing of multiple mind maps"""
        params = job.parameters
        cache_hash_groups = params["cache_hash_groups"]
        
        results = []
        
        for i, group in enumerate(cache_hash_groups):
            # Update progress
            progress = (i / len(cache_hash_groups)) * 100
            job.progress = progress
            await self._store_job(job)
            
            # Create mind map for this group
            mindmap = await self.graph_service.create_mindmap(
                title=f"Bulk Mind Map {i+1}",
                cache_hashes=group["cache_hashes"],
                description=group.get("description"),
                max_nodes=group.get("max_nodes", 100),
                similarity_threshold=group.get("similarity_threshold", 0.3)
            )
            
            results.append({
                "mindmap_id": mindmap.id,
                "cache_hashes": group["cache_hashes"],
                "nodes_created": len(mindmap.nodes),
                "relationships_created": len(mindmap.relationships)
            })
        
        return {
            "total_mindmaps_created": len(results),
            "mindmaps": results
        }
