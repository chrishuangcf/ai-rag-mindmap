"""
Unified Mind Map Service - Auto-creates and updates a single mind map from all RAG data
"""

import asyncio
import requests
from typing import List, Optional
from src.core.config import redis_client
from src.services.cache import get_all_cache_hashes

class UnifiedMindMapService:
    """Service to manage a single unified mind map for all RAG data"""
    
    UNIFIED_MINDMAP_ID = "unified-mindmap"
    MINDMAP_SERVICE_URL = "http://mindmap-service:8003"
    
    def __init__(self):
        self.mindmap_exists = False
        self._is_updating = False  # Lock to prevent concurrent updates
        self._check_mindmap_exists()
    
    def _check_mindmap_exists(self):
        """Check if the unified mind map already exists"""
        try:
            response = requests.get(f"{self.MINDMAP_SERVICE_URL}/api/v1/mindmap/{self.UNIFIED_MINDMAP_ID}")
            self.mindmap_exists = response.status_code == 200
        except Exception as e:
            print(f"Error checking mind map existence: {e}")
            self.mindmap_exists = False
    
    async def trigger_mindmap_update(self):
        """Trigger update/creation of the unified mind map with all cache data"""
        if self._is_updating:
            print("Mind map update already in progress. Skipping.")
            return

        try:
            self._is_updating = True
            # Get all current cache hashes
            cache_hashes = list(get_all_cache_hashes())
            
            if not cache_hashes:
                print("No cache data available for mind map")
                return
            
            print(f"Updating unified mind map with {len(cache_hashes)} cache entries")
            
            # Always recreate the unified mind map to ensure it includes all current data
            await self._create_mindmap(cache_hashes)
                
        except Exception as e:
            print(f"Error triggering mind map update: {e}")
        finally:
            self._is_updating = False
    
    async def _create_mindmap(self, cache_hashes: List[str]):
        """Create a new unified mind map and poll for completion"""
        import httpx
        try:
            payload = {
                "title": "Unified Knowledge Map",
                "description": "Consolidated view of all RAG documents and their relationships",
                "cache_hashes": cache_hashes,
                "max_nodes": 200,
                "similarity_threshold": 0.25,
                "layout": "force"
            }
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(f"{self.MINDMAP_SERVICE_URL}/api/v1/mindmap/", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                job_id = result.get("stats", {}).get("job_id")

                if not job_id:
                    # If no job_id, it was processed synchronously
                    self.UNIFIED_MINDMAP_ID = result['mindmap']['id']
                    self.mindmap_exists = True
                    print(f"Successfully created mind map synchronously with ID: {self.UNIFIED_MINDMAP_ID}")
                    return

                # Poll for batch job completion
                print(f"Mind map creation submitted as batch job: {job_id}. Polling for completion...")
                for _ in range(60):  # Poll for up to 5 minutes (60 * 5s)
                    await asyncio.sleep(5)
                    try:
                        async with httpx.AsyncClient(timeout=10) as client:
                            job_status_res = await client.get(f"{self.MINDMAP_SERVICE_URL}/api/v1/batch/{job_id}")
                        
                        if job_status_res.status_code == 200:
                            job_data = job_status_res.json()
                            if job_data['status'] == 'completed':
                                mindmap_id = job_data.get('result', {}).get('mindmap_id')
                                self.UNIFIED_MINDMAP_ID = mindmap_id
                                self.mindmap_exists = True
                                print(f"Batch job {job_id} completed. Mind map created with ID: {mindmap_id}")
                                return
                            elif job_data['status'] == 'failed':
                                print(f"Error: Batch job {job_id} failed. Reason: {job_data.get('result')}")
                                return
                        else:
                            print(f"Warning: Could not get job status for {job_id} (HTTP {job_status_res.status_code})")
                    except Exception as poll_e:
                        print(f"Warning: Error polling job status: {poll_e}")
                
                print(f"Error: Timed out waiting for batch job {job_id} to complete.")

            else:
                print(f"Failed to create mind map: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error creating unified mind map: {e}")
    
    async def _update_mindmap(self, cache_hashes: List[str]):
        """Update existing unified mind map with new cache hashes"""
        try:
            # For simplicity, we'll recreate the mind map with all current data
            # This ensures we capture any new relationships between old and new data
            payload = {
                "title": "Unified Knowledge Map", 
                "description": "Consolidated view of all RAG documents and their relationships",
                "cache_hashes": cache_hashes,
                "max_nodes": 200,
                "similarity_threshold": 0.25,
                "layout": "force"
            }
            
            # Delete old and create new (simpler than complex updates)
            try:
                requests.delete(f"{self.MINDMAP_SERVICE_URL}/api/v1/mindmap/{self.UNIFIED_MINDMAP_ID}")
            except:
                pass  # Ignore deletion errors
            
            # Create fresh unified mind map
            await self._create_mindmap(cache_hashes)
                
        except Exception as e:
            print(f"Error updating unified mind map: {e}")
    
    def get_mindmap_data(self) -> Optional[dict]:
        """Get the current unified mind map data"""
        try:
            # Get the most recent unified mind map
            response = requests.get(f"{self.MINDMAP_SERVICE_URL}/api/v1/mindmap/?limit=1")
            if response.status_code == 200:
                mindmaps = response.json()
                if mindmaps and len(mindmaps) > 0:
                    # Get the first (most recent) mind map
                    latest_mindmap = mindmaps[0]
                    self.UNIFIED_MINDMAP_ID = latest_mindmap['id']
                    
                    # Now get the full mind map data with nodes and relationships
                    full_response = requests.get(f"{self.MINDMAP_SERVICE_URL}/api/v1/mindmap/{self.UNIFIED_MINDMAP_ID}")
                    if full_response.status_code == 200:
                        return full_response.json()
                    else:
                        print(f"Failed to get full mind map data: {full_response.status_code}")
                        return None
                else:
                    print("No mind maps found")
                    return None
            else:
                print(f"Failed to list mind maps: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error getting mind map data: {e}")
            return None

# Global instance
unified_mindmap_service = UnifiedMindMapService()
