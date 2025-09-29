"""
Main FastAPI application for LangChain Multi-Document RAG API.
This is the simplified main file that imports and integrates all components.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio

# Import all API routers
from src.api.core import router as core_router
from src.api.cache import router as cache_router
from src.api.health import router as health_router
from src.api.debug import router as debug_router
from src.api.mindmap_integration import mindmap_integration_router
from src.services.cache import verify_and_recreate_redis_indices

# Initialize FastAPI app
app = FastAPI(
    title="LangChain Multi-Document RAG API",
    description="Multi-document RAG system with Redis caching and Google search fallback",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080", 
        "http://127.0.0.1:8080", 
        "http://localhost:3000", 
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers (must be before static file mounting)
app.include_router(core_router, prefix="/api", tags=["RAG Core"])
app.include_router(cache_router, prefix="/api", tags=["Cache Management"])
app.include_router(health_router, prefix="/api", tags=["Health Check"])
app.include_router(debug_router, prefix="/api", tags=["Debug & Test"])
app.include_router(mindmap_integration_router, prefix="/api", tags=["Mind Map Integration"])

# Serve static files (web UI) - this must come after API routes
try:
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from fastapi import Response
    
    # Custom static files handler with cache control
    app.mount("/static", StaticFiles(directory="web_ui"), name="static")
    
    # Serve the main HTML file at root
    @app.get("/")
    async def serve_index():
        response = FileResponse("web_ui/index.html")
        # Disable caching for development
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
        
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")
    
    @app.get("/")
    async def root():
        return {
            "message": "LangChain Multi-Document RAG API",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/api/health"
        }

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    print("🚀 Starting LangChain Multi-Document RAG API...")
    print("🔍 Verifying Redis cache indices...")
    await verify_and_recreate_redis_indices()
    print("✅ Startup completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
