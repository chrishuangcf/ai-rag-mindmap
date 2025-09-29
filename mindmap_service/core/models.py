"""
Core models and data structures for the Mind Map Service
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timezone
from enum import Enum
import uuid


class NodeType(str, Enum):
    """Types of nodes in the mind map"""
    CONCEPT = "concept"
    DOCUMENT = "document"
    TOPIC = "topic"
    KEYWORD = "keyword"
    ENTITY = "entity"


class RelationshipType(str, Enum):
    """Types of relationships between nodes"""
    SIMILAR_TO = "SIMILAR_TO"
    RELATED_TO = "RELATED_TO"
    CONTAINS = "CONTAINS"
    DERIVED_FROM = "DERIVED_FROM"
    PART_OF = "PART_OF"


class LayoutType(str, Enum):
    """Layout algorithms for mind map visualization"""
    FORCE = "force"
    CIRCULAR = "circular"
    HIERARCHICAL = "hierarchical"
    RADIAL = "radial"
    GRID = "grid"


class MindMapNode(BaseModel):
    """Mind map node representation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: NodeType
    content: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    size: float = Field(default=20.0, ge=10.0, le=100.0)
    color: Optional[str] = None
    position: Optional[Dict[str, float]] = None  # {x: float, y: float}
    
    model_config = ConfigDict(use_enum_values=True)


class MindMapRelationship(BaseModel):
    """Mind map relationship representation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    type: RelationshipType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(use_enum_values=True)


class MindMap(BaseModel):
    """Complete mind map representation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: Optional[str] = None
    nodes: List[MindMapNode] = Field(default_factory=list)
    relationships: List[MindMapRelationship] = Field(default_factory=list)
    layout: LayoutType = LayoutType.FORCE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cache_hashes: List[str] = Field(default_factory=list)  # RAG cache hashes used
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(use_enum_values=True)


class BatchJob(BaseModel):
    """Batch processing job representation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # "create_mindmap", "update_mindmap", "analyze_documents"
    status: str = "pending"  # pending, running, completed, failed
    priority: int = Field(default=1, ge=1, le=10)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    
    model_config = ConfigDict(use_enum_values=True)


# Request/Response Models

class CreateMindMapRequest(BaseModel):
    """Request to create a new mind map"""
    title: str
    description: Optional[str] = None
    cache_hashes: List[str]  # RAG cache hashes to analyze
    layout: LayoutType = LayoutType.FORCE
    max_nodes: int = Field(default=100, ge=10, le=1000)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    include_documents: bool = True
    include_concepts: bool = True
    include_keywords: bool = True


class UpdateMindMapRequest(BaseModel):
    """Request to update an existing mind map"""
    title: Optional[str] = None
    description: Optional[str] = None
    layout: Optional[LayoutType] = None
    add_cache_hashes: List[str] = Field(default_factory=list)


class QueryMindMapRequest(BaseModel):
    """Request to query mind map data"""
    mindmap_id: Optional[str] = None
    cache_hashes: Optional[List[str]] = None
    node_types: List[NodeType] = Field(default_factory=list)
    similarity_threshold: Optional[float] = None
    limit: int = Field(default=100, ge=1, le=1000)


class BatchJobRequest(BaseModel):
    """Request to create a batch job"""
    type: str
    priority: int = Field(default=1, ge=1, le=10)
    parameters: Dict[str, Any]


class MindMapResponse(BaseModel):
    """Response containing mind map data"""
    mindmap: MindMap
    stats: Dict[str, Any] = Field(default_factory=dict)
    technical_concepts: List[str] = Field(default_factory=list)
    extraction_stats: Dict[str, Any] = Field(default_factory=dict)


class BatchJobResponse(BaseModel):
    """Response containing batch job information"""
    job: BatchJob
    message: str


class MindMapStats(BaseModel):
    """Mind map statistics"""
    total_nodes: int
    total_relationships: int
    node_types_count: Dict[str, int]
    relationship_types_count: Dict[str, int]
    avg_node_degree: float
    max_node_degree: int
    density: float  # Number of actual relationships / possible relationships
    connected_components: int


class GraphAnalysisResult(BaseModel):
    """Result of graph analysis"""
    central_concepts: List[Dict[str, Any]]  # Most connected nodes
    communities: List[List[str]]  # Node clusters
    key_paths: List[List[str]]  # Important connection paths
    outliers: List[str]  # Isolated or unique nodes
    similarity_patterns: Dict[str, Any]  # Patterns in similarity scores
