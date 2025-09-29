from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

from services.concept_extraction import extract_concepts_from_text, Concept
from services.graph_service import GraphService
from core.models import MindMapNode, NodeType, MindMap, MindMapRelationship, RelationshipType
from core.config import settings

router = APIRouter()

class ExtractionRequest(BaseModel):
    text: str

class VectorMindMapRequest(BaseModel):
    """Request to create mind map from vector data with technical keywords"""
    cache_hashes: List[str] = Field(description="List of cache hashes to analyze")
    title: Optional[str] = Field(default=None, description="Custom title for the mind map")
    max_concepts: int = Field(default=20, ge=5, le=100, description="Maximum number of concept nodes")
    similarity_threshold: float = Field(default=0.4, ge=0.1, le=0.9, description="Similarity threshold for relationships")
    focus_on_technical: bool = Field(default=True, description="Focus on technical keywords and concepts")
    extract_from_embeddings: bool = Field(default=True, description="Use vector embeddings for concept extraction")

class TechnicalConcept(BaseModel):
    """Enhanced concept with technical relevance scoring"""
    text: str
    label: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    frequency: int = Field(ge=1)
    embedding: Optional[List[float]] = None
    technical_category: Optional[str] = None

class VectorMindMapResponse(BaseModel):
    """Response containing mind map with technical concepts"""
    mindmap: MindMap
    technical_concepts: List[TechnicalConcept]
    extraction_stats: Dict[str, Any]

class ExtractionResponse(BaseModel):
    concepts: List[Concept]

class TechnicalExtractionResponse(BaseModel):
    """Enhanced response with technical concept analysis"""
    concepts: List[TechnicalConcept]
    extraction_method: str
    vector_similarity_used: bool
    technical_keywords_count: int

def get_graph_service():
    """Dependency to get graph service instance"""
    return GraphService()

@router.post("/extract", response_model=ExtractionResponse)
async def extract_concepts(request: ExtractionRequest):
    """
    Extracts concepts from the provided text.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        concepts = extract_concepts_from_text(request.text)
        return {"concepts": concepts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract_technical", response_model=TechnicalExtractionResponse)
async def extract_technical_concepts(request: ExtractionRequest):
    """
    Extract technical concepts using advanced NLP and vector analysis
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        technical_concepts = await _extract_technical_concepts_from_text(request.text)
        
        return TechnicalExtractionResponse(
            concepts=technical_concepts,
            extraction_method="vector_enhanced_nlp",
            vector_similarity_used=True,
            technical_keywords_count=len([c for c in technical_concepts if c.technical_category])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create_vector_mindmap", response_model=VectorMindMapResponse)
async def create_vector_mindmap(
    request: VectorMindMapRequest,
    graph_service: GraphService = Depends(get_graph_service)
):
    """
    Create a mind map from vector data with technical keywords as titles
    """
    if not request.cache_hashes:
        raise HTTPException(status_code=400, detail="At least one cache hash is required.")

    try:
        # Fetch vector data from RAG system
        vector_data = await _fetch_vector_data(request.cache_hashes)
        
        # Extract technical concepts using vector analysis
        technical_concepts = await _extract_concepts_from_vectors(
            vector_data, 
            max_concepts=request.max_concepts,
            focus_technical=request.focus_on_technical
        )
        
        # Create mind map nodes from technical concepts
        nodes = await _create_technical_nodes(technical_concepts)
        
        # Calculate relationships based on vector similarity
        relationships = await _calculate_vector_relationships(
            nodes, 
            technical_concepts,
            threshold=request.similarity_threshold
        )
        
        # Generate title if not provided
        title = request.title or await _generate_technical_title(technical_concepts)
        
        # Create mind map
        mindmap = MindMap(
            title=title,
            description=f"Technical concept map from {len(request.cache_hashes)} cache(s)",
            nodes=nodes,
            relationships=relationships,
            cache_hashes=request.cache_hashes,
            metadata={
                "extraction_method": "vector_analysis",
                "technical_focus": request.focus_on_technical,
                "concept_count": len(technical_concepts)
            }
        )
        
        # Store in Neo4j
        await graph_service._store_mindmap_in_neo4j(mindmap)
        
        return VectorMindMapResponse(
            mindmap=mindmap,
            technical_concepts=technical_concepts,
            extraction_stats={
                "total_concepts": len(technical_concepts),
                "technical_concepts": len([c for c in technical_concepts if c.technical_category]),
                "avg_relevance": np.mean([c.relevance_score for c in technical_concepts]),
                "vector_dimensions": len(technical_concepts[0].embedding) if technical_concepts and technical_concepts[0].embedding else 0
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating vector mind map: {str(e)}")

async def _fetch_vector_data(cache_hashes: List[str]) -> Dict[str, Any]:
    """Fetch vector embeddings and documents from RAG system"""
    vector_data = {
        "documents": [],
        "embeddings": [],
        "metadata": []
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        for cache_hash in cache_hashes:
            try:
                # Fetch documents
                docs_response = await client.get(
                    f"{settings.RAG_SERVICE_URL}/cache/{cache_hash}/documents"
                )
                docs_response.raise_for_status()
                docs_data = docs_response.json()
                
                # Fetch embeddings
                embeddings_response = await client.get(
                    f"{settings.RAG_SERVICE_URL}/cache/{cache_hash}/embeddings"
                )
                embeddings_response.raise_for_status()
                embeddings_data = embeddings_response.json()
                
                # Combine data
                documents = docs_data.get("documents", [])
                embeddings = embeddings_data.get("embeddings", [])
                
                for i, doc in enumerate(documents):
                    vector_data["documents"].append(doc)
                    vector_data["embeddings"].append(embeddings[i] if i < len(embeddings) else [])
                    vector_data["metadata"].append(doc.get("metadata", {}))
                    
            except Exception as e:
                raise ValueError(f"Failed to fetch vector data for cache {cache_hash}: {str(e)}")
    
    return vector_data

async def _extract_technical_concepts_from_text(text: str) -> List[TechnicalConcept]:
    """Extract technical concepts from text using enhanced NLP"""
    
    # Technical keyword patterns
    technical_patterns = {
        'programming': [
            r'\b(?:API|SDK|HTTP|REST|JSON|XML|SQL|NoSQL|AI|ML|GPU|CPU|RAM|SSD)\b',
            r'\b(?:Python|JavaScript|Java|C\+\+|React|Vue|Django|Flask|FastAPI|Node\.js)\b',
            r'\b(?:Docker|Kubernetes|AWS|Azure|GCP|GitHub|GitLab|CI/CD)\b',
            r'\b(?:machine learning|deep learning|neural network|artificial intelligence)\b',
            r'\b(?:database|repository|framework|library|package|module|component)\b'
        ],
        'data_science': [
            r'\b(?:tensor|matrix|vector|embedding|feature|model|algorithm|training)\b',
            r'\b(?:regression|classification|clustering|optimization|gradient|loss)\b',
            r'\b(?:dataset|dataframe|preprocessing|normalization|visualization)\b'
        ],
        'infrastructure': [
            r'\b(?:server|client|microservice|container|cluster|load balancer)\b',
            r'\b(?:authentication|authorization|encryption|security|firewall)\b',
            r'\b(?:monitoring|logging|metrics|alerting|observability)\b'
        ]
    }
    
    concepts = []
    concept_freq = Counter()
    
    # Extract using regex patterns
    for category, patterns in technical_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                normalized = match.strip().title()
                concept_freq[normalized] += 1
                
    # Extract additional concepts using NLTK
    basic_concepts = extract_concepts_from_text(text)
    for concept in basic_concepts:
        if len(concept.text) > 3:  # Filter short concepts
            concept_freq[concept.text] += 1
    
    # Create technical concepts with relevance scoring
    for concept_text, frequency in concept_freq.items():
        # Calculate relevance based on frequency and technical indicators
        relevance = min(1.0, frequency / 10.0)  # Normalize frequency
        
        # Boost technical terms
        is_technical = any(
            re.search(pattern, concept_text, re.IGNORECASE)
            for patterns in technical_patterns.values()
            for pattern in patterns
        )
        
        if is_technical:
            relevance = min(1.0, relevance * 1.5)
        
        # Determine technical category
        tech_category = None
        for category, patterns in technical_patterns.items():
            if any(re.search(pattern, concept_text, re.IGNORECASE) for pattern in patterns):
                tech_category = category
                break
        
        concepts.append(TechnicalConcept(
            text=concept_text,
            label="technical_concept" if is_technical else "general_concept",
            relevance_score=relevance,
            frequency=frequency,
            technical_category=tech_category
        ))
    
    # Sort by relevance and return top concepts
    return sorted(concepts, key=lambda x: x.relevance_score, reverse=True)[:50]

async def _extract_concepts_from_vectors(
    vector_data: Dict[str, Any], 
    max_concepts: int = 20,
    focus_technical: bool = True
) -> List[TechnicalConcept]:
    """Extract concepts using vector similarity analysis"""
    
    documents = vector_data.get("documents", [])
    embeddings = vector_data.get("embeddings", [])
    
    if not documents:
        return []
    
    # Combine all document text
    all_text = " ".join([doc.get("content", "") for doc in documents])
    
    # Extract technical concepts from combined text
    technical_concepts = await _extract_technical_concepts_from_text(all_text)
    
    # If we have embeddings, use them to enhance concept extraction
    if embeddings and any(emb for emb in embeddings if emb):
        # Convert embeddings to numpy array - LIMIT SIZE TO PREVENT MEMORY ISSUES
        valid_embeddings = [emb for emb in embeddings if emb and len(emb) > 0][:1000]  # Limit to 1000 embeddings
        if valid_embeddings:
            embeddings_matrix = np.array(valid_embeddings)
            
            # Calculate concept embeddings using TF-IDF on concept text
            concept_texts = [concept.text for concept in technical_concepts[:max_concepts]]  # Limit concepts
            if concept_texts:
                # Reduce vector dimensions to save memory
                max_features = min(len(concept_texts), 500, embeddings_matrix.shape[1])
                vectorizer = TfidfVectorizer(max_features=max_features)
                concept_vectors = vectorizer.fit_transform(concept_texts).toarray()
                
                # Enhance relevance scores using vector similarity
                for i, concept in enumerate(technical_concepts[:len(concept_vectors)]):
                    concept.embedding = concept_vectors[i].tolist()
                    
                    # Calculate similarity with document embeddings - USE BATCH PROCESSING
                    if len(concept_vectors[i]) <= embeddings_matrix.shape[1]:
                        # Pad or truncate to match dimensions
                        concept_vector = concept_vectors[i]
                        if len(concept_vector) < embeddings_matrix.shape[1]:
                            concept_vector = np.pad(concept_vector, (0, embeddings_matrix.shape[1] - len(concept_vector)))
                        else:
                            concept_vector = concept_vector[:embeddings_matrix.shape[1]]
                        
                        # Process in smaller batches to avoid memory issues
                        batch_size = 100
                        similarities = []
                        for batch_start in range(0, len(embeddings_matrix), batch_size):
                            batch_end = min(batch_start + batch_size, len(embeddings_matrix))
                            batch_similarities = cosine_similarity([concept_vector], embeddings_matrix[batch_start:batch_end])[0]
                            similarities.extend(batch_similarities)
                        
                        avg_similarity = np.mean(similarities)
                        
                        # Boost relevance based on vector similarity
                        concept.relevance_score = min(1.0, concept.relevance_score + avg_similarity * 0.3)
    
    # Filter and sort concepts
    if focus_technical:
        # Prioritize technical concepts
        technical_concepts = [c for c in technical_concepts if c.technical_category or c.relevance_score > 0.3]
    
    # Return top concepts
    return sorted(technical_concepts, key=lambda x: x.relevance_score, reverse=True)[:max_concepts]

async def _create_technical_nodes(technical_concepts: List[TechnicalConcept]) -> List[MindMapNode]:
    """Create mind map nodes from technical concepts"""
    nodes = []
    
    for concept in technical_concepts:
        # Determine node type based on technical category
        node_type = NodeType.KEYWORD
        if concept.technical_category == 'programming':
            node_type = NodeType.CONCEPT
        elif concept.technical_category in ['data_science', 'infrastructure']:
            node_type = NodeType.TOPIC
        
        # Calculate node size based on relevance
        size = max(15.0, min(50.0, concept.relevance_score * 50))
        
        node = MindMapNode(
            name=concept.text,
            type=node_type,
            content=f"Technical concept with {concept.frequency} occurrences",
            embedding=concept.embedding,
            size=size,
            metadata={
                "relevance_score": concept.relevance_score,
                "frequency": concept.frequency,
                "technical_category": concept.technical_category,
                "extraction_method": "vector_analysis"
            }
        )
        nodes.append(node)
    
    return nodes

async def _calculate_vector_relationships(
    nodes: List[MindMapNode],
    technical_concepts: List[TechnicalConcept],
    threshold: float = 0.4
) -> List[MindMapRelationship]:
    """Calculate relationships between nodes using vector similarity"""
    relationships = []
    
    # Create embeddings matrix for similarity calculation
    embeddings = []
    valid_nodes = []
    
    for i, node in enumerate(nodes):
        if node.embedding and len(node.embedding) > 0:
            embeddings.append(node.embedding)
            valid_nodes.append((i, node))
    
    # If we have valid embeddings, use vector similarity
    if len(embeddings) >= 2:
        # MEMORY OPTIMIZATION: Process in smaller chunks
        max_nodes_for_vector = 50  # Limit to prevent memory issues
        if len(embeddings) > max_nodes_for_vector:
            embeddings = embeddings[:max_nodes_for_vector]
            valid_nodes = valid_nodes[:max_nodes_for_vector]
        
        try:
            embeddings_matrix = np.array(embeddings)
            
            # Use chunked similarity calculation for large matrices
            chunk_size = 20
            similarity_results = []
            
            for i in range(0, len(embeddings_matrix), chunk_size):
                end_i = min(i + chunk_size, len(embeddings_matrix))
                chunk_similarities = cosine_similarity(embeddings_matrix[i:end_i], embeddings_matrix)
                similarity_results.append(chunk_similarities)
            
            # Combine results
            similarity_matrix = np.vstack(similarity_results)
            
            # Create relationships based on similarity
            for i, (node_i_idx, node_i) in enumerate(valid_nodes):
                for j, (node_j_idx, node_j) in enumerate(valid_nodes):
                    if i < j and i < len(similarity_matrix) and j < len(similarity_matrix[0]):
                        similarity = similarity_matrix[i][j]
                        
                        if similarity > threshold:
                            # Determine relationship type based on similarity and categories
                            rel_type = RelationshipType.SIMILAR_TO
                            
                            node_i_concept = technical_concepts[node_i_idx] if node_i_idx < len(technical_concepts) else None
                            node_j_concept = technical_concepts[node_j_idx] if node_j_idx < len(technical_concepts) else None
                            
                            if (node_i_concept and node_j_concept and 
                                node_i_concept.technical_category == node_j_concept.technical_category):
                                rel_type = RelationshipType.RELATED_TO
                            
                            relationship = MindMapRelationship(
                                source_id=node_i.id,
                                target_id=node_j.id,
                                type=rel_type,
                                weight=float(similarity),
                                similarity=float(similarity),
                                metadata={
                                    "calculation_method": "vector_similarity",
                                    "similarity_score": float(similarity)
                                }
                            )
                            relationships.append(relationship)
                            
        except MemoryError:
            print("Memory error in vector similarity calculation, falling back to text similarity")
            # Fall through to the fallback method
        except Exception as e:
            print(f"Error in vector similarity calculation: {e}, falling back to text similarity")
            # Fall through to the fallback method
    
    # Fallback method (existing code)
    if not relationships:  # Only use fallback if vector method failed or no embeddings
        # Fallback: Create relationships based on technical categories and text similarity
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i < j:  # Avoid duplicate relationships
                    concept_i = technical_concepts[i] if i < len(technical_concepts) else None
                    concept_j = technical_concepts[j] if j < len(technical_concepts) else None
                    
                    if concept_i and concept_j:
                        # Calculate text similarity using simple methods
                        text_similarity = _calculate_text_similarity(concept_i.text, concept_j.text)
                        
                        # Check if they're in the same technical category
                        same_category = (concept_i.technical_category and 
                                       concept_j.technical_category and
                                       concept_i.technical_category == concept_j.technical_category)
                        
                        # Create relationship if similar enough or same category
                        if text_similarity > threshold or same_category:
                            rel_type = RelationshipType.RELATED_TO if same_category else RelationshipType.SIMILAR_TO
                            weight = 0.8 if same_category else text_similarity
                            
                            relationship = MindMapRelationship(
                                source_id=node_i.id,
                                target_id=node_j.id,
                                type=rel_type,
                                weight=weight,
                                similarity=weight,
                                metadata={
                                    "calculation_method": "text_similarity_fallback",
                                    "similarity_score": text_similarity,
                                    "same_category": same_category
                                }
                            )
                            relationships.append(relationship)
    
    return relationships

def _calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity between two concept names"""
    # Convert to lowercase for comparison
    t1, t2 = text1.lower(), text2.lower()
    
    # Check for substring relationships
    if t1 in t2 or t2 in t1:
        return 0.7
    
    # Check for common words (simple approach)
    words1 = set(t1.split())
    words2 = set(t2.split())
    
    if words1 & words2:  # Common words exist
        jaccard = len(words1 & words2) / len(words1 | words2)
        return jaccard
    
    # Check for similar prefixes/suffixes
    if (len(t1) > 3 and len(t2) > 3 and 
        (t1[:3] == t2[:3] or t1[-3:] == t2[-3:])):
        return 0.5
    
    return 0.0

async def _generate_technical_title(technical_concepts: List[TechnicalConcept]) -> str:
    """Generate a technical title based on extracted concepts"""
    if not technical_concepts:
        return "Technical Concept Map"
    
    # Get top technical concepts
    top_concepts = [c for c in technical_concepts[:5] if c.technical_category]
    
    if not top_concepts:
        top_concepts = technical_concepts[:3]
    
    # Create title from top concepts
    concept_names = [c.text for c in top_concepts[:3]]
    
    if len(concept_names) >= 2:
        return f"{concept_names[0]} & {concept_names[1]} Technical Map"
    elif concept_names:
        return f"{concept_names[0]} Technical Concept Map"
    else:
        return "Vector-Based Technical Mind Map"
