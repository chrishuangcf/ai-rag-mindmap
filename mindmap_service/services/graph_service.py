"""
Graph Service - Handles Neo4j operations and graph algorithms
"""

import asyncio
import json
from typing import List, Dict, Optional, Tuple, Any
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import AsyncSession

from core.models import (
    MindMap, MindMapNode, MindMapRelationship, NodeType, 
    RelationshipType, GraphAnalysisResult, MindMapStats
)
from core.database import get_neo4j_session
from core.config import settings
from services.concept_extraction import ConceptExtractor


class GraphService:
    """Service for managing graph operations and Neo4j interactions"""
    
    def __init__(self):
        self.session = None
        self.concept_extractor = ConceptExtractor()  # Initialize the concept extractor
    
    async def create_mindmap(
        self, 
        title: str, 
        cache_hashes: List[str],
        description: Optional[str] = None,
        max_nodes: int = 100,
        similarity_threshold: float = 0.3
    ) -> MindMap:
        """Create a new mind map from RAG data"""
        
        # Fetch RAG data for the given cache hashes
        rag_data = await self._fetch_rag_data(cache_hashes)
        
        # Extract concepts and create nodes
        nodes = await self._create_nodes_from_rag_data(rag_data, max_nodes)
        
        # Calculate relationships between nodes
        relationships = await self._calculate_relationships(nodes, similarity_threshold)
        
        # Create mind map
        mindmap = MindMap(
            title=title,
            description=description,
            nodes=nodes,
            relationships=relationships,
            cache_hashes=cache_hashes
        )
        
        # Store in Neo4j
        await self._store_mindmap_in_neo4j(mindmap)
        
        return mindmap
    
    async def get_mindmap(self, mindmap_id: str) -> Optional[MindMap]:
        """Retrieve a mind map by ID"""
        async with await get_neo4j_session() as session:
            # Fetch mindmap metadata
            result = await session.run(
                """
                MATCH (m:MindMap {id: $mindmap_id})
                RETURN m
                """,
                mindmap_id=mindmap_id
            )
            mindmap_record = await result.single()
            
            if not mindmap_record:
                return None
            
            # Fetch nodes
            nodes_result = await session.run(
                """
                MATCH (m:MindMap {id: $mindmap_id})-[:CONTAINS]->(n)
                RETURN n
                """,
                mindmap_id=mindmap_id
            )
            
            nodes = []
            async for record in nodes_result:
                node_data = dict(record["n"])
                
                # Deserialize JSON strings back to objects
                if node_data.get('metadata') and isinstance(node_data['metadata'], str):
                    try:
                        node_data['metadata'] = json.loads(node_data['metadata'])
                    except json.JSONDecodeError:
                        node_data['metadata'] = {}
                
                if node_data.get('embedding') and isinstance(node_data['embedding'], str):
                    try:
                        node_data['embedding'] = json.loads(node_data['embedding'])
                    except json.JSONDecodeError:
                        node_data['embedding'] = None
                
                if node_data.get('position') and isinstance(node_data['position'], str):
                    try:
                        node_data['position'] = json.loads(node_data['position'])
                    except json.JSONDecodeError:
                        node_data['position'] = None
                
                nodes.append(MindMapNode(**node_data))
            
            # Fetch relationships
            rels_result = await session.run(
                """
                MATCH (m:MindMap {id: $mindmap_id})-[:CONTAINS]->(n1)
                MATCH (m)-[:CONTAINS]->(n2)
                MATCH (n1)-[r]->(n2)
                RETURN n1.id as source_id, n2.id as target_id, type(r) as rel_type, r
                """,
                mindmap_id=mindmap_id
            )
            
            relationships = []
            async for record in rels_result:
                rel_data = dict(record["r"])
                rel_data.update({
                    "source_id": record["source_id"],
                    "target_id": record["target_id"],
                    "type": record["rel_type"]
                })
                
                # Deserialize JSON strings back to objects
                if rel_data.get('metadata') and isinstance(rel_data['metadata'], str):
                    try:
                        rel_data['metadata'] = json.loads(rel_data['metadata'])
                    except json.JSONDecodeError:
                        rel_data['metadata'] = {}
                
                relationships.append(MindMapRelationship(**rel_data))
            
            # Create and return mindmap
            mindmap_data = dict(mindmap_record["m"])
            
            # Convert Neo4j DateTime objects to Python datetime objects
            if mindmap_data.get('created_at') and hasattr(mindmap_data['created_at'], 'to_native'):
                mindmap_data['created_at'] = mindmap_data['created_at'].to_native()
            if mindmap_data.get('updated_at') and hasattr(mindmap_data['updated_at'], 'to_native'):
                mindmap_data['updated_at'] = mindmap_data['updated_at'].to_native()
            
            mindmap_data.update({
                "nodes": nodes,
                "relationships": relationships
            })
            
            return MindMap(**mindmap_data)
    
    async def update_mindmap(
        self, 
        mindmap_id: str, 
        add_cache_hashes: List[str] = None
    ) -> MindMap:
        """Update an existing mind map"""
        mindmap = await self.get_mindmap(mindmap_id)
        if not mindmap:
            raise ValueError(f"Mind map with ID {mindmap_id} not found")
        
        # Update cache hashes
        if add_cache_hashes:
            mindmap.cache_hashes.extend(add_cache_hashes)
        
        # Re-generate nodes and relationships
        rag_data = await self._fetch_rag_data(mindmap.cache_hashes)
        new_nodes = await self._create_nodes_from_rag_data(rag_data)
        new_relationships = await self._calculate_relationships(new_nodes)
        
        mindmap.nodes = new_nodes
        mindmap.relationships = new_relationships
        
        # Update in Neo4j
        await self._update_mindmap_in_neo4j(mindmap)
        
        return mindmap
    
    async def analyze_graph(self, mindmap_id: str) -> GraphAnalysisResult:
        """Perform graph analysis on a mind map"""
        mindmap = await self.get_mindmap(mindmap_id)
        if not mindmap:
            raise ValueError(f"Mind map with ID {mindmap_id} not found")
        
        # Create NetworkX graph for analysis
        G = nx.Graph()
        
        # Add nodes
        for node in mindmap.nodes:
            G.add_node(node.id, **node.dict())
        
        # Add edges
        for rel in mindmap.relationships:
            G.add_edge(rel.source_id, rel.target_id, weight=rel.weight)
        
        # Calculate centrality measures
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        # Find communities
        communities = list(nx.community.greedy_modularity_communities(G))
        
        # Find key paths (shortest paths between high-centrality nodes)
        central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        key_paths = []
        for i, (node1, _) in enumerate(central_nodes):
            for node2, _ in central_nodes[i+1:]:
                try:
                    path = nx.shortest_path(G, node1, node2)
                    key_paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        # Identify outliers (nodes with low connectivity)
        outliers = [node for node, degree in G.degree() if degree <= 1]
        
        return GraphAnalysisResult(
            central_concepts=[
                {"id": node_id, "centrality": cent, "betweenness": betweenness.get(node_id, 0)}
                for node_id, cent in central_nodes
            ],
            communities=[[node for node in community] for community in communities],
            key_paths=key_paths,
            outliers=outliers,
            similarity_patterns=self._analyze_similarity_patterns(mindmap.relationships)
        )
    
    async def get_mindmap_stats(self, mindmap_id: str) -> MindMapStats:
        """Get statistics for a mind map"""
        mindmap = await self.get_mindmap(mindmap_id)
        if not mindmap:
            raise ValueError(f"Mind map with ID {mindmap_id} not found")
        
        # Count node types
        node_types_count = {}
        for node in mindmap.nodes:
            node_types_count[node.type] = node_types_count.get(node.type, 0) + 1
        
        # Count relationship types
        rel_types_count = {}
        for rel in mindmap.relationships:
            rel_types_count[rel.type] = rel_types_count.get(rel.type, 0) + 1
        
        # Calculate graph metrics
        total_nodes = len(mindmap.nodes)
        total_relationships = len(mindmap.relationships)
        
        # Calculate average node degree
        node_degrees = {}
        for rel in mindmap.relationships:
            node_degrees[rel.source_id] = node_degrees.get(rel.source_id, 0) + 1
            node_degrees[rel.target_id] = node_degrees.get(rel.target_id, 0) + 1
        
        avg_node_degree = sum(node_degrees.values()) / len(node_degrees) if node_degrees else 0
        max_node_degree = max(node_degrees.values()) if node_degrees else 0
        
        # Calculate density
        max_possible_edges = total_nodes * (total_nodes - 1) / 2
        density = total_relationships / max_possible_edges if max_possible_edges > 0 else 0
        
        return MindMapStats(
            total_nodes=total_nodes,
            total_relationships=total_relationships,
            node_types_count=node_types_count,
            relationship_types_count=rel_types_count,
            avg_node_degree=avg_node_degree,
            max_node_degree=max_node_degree,
            density=density,
            connected_components=1  # Simplified for now
        )
    
    async def _fetch_rag_data(self, cache_hashes: List[str]) -> Dict[str, Any]:
        """Fetch RAG data from the main application"""
        import httpx
        import json
        
        rag_data = {
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
                    
                    # Combine documents and embeddings
                    documents = docs_data.get("documents", [])
                    embeddings = embeddings_data.get("embeddings", [])
                    
                    for i, doc in enumerate(documents):
                        rag_data["documents"].append({
                            "content": doc.get("content", "")
                        })
                        
                        # Get corresponding embedding if available
                        if i < len(embeddings):
                            rag_data["embeddings"].append(embeddings[i])
                        else:
                            rag_data["embeddings"].append([])
                            
                        rag_data["metadata"].append(doc.get("metadata", {}))

                except httpx.HTTPStatusError as e:
                    print(f"HTTP error fetching RAG data for cache {cache_hash}: {e}")
                    raise ValueError(f"Failed to fetch RAG data for cache {cache_hash}") from e
                except Exception as e:
                    print(f"Failed to fetch RAG data for cache {cache_hash}: {e}")
                    raise ValueError(f"Failed to fetch RAG data for cache {cache_hash}") from e
        
        return rag_data
    
    async def _create_nodes_from_rag_data(
        self, 
        rag_data: Dict[str, Any], 
        max_nodes: int = 100
    ) -> List[MindMapNode]:
        """Create mind map nodes from RAG data using advanced concept extraction"""
        nodes = []
        
        documents = rag_data.get("documents", [])
        embeddings = rag_data.get("embeddings", [])
        metadata = rag_data.get("metadata", [])
        
        if not documents:
            print("⚠️ DEBUG: No documents found in RAG data")
            return nodes
        
        print(f"🔍 DEBUG: Processing {len(documents)} documents for concept extraction")
        
        # Extract all text content for batch concept extraction
        all_texts = []
        for doc in documents:
            content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
            if content and len(content.strip()) > 10:
                all_texts.append(content)
        
        if not all_texts:
            print("⚠️ DEBUG: No valid text content found in documents")
            return nodes
        
        print(f"🔍 DEBUG: Extracted {len(all_texts)} valid text segments")
        
        # Use advanced concept extraction
        try:
            vector_concepts = self.concept_extractor.extract_vector_concepts(
                texts=all_texts,
                embeddings=embeddings if embeddings else None,
                max_concepts=max_nodes,
                min_frequency=1  # Lower threshold for more concepts
            )
            
            print(f"🔍 DEBUG: Extracted {len(vector_concepts)} vector concepts")
            
            # Create nodes from concepts
            for i, concept in enumerate(vector_concepts):
                if len(nodes) >= max_nodes:
                    break
                
                # Determine node type based on concept properties
                node_type = NodeType.CONCEPT
                if concept.is_technical:
                    node_type = NodeType.ENTITY if concept.technical_category == 'technical_acronym' else NodeType.CONCEPT
                
                # Find relevant document for this concept
                source_doc_index = i % len(documents) if documents else 0
                source_meta = metadata[source_doc_index] if source_doc_index < len(metadata) else {}
                
                node = MindMapNode(
                    name=concept.text,
                    type=node_type,
                    content=f"Technical concept: {concept.text}" if concept.is_technical else concept.text,
                    size=min(max(concept.frequency * 5, 10), 50),  # Scale size based on frequency
                    embedding=concept.embedding,
                    metadata={
                        "is_technical": concept.is_technical,
                        "technical_category": concept.technical_category,
                        "relevance_score": concept.relevance_score,
                        "frequency": concept.frequency,
                        "source_url": source_meta.get("source_url", ""),
                        "cache_hash": source_meta.get("cache_hash", ""),
                        "concept_label": concept.label
                    }
                )
                nodes.append(node)
                
                print(f"🔍 DEBUG: Created node '{concept.text}' (technical: {concept.is_technical}, freq: {concept.frequency})")
        
        except Exception as e:
            print(f"❌ DEBUG: Error in concept extraction: {e}")
            import traceback
            print(f"❌ DEBUG: Traceback: {traceback.format_exc()}")
            # Fallback to simple extraction if advanced method fails
            return await self._create_nodes_simple_fallback(rag_data, max_nodes)
        
        print(f"✅ DEBUG: Created {len(nodes)} nodes total")
        return nodes
    
    async def _create_nodes_simple_fallback(
        self, 
        rag_data: Dict[str, Any], 
        max_nodes: int = 100
    ) -> List[MindMapNode]:
        """Fallback simple node creation if advanced extraction fails"""
        nodes = []
        documents = rag_data.get("documents", [])
        embeddings = rag_data.get("embeddings", [])
        metadata = rag_data.get("metadata", [])
        
        # Create document nodes as fallback
        for i, (doc, embedding, meta) in enumerate(zip(documents, embeddings, metadata)):
            if len(nodes) >= max_nodes:
                break
                
            # Extract simple concepts from document content
            concepts = self._extract_concepts(doc.get("content", ""))
            
            # Create concept nodes
            for concept in concepts[:3]:  # Limit concepts per document
                if len(nodes) >= max_nodes:
                    break
                    
                node = MindMapNode(
                    name=concept,
                    type=NodeType.CONCEPT,
                    content=doc.get("content", "")[:500],  # Truncate content
                    embedding=embedding,
                    metadata={
                        "source_url": meta.get("source_url", ""),
                        "cache_hash": meta.get("cache_hash", ""),
                        "document_index": i,
                        "is_technical": False,
                        "fallback_method": True
                    }
                )
                nodes.append(node)
        
        return nodes
    
    async def _calculate_relationships(
        self, 
        nodes: List[MindMapNode], 
        similarity_threshold: float = 0.3
    ) -> List[MindMapRelationship]:
        """Calculate relationships between nodes based on embeddings"""
        relationships = []
        
        # Extract embeddings
        embeddings = []
        valid_nodes = []
        
        for node in nodes:
            if node.embedding:
                embeddings.append(node.embedding)
                valid_nodes.append(node)
        
        if len(embeddings) < 2:
            return relationships
        
        # Calculate similarity matrix
        embeddings_array = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # Create relationships for similar nodes
        for i, node1 in enumerate(valid_nodes):
            for j, node2 in enumerate(valid_nodes[i+1:], i+1):
                similarity = similarity_matrix[i][j]
                
                if similarity > similarity_threshold:
                    relationship = MindMapRelationship(
                        source_id=node1.id,
                        target_id=node2.id,
                        type=RelationshipType.SIMILAR_TO,
                        weight=similarity,
                        similarity=similarity,
                        metadata={
                            "calculation_method": "cosine_similarity",
                            "threshold_used": similarity_threshold
                        }
                    )
                    relationships.append(relationship)
        
        return relationships
    
    async def _store_mindmap_in_neo4j(self, mindmap: MindMap):
        """Store mind map in Neo4j database"""
        async with await get_neo4j_session() as session:
            # Create mindmap node
            await session.run(
                """
                CREATE (m:MindMap {
                    id: $id,
                    title: $title,
                    description: $description,
                    layout: $layout,
                    created_at: $created_at,
                    updated_at: $updated_at,
                    cache_hashes: $cache_hashes
                })
                """,
                **mindmap.dict(exclude={"nodes", "relationships"})
            )
            
            # Create nodes
            for node in mindmap.nodes:
                # Get node type string value (works with both enum and string)
                node_type = node.type if isinstance(node.type, str) else node.type.value
                
                # Prepare node data with serialized complex fields
                node_data = node.dict()
                # Serialize metadata to JSON string
                node_data['metadata'] = json.dumps(node_data.get('metadata', {}))
                # Serialize embedding to JSON string if present
                if node_data.get('embedding'):
                    node_data['embedding'] = json.dumps(node_data['embedding'])
                else:
                    node_data['embedding'] = None
                # Serialize position to JSON string if present
                if node_data.get('position'):
                    node_data['position'] = json.dumps(node_data['position'])
                else:
                    node_data['position'] = None
                
                await session.run(
                    f"""
                    CREATE (n:{node_type.capitalize()} {{
                        id: $id,
                        name: $name,
                        type: $type,
                        content: $content,
                        embedding: $embedding,
                        size: $size,
                        color: $color,
                        position: $position,
                        metadata: $metadata
                    }})
                    """,
                    **node_data
                )
                
                # Link to mindmap
                await session.run(
                    """
                    MATCH (m:MindMap {id: $mindmap_id})
                    MATCH (n {id: $node_id})
                    CREATE (m)-[:CONTAINS]->(n)
                    """,
                    mindmap_id=mindmap.id,
                    node_id=node.id
                )
            
            # Create relationships
            for rel in mindmap.relationships:
                # Get relationship type string value (works with both enum and string)
                rel_type = rel.type if isinstance(rel.type, str) else rel.type.value
                
                # Prepare relationship data with serialized complex fields
                rel_data = rel.dict()
                # Serialize metadata to JSON string
                rel_data['metadata'] = json.dumps(rel_data.get('metadata', {}))
                
                await session.run(
                    f"""
                    MATCH (n1 {{id: $source_id}})
                    MATCH (n2 {{id: $target_id}})
                    CREATE (n1)-[r:{rel_type} {{
                        id: $id,
                        weight: $weight,
                        similarity: $similarity,
                        metadata: $metadata
                    }}]->(n2)
                    """,
                    **rel_data
                )
    
    async def _update_mindmap_in_neo4j(self, mindmap: MindMap):
        """Update mind map in Neo4j database"""
        async with await get_neo4j_session() as session:
            # Delete existing nodes and relationships
            await session.run(
                """
                MATCH (m:MindMap {id: $mindmap_id})-[:CONTAINS]->(n)
                DETACH DELETE n
                """,
                mindmap_id=mindmap.id
            )
            
            # Update mindmap metadata
            await session.run(
                """
                MATCH (m:MindMap {id: $id})
                SET m.updated_at = $updated_at,
                    m.cache_hashes = $cache_hashes
                """,
                id=mindmap.id,
                updated_at=mindmap.updated_at,
                cache_hashes=mindmap.cache_hashes
            )
            
            # Re-create nodes and relationships
            await self._store_mindmap_in_neo4j(mindmap)
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using multiple techniques"""
        import re
        from collections import Counter
        
        if not text or len(text.strip()) < 10:
            return ["Unknown Concept"]
        
        concepts = set()
        
        # Method 1: Extract capitalized words (proper nouns, names, etc.)
        capitalized_words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        concepts.update([word for word in capitalized_words if len(word) > 3])
        
        # Method 2: Extract important technical terms and acronyms
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        concepts.update([word for word in acronyms if 2 <= len(word) <= 6])
        
        # Method 3: Extract quoted terms (often important concepts)
        quoted_terms = re.findall(r'"([^"]{3,30})"', text)
        concepts.update([term.strip() for term in quoted_terms])
        
        # Method 4: Extract common programming/technical patterns
        tech_patterns = [
            r'\b(?:API|SDK|HTTP|REST|JSON|XML|SQL|NoSQL|AI|ML|GPU|CPU)\b',
            r'\b(?:Python|JavaScript|Java|C\+\+|React|Vue|Django|Flask|FastAPI)\b',
            r'\b(?:Docker|Kubernetes|AWS|Azure|GCP|GitHub|GitLab)\b',
            r'\b(?:machine learning|deep learning|neural network|artificial intelligence)\b',
            r'\b(?:database|repository|framework|library|package|module)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.update([match.lower().title() for match in matches])
        
        # Method 5: Extract frequent meaningful words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        # Filter out common words
        stop_words = {'that', 'this', 'with', 'from', 'they', 'have', 'will', 'been', 
                     'their', 'would', 'there', 'could', 'other', 'more', 'very', 
                     'what', 'know', 'just', 'time', 'year', 'work', 'well', 'way',
                     'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day',
                     'most', 'us', 'is', 'at', 'it', 'we', 'be', 'he', 'they', 'are',
                     'for', 'an', 'on', 'as', 'you', 'do', 'by', 'to', 'of', 'and',
                     'in', 'the', 'a', 'can', 'had', 'her', 'was', 'one', 'our',
                     'but', 'not', 'or', 'his', 'him', 'she', 'has', 'how'}
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 4]
        word_freq = Counter(filtered_words)
        
        # Add most frequent words as concepts
        for word, freq in word_freq.most_common(5):
            if freq > 1:  # Must appear more than once
                concepts.add(word.title())
        
        # Convert to list and prioritize
        concept_list = list(concepts)
        
        # If we have too few concepts, add some generic but meaningful ones
        if len(concept_list) < 3:
            # Extract the first few sentences and use them to generate concepts
            sentences = re.split(r'[.!?]', text)[:3]
            for sentence in sentences:
                words_in_sentence = re.findall(r'\b[A-Za-z]{5,}\b', sentence)
                concept_list.extend([word.title() for word in words_in_sentence[:2]])
        
        # Remove duplicates and limit to reasonable number
        final_concepts = list(dict.fromkeys(concept_list))[:8]
        
        return final_concepts if final_concepts else ["Unknown Concept"]
    
    def _analyze_similarity_patterns(self, relationships: List[MindMapRelationship]) -> Dict[str, Any]:
        """Analyze patterns in similarity scores"""
        similarities = [rel.similarity for rel in relationships if rel.similarity is not None]
        
        if not similarities:
            return {}
        
        return {
            "avg_similarity": np.mean(similarities),
            "max_similarity": np.max(similarities),
            "min_similarity": np.min(similarities),
            "std_similarity": np.std(similarities),
            "total_relationships": len(relationships)
        }
