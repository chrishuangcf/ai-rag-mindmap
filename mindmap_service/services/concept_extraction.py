import nltk
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

class Concept(BaseModel):
    text: str
    label: str

class VectorConcept(BaseModel):
    """Extended concept with vector analysis capabilities"""
    text: str
    label: str
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency: int = Field(default=1, ge=1)
    embedding: Optional[List[float]] = None
    technical_category: Optional[str] = None
    is_technical: bool = Field(default=False)
    
class ConceptExtractor:
    """Advanced concept extractor with vector analysis"""
    
    def __init__(self):
        self.technical_patterns = {
            'programming': [
                r'\b(?:API|SDK|HTTP|REST|JSON|XML|SQL|NoSQL|AI|ML|GPU|CPU|RAM|SSD|IDE)\b',
                r'\b(?:Python|JavaScript|Java|C\+\+|React|Vue|Django|Flask|FastAPI|Node\.js|TypeScript)\b',
                r'\b(?:Docker|Kubernetes|AWS|Azure|GCP|GitHub|GitLab|CI/CD|DevOps)\b',
                r'\b(?:framework|library|package|module|component|service|microservice)\b',
                r'\b(?:function|method|class|object|variable|parameter|argument)\b'
            ],
            'data_science': [
                r'\b(?:machine learning|deep learning|neural network|artificial intelligence)\b',
                r'\b(?:tensor|matrix|vector|embedding|feature|model|algorithm|training)\b',
                r'\b(?:regression|classification|clustering|optimization|gradient|loss)\b',
                r'\b(?:dataset|dataframe|preprocessing|normalization|visualization)\b',
                r'\b(?:pandas|numpy|sklearn|tensorflow|pytorch|keras)\b'
            ],
            'infrastructure': [
                r'\b(?:server|client|database|repository|cache|queue|message broker)\b',
                r'\b(?:load balancer|proxy|gateway|router|firewall|VPN)\b',
                r'\b(?:authentication|authorization|encryption|security|SSL|TLS)\b',
                r'\b(?:monitoring|logging|metrics|alerting|observability|tracing)\b',
                r'\b(?:container|cluster|orchestration|deployment|scaling)\b'
            ],
            'web_development': [
                r'\b(?:HTML|CSS|DOM|AJAX|WebSocket|GraphQL|gRPC)\b',
                r'\b(?:frontend|backend|fullstack|responsive|mobile-first)\b',
                r'\b(?:webpack|babel|sass|less|bootstrap|tailwind)\b',
                r'\b(?:SPA|PWA|SSR|SSG|CSR|hydration)\b'
            ]
        }
        
        self.stop_words = {
            'that', 'this', 'with', 'from', 'have', 'will', 'been',
            'their', 'would', 'there', 'could', 'other', 'more', 'very',
            'what', 'know', 'just', 'time', 'year', 'work', 'well', 'way',
            'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day',
            'most', 'us', 'is', 'at', 'it', 'we', 'be', 'he', 'they', 'are',
            'for', 'an', 'on', 'as', 'you', 'do', 'by', 'to', 'of', 'and',
            'in', 'the', 'a', 'can', 'had', 'her', 'was', 'one', 'our',
            'but', 'not', 'or', 'his', 'him', 'she', 'has', 'how', 'then',
            'than', 'them', 'each', 'which', 'when', 'where', 'why', 'who'
        }
    
    def extract_vector_concepts(
        self, 
        texts: List[str], 
        embeddings: Optional[List[List[float]]] = None,
        max_concepts: int = 30,
        min_frequency: int = 2
    ) -> List[VectorConcept]:
        """Extract concepts using vector analysis and technical keyword detection"""
        
        if not texts:
            return []
        
        # Combine all texts for analysis
        combined_text = " ".join(texts)
        
        # Extract technical concepts
        technical_concepts = self._extract_technical_concepts(combined_text)
        
        # Extract general concepts using NLTK
        general_concepts = self._extract_general_concepts(combined_text)
        
        # Combine and deduplicate
        all_concepts = {}
        
        # Add technical concepts with higher priority
        for concept in technical_concepts:
            all_concepts[concept.text.lower()] = concept
        
        # Add general concepts if not already present
        for concept in general_concepts:
            key = concept.text.lower()
            if key not in all_concepts:
                all_concepts[key] = VectorConcept(
                    text=concept.text,
                    label=concept.label,
                    frequency=1,
                    relevance_score=0.5,
                    is_technical=False
                )
        
        # Convert embedding information if available
        if embeddings:
            concept_embeddings = self._calculate_concept_embeddings(
                list(all_concepts.values()), 
                texts, 
                embeddings
            )
            
            for i, concept in enumerate(all_concepts.values()):
                if i < len(concept_embeddings):
                    concept.embedding = concept_embeddings[i]
        
        # Filter by frequency and sort by relevance
        filtered_concepts = [
            concept for concept in all_concepts.values() 
            if concept.frequency >= min_frequency
        ]
        
        # Sort by relevance score and technical priority
        filtered_concepts.sort(key=lambda x: (x.is_technical, x.relevance_score), reverse=True)
        
        return filtered_concepts[:max_concepts]
    
    def _extract_technical_concepts(self, text: str) -> List[VectorConcept]:
        """Extract technical concepts with category classification"""
        concepts = []
        concept_freq = Counter()
        concept_categories = {}
        
        # Extract using technical patterns
        self._extract_pattern_concepts(text, concept_freq, concept_categories)
        
        # Extract acronyms and technical terms
        self._extract_acronyms(text, concept_freq, concept_categories)
        
        # Extract camelCase and PascalCase identifiers  
        self._extract_code_identifiers(text, concept_freq, concept_categories)
        
        # Create VectorConcept objects
        for concept_text, frequency in concept_freq.items():
            relevance = self._calculate_technical_relevance(concept_text, frequency)
            category = concept_categories.get(concept_text)
            
            concepts.append(VectorConcept(
                text=concept_text,
                label='technical_concept',
                frequency=frequency,
                relevance_score=relevance,
                technical_category=category,
                is_technical=True
            ))
        
        return concepts
    
    def _extract_pattern_concepts(self, text: str, concept_freq: Counter, concept_categories: Dict):
        """Extract concepts using technical patterns"""
        for category, patterns in self.technical_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    normalized = self._normalize_concept(match)
                    if normalized and len(normalized) > 2:
                        concept_freq[normalized] += 1
                        concept_categories[normalized] = category
    
    def _extract_acronyms(self, text: str, concept_freq: Counter, concept_categories: Dict):
        """Extract acronyms and technical abbreviations"""
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
        excluded_acronyms = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'YOU', 'CAN', 'NOT'}
        
        for acronym in acronyms:
            if len(acronym) >= 2 and acronym not in excluded_acronyms:
                concept_freq[acronym] += 1
                concept_categories[acronym] = 'technical_acronym'
    
    def _extract_code_identifiers(self, text: str, concept_freq: Counter, concept_categories: Dict):
        """Extract camelCase and PascalCase programming identifiers"""
        camel_case = re.findall(r'\b[a-z][a-zA-Z]*[A-Z][a-zA-Z]*\b', text)
        pascal_case = re.findall(r'\b[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*\b', text)
        
        for identifier in camel_case + pascal_case:
            if len(identifier) > 4:
                concept_freq[identifier] += 1
                concept_categories[identifier] = 'programming'
    
    def _extract_general_concepts(self, text: str) -> List[Concept]:
        """Extract general concepts using NLTK"""
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        concepts = []

        # Noun Chunking
        grammar = "NP: {<DT>?<JJ>*<NN.*>+}"
        cp = nltk.RegexpParser(grammar)
        tree = cp.parse(tagged)
        
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            phrase = " ".join(word for word, tag in subtree.leaves())
            normalized = self._normalize_concept(phrase)
            if normalized and len(normalized) > 3:
                concepts.append(Concept(text=normalized, label="noun_phrase"))

        # Named Entity Recognition
        try:
            entities = nltk.chunk.ne_chunk(tagged)
            for entity in entities:
                if isinstance(entity, nltk.tree.Tree) and entity.label() != 'S':
                    entity_text = ' '.join([word for word, tag in entity.leaves()])
                    normalized = self._normalize_concept(entity_text)
                    if normalized:
                        concepts.append(Concept(text=normalized, label=entity.label()))
        except Exception:
            pass  # Skip NER if it fails
        
        return concepts
    
    def _normalize_concept(self, concept: str) -> Optional[str]:
        """Normalize concept text"""
        if not concept:
            return None
        
        # Clean and normalize
        normalized = re.sub(r'[^\w\s\-\.]', '', concept).strip()
        
        # Skip if too short or all numbers
        if len(normalized) < 2 or normalized.isdigit():
            return None
        
        # Skip common stop words
        if normalized.lower() in self.stop_words:
            return None
        
        return normalized.title()
    
    def _calculate_technical_relevance(self, concept: str, frequency: int) -> float:
        """Calculate relevance score for technical concepts"""
        base_score = min(1.0, frequency / 5.0)  # Base score from frequency
        
        # Boost for length (longer technical terms are often more specific)
        length_boost = min(0.3, len(concept) / 20.0)
        
        # Boost for technical indicators
        technical_boost = 0.0
        
        # Check for technical patterns
        for patterns in self.technical_patterns.values():
            for pattern in patterns:
                if re.search(pattern, concept, re.IGNORECASE):
                    technical_boost = 0.4
                    break
            if technical_boost > 0:
                break
        
        # Boost for acronyms
        if concept.isupper() and len(concept) >= 2:
            technical_boost = max(technical_boost, 0.3)
        
        # Boost for camelCase/PascalCase
        if re.match(r'^[a-zA-Z]*[A-Z][a-zA-Z]*$', concept):
            technical_boost = max(technical_boost, 0.2)
        
        return min(1.0, base_score + length_boost + technical_boost)
    
    def _calculate_concept_embeddings(
        self, 
        concepts: List[VectorConcept], 
        texts: List[str], 
        embeddings: List[List[float]]
    ) -> List[List[float]]:
        """Calculate embeddings for concepts based on document embeddings"""
        if not embeddings or not concepts:
            return []
        
        # Create TF-IDF vectors for concepts
        concept_texts = [concept.text for concept in concepts]
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=min(100, len(concept_texts)),
                stop_words='english',
                lowercase=True
            )
            
            # Fit on all text to get vocabulary
            all_text = " ".join(texts + concept_texts)
            vectorizer.fit([all_text])
            
            # Transform concept texts
            concept_vectors = vectorizer.transform(concept_texts).toarray()
            
            # If we have document embeddings, enhance concept embeddings
            if embeddings and len(embeddings) > 0:
                doc_embeddings = np.array([emb for emb in embeddings if emb])
                if len(doc_embeddings) > 0:
                    # Calculate similarity between concepts and documents
                    enhanced_embeddings = []
                    
                    for concept_vec in concept_vectors:
                        if len(concept_vec) > 0:
                            # Simple approach: use TF-IDF vector as embedding
                            enhanced_embeddings.append(concept_vec.tolist())
                        else:
                            enhanced_embeddings.append([])
                    
                    return enhanced_embeddings
            
            return [vec.tolist() for vec in concept_vectors]
            
        except Exception:
            # Fallback: return empty embeddings
            return [[] for _ in concepts]

# Global extractor instance
_extractor = ConceptExtractor()

def extract_concepts_from_text(text: str) -> List[Concept]:
    """
    Extracts concepts (noun chunks and entities) from the provided text using NLTK.
    """
    return _extractor._extract_general_concepts(text)

def extract_vector_concepts_from_texts(
    texts: List[str],
    embeddings: Optional[List[List[float]]] = None,
    max_concepts: int = 30
) -> List[VectorConcept]:
    """
    Extract concepts from multiple texts using vector analysis
    """
    return _extractor.extract_vector_concepts(texts, embeddings, max_concepts)

def extract_technical_keywords(text: str, max_keywords: int = 20) -> List[VectorConcept]:
    """
    Extract technical keywords from text with relevance scoring
    """
    technical_concepts = _extractor._extract_technical_concepts(text)
    return sorted(technical_concepts, key=lambda x: x.relevance_score, reverse=True)[:max_keywords]