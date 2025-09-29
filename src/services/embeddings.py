"""
Custom embeddings and vector store implementations for the LangChain RAG API.
Includes TF-IDF embeddings and in-memory vector store for Redis fallback.
"""

import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class SimpleTFIDFEmbeddings:
    """Custom TF-IDF based embeddings for local processing."""
    
    def __init__(self, max_features=500):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        self.fitted = False
        self.max_features = max_features

    def embed_documents(self, texts):
        """Generate embeddings for a list of documents."""
        # Handle empty input
        if not texts:
            return []
            
        # Extract text content based on input type
        text_contents = []
        for item in texts:
            if hasattr(item, 'page_content'):
                text_contents.append(str(item.page_content))
            elif isinstance(item, str):
                text_contents.append(item)
            elif isinstance(item, dict):
                text_contents.append(str(item.get('content', item.get('text', str(item)))))
            else:
                text_contents.append(str(item))
            
        if not self.fitted:
            self.vectorizer.fit(text_contents)
            self.fitted = True
        vectors = self.vectorizer.transform(text_contents).toarray()
        padded_vectors = np.zeros((len(vectors), self.max_features))
        min_features = min(vectors.shape[1], self.max_features)
        padded_vectors[:, :min_features] = vectors[:, :min_features]
        return padded_vectors.tolist()

    def embed_query(self, text):
        """Generate embedding for a single query."""
        if not self.fitted:
            raise ValueError("Must embed documents first")
        
        # Ensure text is a string
        if hasattr(text, 'page_content'):
            query_text = str(text.page_content)
        elif isinstance(text, dict):
            if 'page_content' in text:
                query_text = str(text['page_content'])
            elif 'content' in text:
                query_text = str(text['content'])
            elif 'text' in text:
                query_text = str(text['text'])
            else:
                query_text = str(text)
        else:
            query_text = str(text)
            
        vector = self.vectorizer.transform([query_text]).toarray()[0]
        padded_vector = np.zeros(self.max_features)
        min_features = min(len(vector), self.max_features)
        padded_vector[:min_features] = vector[:min_features]
        return padded_vector.tolist()

    def save_vectorizer(self, r_client, key: str):
        """Save the fitted vectorizer to Redis or in-memory cache."""
        vectorizer_data = {
            'vectorizer': self.vectorizer,
            'fitted': self.fitted,
            'max_features': self.max_features
        }
        if r_client:
            r_client.set(key, pickle.dumps(vectorizer_data))
        else:
            # Fallback to in-memory cache
            if not hasattr(self, '_memory_cache'):
                self.__class__._memory_cache = {}
            self.__class__._memory_cache[key] = vectorizer_data

    def load_vectorizer(self, r_client, key: str):
        """Load the fitted vectorizer from Redis or in-memory cache."""
        try:
            if r_client:
                print(f"Loading vectorizer from Redis with key: {key}")
                data_bytes = r_client.get(key)
                if data_bytes is None:
                    raise ValueError(f"Vectorizer not found in Redis: {key}")
                data = pickle.loads(data_bytes)
                print("Successfully loaded vectorizer from Redis")
            else:
                print(f"Loading vectorizer from memory cache with key: {key}")
                # Fallback to in-memory cache
                if not hasattr(self, '_memory_cache') or key not in self.__class__._memory_cache:
                    raise ValueError(f"Vectorizer not found in memory cache: {key}")
                data = self.__class__._memory_cache[key]
                print("Successfully loaded vectorizer from memory cache")
            
            self.vectorizer = data['vectorizer']
            self.fitted = data['fitted']
            self.max_features = data['max_features']
            print(f"Vectorizer loaded successfully, fitted: {self.fitted}, max_features: {self.max_features}")
            
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
            raise

    @classmethod
    def from_saved(cls, r_client, key: str):
        """Create instance from saved vectorizer in Redis or in-memory cache."""
        try:
            print(f"Creating SimpleTFIDFEmbeddings from saved data with key: {key}")
            instance = cls()
            instance.load_vectorizer(r_client, key)
            print("Successfully created SimpleTFIDFEmbeddings from saved data")
            return instance
        except Exception as e:
            print(f"Error creating SimpleTFIDFEmbeddings from saved data: {e}")
            raise


class SimpleInMemoryVectorStore:
    """Simple in-memory vector store as fallback when Redis is not available."""
    
    def __init__(self, embeddings, documents=None):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
        
        if documents:
            self.add_documents(documents)
    
    def add_documents(self, documents):
        """Add documents to the vector store."""
        print(f"Adding {len(documents)} documents to in-memory vector store")
        if documents:
            print(f"First document type: {type(documents[0])}")
            print(f"First document has page_content: {hasattr(documents[0], 'page_content')}")
        
        vectors = self.embeddings.embed_documents(documents)
        
        self.documents.extend(documents)
        self.vectors.extend(vectors)
    
    def similarity_search(self, query: str, k: int = 4):
        """Simple similarity search using cosine similarity."""
        if not self.documents:
            print(f"No documents in vector store for similarity search")
            return []
        
        print(f"Performing similarity search with {len(self.documents)} documents for query: '{query}'")
        
        try:
            query_vector = self.embeddings.embed_query(query)
            print(f"Query vector generated successfully, length: {len(query_vector)}")
        except Exception as e:
            print(f"Error generating query vector: {e}")
            return []
        
        similarities = []
        
        for i, doc_vector in enumerate(self.vectors):
            try:
                # Calculate cosine similarity
                dot_product = sum(a * b for a, b in zip(query_vector, doc_vector))
                norm_query = sum(a * a for a in query_vector) ** 0.5
                norm_doc = sum(a * a for a in doc_vector) ** 0.5
                
                if norm_query > 0 and norm_doc > 0:
                    similarity = dot_product / (norm_query * norm_doc)
                else:
                    similarity = 0.0
                    
                similarities.append((i, similarity))
            except Exception as e:
                print(f"Error calculating similarity for document {i}: {e}")
                continue
        
        print(f"Calculated {len(similarities)} similarities")
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = [self.documents[i] for i, similarity in similarities[:k]]
        
        print(f"Returning top {len(top_results)} results")
        if top_results:
            print(f"Top similarity score: {similarities[0][1] if similarities else 'None'}")
        
        return top_results
    
    def as_retriever(self, search_kwargs=None):
        """Return a retriever interface."""
        return SimpleRetriever(self, search_kwargs=search_kwargs)


class SimpleRetriever:
    """Simple retriever wrapper for in-memory vector store."""
    
    def __init__(self, vectorstore, search_kwargs=None):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs or {}
    
    def get_relevant_documents(self, query: str):
        k = self.search_kwargs.get('k', 4)
        return self.vectorstore.similarity_search(query, k=k)
    
    def invoke(self, query: str, config=None, **kwargs):
        """LangChain compatibility method."""
        return self.get_relevant_documents(query)
    
    def with_config(self, config=None, **kwargs):
        """LangChain compatibility method - returns self for chaining."""
        return self
    
    def __call__(self, query: str):
        """Make the retriever callable."""
        return self.get_relevant_documents(query)


def create_redis_schema(embedding_dim: int = 500):
    """Create Redis search schema for vector storage - legacy function for compatibility."""
    try:
        from src.core.config import REDIS_SEARCH_AVAILABLE
        from redis.commands.search.field import VectorField, TextField
        
        if not REDIS_SEARCH_AVAILABLE:
            return None
            
        return [
            TextField("content"),
            TextField("source_url"),
            VectorField(
                "content_vector",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": embedding_dim,
                    "DISTANCE_METRIC": "COSINE",
                    "INITIAL_CAP": 1000,
                    "M": 16,
                    "EF_CONSTRUCTION": 200
                }
            )
        ]
    except ImportError:
        return None
