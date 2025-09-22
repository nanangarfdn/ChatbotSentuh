import chromadb
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict, Optional
import numpy as np
from chromadb.config import Settings

class IndonesianVectorEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize Indonesian vector embeddings system
        Using multilingual model that works well with Indonesian text
        """
        print("🔄 Loading SentenceTransformer model...")
        self.model = SentenceTransformer(model_name)
        print("✅ SentenceTransformer model loaded")
        
        # Initialize ChromaDB with optimized settings
        print("🔄 Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        # Create or get collection with optimized settings
        self.collection = self.client.get_or_create_collection(
            name="sentuh_tanahku_knowledge",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 16,  # Reduced for faster search
                "hnsw:ef_construction": 100,  # Reduced for faster indexing
                "hnsw:ef_search": 50  # Reduced for faster search
            }
        )
        print("✅ ChromaDB initialized")
        
        # Cache for embeddings to avoid recomputation
        self._embedding_cache = {}
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        return self.model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
    
    def add_knowledge_to_database(self, processed_data: List[Dict]):
        """Add processed knowledge data to vector database with smart caching"""
        
        # Check if we need to rebuild (data changed)
        current_hash = hash(str(sorted([item['question'] for item in processed_data])))
        
        # Check if collection already has data with same hash
        try:
            existing_items = self.collection.get(limit=1)
            if existing_items and existing_items['metadatas']:
                stored_hash = existing_items['metadatas'][0].get('data_hash')
                if stored_hash == str(current_hash):
                    print(f"✅ Vector database up-to-date ({len(processed_data)} documents)")
                    return
        except Exception:
            pass  # Continue with rebuild if check fails
        
        print(f"🔄 Building vector database for {len(processed_data)} documents...")
        
        documents = []
        metadatas = []
        ids = []
        
        for i, item in enumerate(processed_data):
            # Combine question and answer for better context
            combined_text = f"Q: {item['question']} A: {item['answer']}"
            
            documents.append(combined_text)
            metadatas.append({
                "category": item["category"],
                "question": item["question"],
                "answer": item["answer"],
                "type": "qa_pair",
                "data_hash": str(current_hash)  # Store hash for change detection
            })
            ids.append(f"qa_{i}")
        
        # Add to ChromaDB with embeddings
        print("🔄 Generating embeddings...")
        embeddings = self.encode_texts(documents)
        print("✅ Embeddings generated")
        
        # Clear existing collection safely
        try:
            # Get all existing IDs first
            existing_items = self.collection.get()
            if existing_items and existing_items['ids']:
                # Delete existing items by their IDs
                self.collection.delete(ids=existing_items['ids'])
                print(f"🗑️ Cleared {len(existing_items['ids'])} existing documents")
        except Exception as e:
            print(f"Warning: Could not clear existing collection: {e}")
            # If clearing fails, try to create a new collection
            try:
                self.client.delete_collection("sentuh_tanahku_knowledge")
                self.collection = self.client.create_collection(
                    name="sentuh_tanahku_knowledge",
                    metadata={
                        "hnsw:space": "cosine",
                        "hnsw:M": 16,
                        "hnsw:ef_construction": 100,
                        "hnsw:ef_search": 50
                    }
                )
                print("🔄 Created fresh collection after clearing failed")
            except Exception as e2:
                print(f"Warning: Could not recreate collection: {e2}")
        
        # Add new documents in batches for better performance
        print("🔄 Adding documents to vector database...")
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist()
            )
        
        print(f"✅ Vector database ready with {len(documents)} documents")
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar documents using semantic similarity"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        similar_docs = []
        if results and results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                similar_docs.append({
                    "document": doc,
                    "metadata": results['metadatas'][0][i],
                    "similarity_score": 1 - results['distances'][0][i]  # Convert distance to similarity
                })
        
        return similar_docs
    
    def get_relevant_context(self, query: str, max_context_length: int = 1000) -> str:
        """Get relevant context for a query within token limits"""
        similar_docs = self.search_similar(query, n_results=3)
        
        context_parts = []
        current_length = 0
        
        for doc in similar_docs:
            answer = doc['metadata']['answer']
            if current_length + len(answer) <= max_context_length:
                context_parts.append(f"Q: {doc['metadata']['question']}\nA: {answer}")
                current_length += len(answer)
            else:
                break
        
        return "\n\n".join(context_parts)
    
    def update_knowledge(self, new_data: List[Dict]):
        """Update knowledge base with new data"""
        existing_count = self.collection.count()
        
        documents = []
        metadatas = []
        ids = []
        
        for i, item in enumerate(new_data):
            combined_text = f"Q: {item['question']} A: {item['answer']}"
            
            documents.append(combined_text)
            metadatas.append({
                "category": item["category"],
                "question": item["question"],
                "answer": item["answer"],
                "type": "qa_pair"
            })
            ids.append(f"qa_{existing_count + i}")
        
        # Add new documents
        embeddings = self.encode_texts(documents)
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist()
        )
        
        print(f"Added {len(documents)} new documents to vector database")
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        count = self.collection.count()
        
        # Get category distribution
        results = self.collection.get(include=["metadatas"])
        categories = {}
        
        if results and results['metadatas']:
            for metadata in results['metadatas']:
                category = metadata.get('category', 'Unknown')
                categories[category] = categories.get(category, 0) + 1
        
        return {
            "total_documents": count,
            "categories": categories
        }

if __name__ == "__main__":
    # Initialize embeddings system
    embeddings = IndonesianVectorEmbeddings()
    
    # Load processed data
    with open('processed_knowledge.json', 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    
    # Add to vector database
    embeddings.add_knowledge_to_database(processed_data)
    
    # Test search
    test_query = "Bagaimana cara download aplikasi Sentuh Tanahku?"
    results = embeddings.search_similar(test_query)
    
    print(f"\nTest search for: '{test_query}'")
    for i, result in enumerate(results):
        print(f"{i+1}. Similarity: {result['similarity_score']:.3f}")
        print(f"   Q: {result['metadata']['question']}")
        print(f"   A: {result['metadata']['answer'][:100]}...")
        print()
    
    # Print statistics
    stats = embeddings.get_statistics()
    print("Database Statistics:")
    print(f"Total documents: {stats['total_documents']}")
    print("Categories:", stats['categories'])