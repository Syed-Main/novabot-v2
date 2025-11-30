"""
Embeddings store using FastEmbed (Lightweight ONNX)
"""
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple
import os

# Use FastEmbed (Lightweight ONNX)
try:
    from fastembed import TextEmbedding
    print("Loading FastEmbed model (BGE-Small)...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    EMBEDDING_DIMENSION = 384
    print("[OK] FastEmbed ready")
except Exception as e:
    print(f"[FAIL] Failed to load embedding model: {e}")
    print("Run: pip install fastembed")
    raise


class EmbeddingsStore:
    def __init__(self):
        self.embeddings = []
        self.qa_pairs = []
        self.dimension = EMBEDDING_DIMENSION
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text using FastEmbed"""
        try:
            # FastEmbed returns a generator of vectors
            return list(embedding_model.embed([text]))[0].tolist()
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0] * self.dimension
    
    def get_batch_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Get embeddings for multiple texts in batches"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # FastEmbed returns a generator, convert to list
                embeddings = list(embedding_model.embed(batch))
                # Convert numpy arrays to lists
                embeddings = [e.tolist() for e in embeddings]
                all_embeddings.extend(embeddings)
                print(f"  Embedded {len(all_embeddings)}/{len(texts)} texts...")
            except Exception as e:
                print(f"Error in batch: {e}")
                all_embeddings.extend([[0] * self.dimension] * len(batch))
        
        return all_embeddings
    
    def build_from_processed_data(self, processed_data_path: str = 'data/processed_chats.json'):
        """Build embeddings from processed Q&A data"""
        print("Loading processed data...")
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.qa_pairs = data
        
        print(f"Creating embeddings for {len(data)} Q&A pairs...")
        
        # Create combined text for better semantic search
        texts = []
        for qa in data:
            # Combine question and answer for richer context
            combined = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
            texts.append(combined)
        
        # Get embeddings in batches
        self.embeddings = self.get_batch_embeddings(texts)
        
        print("[OK] Embeddings created successfully!")
    
    def save(self, path: str = 'data/embeddings.pkl'):
        """Save embeddings and Q&A pairs to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'embeddings': np.array(self.embeddings),
            'qa_pairs': self.qa_pairs
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"[OK] Embeddings saved to {path}")
    
    def load(self, path: str = 'data/embeddings.pkl'):
        """Load embeddings from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings'].tolist()
        self.qa_pairs = data['qa_pairs']
        
        print(f"[OK] Loaded {len(self.embeddings)} embeddings")
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search_similar(self, query: str, top_k: int = 5, category_filter: str = None) -> List[Tuple[Dict, float]]:
        """Search for similar Q&A pairs"""
        if not self.embeddings:
            print("No embeddings loaded!")
            return []
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for idx, embedding in enumerate(self.embeddings):
            # Apply category filter if specified
            if category_filter and self.qa_pairs[idx].get('category') != category_filter:
                continue
            
            similarity = self.cosine_similarity(query_embedding, embedding)
            similarities.append((idx, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K results
        results = []
        for idx, score in similarities[:top_k]:
            results.append((self.qa_pairs[idx], score))
        
        return results
    
    def get_examples_by_category(self, category: str, limit: int = 3) -> List[Dict]:
        """Get example Q&A pairs from a specific category"""
        examples = [qa for qa in self.qa_pairs if qa.get('category') == category]
        return examples[:limit]


if __name__ == "__main__":
    # Build embeddings
    store = EmbeddingsStore()
    store.build_from_processed_data('data/processed_chats.json')
    store.save('data/embeddings.pkl')
    
    # Test search
    print("\n--- Testing Search ---")
    results = store.search_similar("How do I find research opportunities?", top_k=3)
    
    print("\nTop 3 Similar Q&A Pairs:")
    for qa, score in results:
        print(f"\nSimilarity: {score:.3f}")
        print(f"Q: {qa['question'][:80]}...")
        print(f"A: {qa['answer'][:80]}...")