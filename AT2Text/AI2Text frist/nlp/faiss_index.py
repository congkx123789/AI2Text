"""
FAISS indexing for fast similarity search.

Creates FAISS indexes for Word2Vec and Phon2Vec embeddings to enable
fast nearest neighbor search for contextual biasing and N-best rescoring.
"""

import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")


class EmbeddingIndex:
    """
    FAISS index for embedding similarity search.
    
    Creates and manages FAISS indexes for semantic (Word2Vec) and
    phonetic (Phon2Vec) embeddings for fast similarity search.
    """
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension
            index_type: Index type - "flat" (exact) or "ivf" (approximate)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_type = index_type
        
        if index_type == "flat":
            # Exact search
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            # Approximate search (faster for large datasets)
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 1024)  # nlist=1024
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.vocab = []  # Vocabulary for index mapping
    
    def add_embeddings(self, embeddings: np.ndarray, vocab: List[str]):
        """
        Add embeddings to index.
        
        Args:
            embeddings: Embedding vectors (n, dimension)
            vocab: List of vocabulary tokens corresponding to embeddings
        """
        if len(embeddings) != len(vocab):
            raise ValueError("Embeddings and vocabulary length mismatch")
        
        # Normalize embeddings (L2 normalization)
        faiss.normalize_L2(embeddings)
        
        # Add to index
        if self.index_type == "ivf":
            # Train IVF index
            if not self.index.is_trained:
                self.index.train(embeddings)
        
        self.index.add(embeddings.astype(np.float32))
        self.vocab = vocab
    
    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query embedding vector (dimension,)
            k: Number of neighbors to return
            
        Returns:
            List of (token, distance) tuples, sorted by distance
        """
        # Normalize query
        query = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        # Search
        distances, indices = self.index.search(query, k)
        
        # Format results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.vocab):
                results.append((self.vocab[idx], float(dist)))
        
        return results
    
    def save(self, filepath: str):
        """Save index to file."""
        faiss.write_index(self.index, filepath)
        
        # Save vocabulary
        vocab_path = str(Path(filepath).with_suffix('.vocab'))
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for token in self.vocab:
                f.write(f"{token}\n")
    
    def load(self, filepath: str):
        """Load index from file."""
        self.index = faiss.read_index(filepath)
        
        # Load vocabulary
        vocab_path = str(Path(filepath).with_suffix('.vocab'))
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = [line.strip() for line in f if line.strip()]


def build_faiss_index(embeddings_dict: dict, output_path: str, index_type: str = "flat") -> EmbeddingIndex:
    """
    Build FAISS index from embeddings dictionary.
    
    Args:
        embeddings_dict: Dictionary mapping tokens to embeddings
        output_path: Path to save index
        index_type: Index type ("flat" or "ivf")
        
    Returns:
        EmbeddingIndex instance
    """
    if not embeddings_dict:
        raise ValueError("Empty embeddings dictionary")
    
    # Get dimension from first embedding
    first_token = next(iter(embeddings_dict.keys()))
    dimension = len(embeddings_dict[first_token])
    
    # Create index
    index = EmbeddingIndex(dimension, index_type)
    
    # Prepare data
    vocab = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[token] for token in vocab])
    
    # Add to index
    index.add_embeddings(embeddings, vocab)
    
    # Save
    index.save(output_path)
    
    return index


if __name__ == "__main__":
    # Test FAISS index
    if FAISS_AVAILABLE:
        # Create dummy embeddings
        embeddings = {
            'xin': np.random.randn(256),
            'chào': np.random.randn(256),
            'việt': np.random.randn(256),
            'nam': np.random.randn(256)
        }
        
        index = build_faiss_index(embeddings, "test_index.faiss")
        
        # Search
        query = np.random.randn(256)
        results = index.search(query, k=2)
        print(f"Search results: {results}")
    else:
        print("FAISS not available")

