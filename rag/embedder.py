import torch
from typing import List
from sentence_transformers import SentenceTransformer

class Embedder:
    """
    A robust, memory-efficient wrapper for SentenceTransformer embedding models.
    Handles device allocation (CPU/GPU) and batch processing to prevent OOM errors.
    """

    DEFAULT_MODEL = "BAAI/bge-m3"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initializes the embedding model and automatically assigns it to the most 
        efficient available hardware accelerator (CUDA, MPS, or CPU).
        """
        self.model_name = model_name
        self.device = self._detect_device()
        
        # Load the model directly to the optimal device
        self.model = SentenceTransformer(model_name, device=self.device)

    def _detect_device(self) -> str:
        """
        Dynamically determine the best available hardware architecture.
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps" # For Apple Silicon Macs
        return "cpu"

    def embed_text(self, text: str) -> List[float]:
        """
        Generate a normalized embedding vector for a single string.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or solely whitespace.")

        embedding = self.model.encode(
            text.strip(),
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embedding.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using batch processing to 
        manage memory effectively during large document ingestion.
        """
        if not texts:
            return []

        # Filter out empty strings safely instead of crashing the entire process
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        
        if not valid_texts:
            raise ValueError("The provided list contains no valid text to embed.")

        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True # Helpful when processing hundreds of chunks
        )

        return embeddings.tolist()

if __name__ == "__main__":
    # Test initialization and device allocation
    emb = Embedder()
    print(f"Model loaded on: {emb.device}")
    
    sample_vectors = emb.embed_texts(["Machine learning", "RAG systems"])
    print(f"Generated {len(sample_vectors)} vectors. Dimension: {len(sample_vectors[0])}")