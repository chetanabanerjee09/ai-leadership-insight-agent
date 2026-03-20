from __future__ import annotations

from typing import List
import logging

from google import genai

from configs.config import cfg

logger = logging.getLogger(__name__)


class GeminiEmbedding:
    """
    Gemini embedding wrapper using gemini-embedding-001.

    Usage
    -----
    embedder = GeminiEmbedding()
    vectors  = embedder.embed_documents(["text one", "text two"])
    vector   = embedder.embed_query("what is the meaning of life?")
    """

    def __init__(self):
        if not cfg.gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY is not configured.")
        self.client = genai.Client(api_key=cfg.gemini_api_key)
        self.model_name = cfg.gemini_embedding_model
        self.embedding_dim = cfg.gemini_embedding_dim
        self.batch_size = cfg.gemini_embedding_batch_size
        logger.info(f"GeminiEmbedding initialised. Model: {self.model_name}, Dim: {self.embedding_dim}")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
        )
        vectors = [emb.values for emb in result.embeddings]
        for idx, vec in enumerate(vectors):
            if len(vec) != self.embedding_dim:
                raise ValueError(
                    f"Dimension mismatch at index {idx}: "
                    f"expected {self.embedding_dim}, got {len(vec)}"
                )
        return vectors

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = None,
    ) -> List[List[float]]:
        """Embed a list of texts in batches. Returns vectors in input order."""
        if not texts:
            return []

        bs = batch_size or self.batch_size
        all_vectors: List[List[float]] = []
        total_batches = (len(texts) + bs - 1) // bs

        for batch_num, start in enumerate(range(0, len(texts), bs), 1):
            batch = texts[start: start + bs]
            logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)...")
            vectors = self._embed(batch)
            all_vectors.extend(vectors)

        logger.info(f"Embedded {len(all_vectors)} texts.")
        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        if not text or not text.strip():
            raise ValueError("Cannot embed an empty query.")
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
        )
        vector = result.embeddings[0].values
        if len(vector) != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: "
                f"expected {self.embedding_dim}, got {len(vector)}"
            )
        return vector