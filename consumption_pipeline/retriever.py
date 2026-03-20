import logging
from typing import Any, Dict, List

from pymilvus import Collection, connections, utility

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs.config import cfg, validate

sys.path.append(os.path.join(os.path.dirname(__file__), "../creation_pipeline"))
from embedding import GeminiEmbedding

logger = logging.getLogger(__name__)


def _connect_milvus() -> None:
    connections.connect(alias="default", uri=cfg.milvus_uri, token=cfg.milvus_token)
    logger.info(f"Connected to Zilliz Cloud: {cfg.milvus_uri}")


class Retriever:
    """
    Embeds a query and fetches the top-k most similar chunks from Milvus.

    Returns
    -------
    List of dicts: {text, question, page_number, score}
    """

    def __init__(
        self,
        collection_name: str = None,
        client_id: str = "",
        project_id: str = "",
    ):
        validate()

        self.collection_name = collection_name or cfg.milvus_collection_name
        self.client_id       = client_id
        self.project_id      = project_id
        self.top_k           = cfg.retrieval_top_k

        self.embedder = GeminiEmbedding()

        _connect_milvus()

        if not utility.has_collection(self.collection_name):
            raise RuntimeError(
                f"Milvus collection '{self.collection_name}' does not exist. "
                "Run the ingestion pipeline first."
            )

        self.collection = Collection(self.collection_name)
        self.collection.load()
        logger.info(f"Retriever ready. Collection: '{self.collection_name}', top_k: {self.top_k}")

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _keep_unique_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate chunks that have identical text.
        Milvus expands one chunk into multiple rows (one per question),
        so the same text can appear several times in results.
        Preserves original order and keeps the first occurrence (highest score).
        """
        seen   = set()
        unique = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            unique.append(chunk)
        return unique

    def retrieve(
        self,
        query: str,
        client_id: str = None,
        project_id: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed the query and return the top-k matching chunks.

        Parameters
        ----------
        query      : user's question string
        client_id  : overrides default
        project_id : overrides default

        Returns
        -------
        List of dicts with keys: text, question, page_number, score
        """
        qclient  = client_id  or self.client_id
        qproject = project_id or self.project_id

        logger.info(f"Retrieving chunks for query: '{query[:80]}...'")

        query_vector = self.embedder.embed_query(query)

        results = self.collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {}},
            limit=self.top_k,
            expr=f'client_id == "{qclient}" && project_id == "{qproject}"',
            output_fields=["text", "question", "page_number"],
        )

        chunks = []
        for hit in results[0]:
            chunks.append({
                "text":        hit.entity.get("text", ""),
                "question":    hit.entity.get("question", ""),
                "page_number": hit.entity.get("page_number", 0),
                "score":       hit.distance,
            })

        unique_chunks = self._keep_unique_chunks(chunks)
        logger.info(
            f"Fetched {len(chunks)} rows → "
            f"{len(unique_chunks)} unique chunks after deduplication."
        )
        return unique_chunks