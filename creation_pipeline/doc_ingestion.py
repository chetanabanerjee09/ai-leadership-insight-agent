import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from configs.config import cfg, validate
from chunking.chunking_pipeline import ChunkingPipeline, ChunkRecord
from embedding import GeminiEmbedding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_pages_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extract text per page from a local PDF using PyMuPDF.
    Returns list of (page_number, page_text) tuples (1-indexed).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF is required: pip install pymupdf")

    pages = []
    with fitz.open(pdf_path) as doc:
        for i in range(len(doc)):
            text = doc[i].get_text("text")
            if text and text.strip():
                pages.append((i + 1, text))

    logger.info(f"Extracted {len(pages)} pages from {pdf_path}")
    return pages


def _detect_file_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    raise ValueError(
        f"Unsupported file type '{ext}' for: {path}. Only .pdf is supported."
    )


# ---------------------------------------------------------------------------
# Milvus connection
# ---------------------------------------------------------------------------

def _connect_milvus() -> None:
    connections.connect(alias="default", uri=cfg.milvus_uri, token=cfg.milvus_token)
    logger.info(f"Connected to Zilliz Cloud: {cfg.milvus_uri}")


# ---------------------------------------------------------------------------
# DocIngestion
# ---------------------------------------------------------------------------

class DocIngestion:
    """
    Ingests local PDF / TXT files into Milvus.

    Steps
    -----
    1. Auto-create collection if it does not exist
    2. Read file from local path
    3. Extract text (per page for PDF, whole file for TXT)
    4. ChunkingPipeline  →  List[ChunkRecord]  (chunk + generate questions)
    5. Expand rows       →  one row per (chunk × question)
    6. GeminiEmbedding   →  embed combined "question + chunk" text
    7. Insert into Milvus in batches
    """

    def __init__(
        self,
        collection_name: str = None,
        client_id: str = "",
        project_id: str = "",
    ):
        validate()   # raises early if GEMINI_API_KEY etc. are missing

        self.collection_name = collection_name or cfg.milvus_collection_name
        self.client_id       = client_id
        self.project_id      = project_id

        self.pipeline = ChunkingPipeline()
        self.embedder  = GeminiEmbedding()

        _connect_milvus()
        self._load_collection(self.collection_name)

    # ------------------------------------------------------------------
    # Collection creation
    # ------------------------------------------------------------------

    def create_collection(self, name: str) -> Collection:
        """
        Create the Milvus collection with the required schema and vector index.
        Safe to call multiple times — skips creation if collection already exists.

        Schema
        ------
        id            INT64        auto-increment primary key
        text          VARCHAR      chunk passage
        question      VARCHAR      one question per row
        dense_vector  FLOAT_VECTOR dim = cfg.gemini_embedding_dim (3072)
        client_id     VARCHAR
        project_id    VARCHAR
        page_number   INT64
        """
        if utility.has_collection(name):
            logger.info(f"Collection '{name}' already exists — skipping creation.")
            return Collection(name)

        fields = [
            FieldSchema(name="id",           dtype=DataType.INT64,       is_primary=True, auto_id=True),
            FieldSchema(name="text",         dtype=DataType.VARCHAR,      max_length=65535),
            FieldSchema(name="question",     dtype=DataType.VARCHAR,      max_length=65535),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=cfg.gemini_embedding_dim),
            FieldSchema(name="client_id",    dtype=DataType.VARCHAR,      max_length=256),
            FieldSchema(name="project_id",   dtype=DataType.VARCHAR,      max_length=256),
            FieldSchema(name="page_number",  dtype=DataType.INT64),
        ]

        schema     = CollectionSchema(fields=fields, description="RAG pipeline collection")
        collection = Collection(name=name, schema=schema)
        logger.info(f"Collection '{name}' created.")

        collection.create_index(
            field_name="dense_vector",
            index_params={
                "metric_type": "COSINE",
                "index_type":  "IVF_FLAT",
                "params":      {"nlist": 128},
            },
        )
        logger.info("Vector index created (COSINE / IVF_FLAT).")
        return collection

    # ------------------------------------------------------------------
    # Collection load + validate
    # ------------------------------------------------------------------

    def _load_collection(self, name: str) -> None:
        # Auto-create if it does not exist
        self.create_collection(name)

        self.collection = Collection(name)
        self.collection.load()

        schema_fields = {f.name for f in self.collection.schema.fields}
        required = {"text", "question", "dense_vector", "client_id", "project_id", "page_number"}
        missing  = required - schema_fields
        if missing:
            raise RuntimeError(f"Collection schema missing required fields: {missing}")

        for f in self.collection.schema.fields:
            if f.name == "dense_vector":
                col_dim = f.params.get("dim")
                if col_dim != self.embedder.embedding_dim:
                    raise RuntimeError(
                        f"Vector dimension mismatch: "
                        f"collection expects {col_dim}D, "
                        f"embedder produces {self.embedder.embedding_dim}D."
                    )
                break

        logger.info(f"Collection '{name}' loaded and validated.")

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert_documents(
        self,
        file_paths: List[str],
        client_id: str = None,
        project_id: str = None,
    ) -> Dict[str, Any]:
        """
        Process local files and insert records into Milvus.

        Parameters
        ----------
        file_paths : list of local file paths  e.g. ["data/policy.pdf"]
        client_id  : overrides the default set in __init__
        project_id : overrides the default set in __init__

        Returns
        -------
        {"status": "success", "rows_inserted": int, ...}
        {"status": "error",   "error": str, ...}
        """
        if not file_paths:
            return {"status": "error", "error": "No file paths provided."}

        ins_client  = client_id  or self.client_id
        ins_project = project_id or self.project_id

        try:
            # ---- STEP 1: read + extract text -----------------------------
            all_records: List[ChunkRecord] = []

            for path in file_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")

                file_type = _detect_file_type(path)
                logger.info(f"Processing {file_type.upper()}: {path}")

                if file_type == "pdf":
                    for page_num, page_text in _extract_pages_from_pdf(path):
                        all_records.extend(self.pipeline.process_document(
                            document_text=page_text,
                            source=path,
                            page_number=page_num,
                            extraction_method="pymupdf",
                        ))

            if not all_records:
                return {"status": "error", "error": "No records produced from the provided files."}

            logger.info(f"Total ChunkRecords: {len(all_records)}")

            # ---- STEP 2: expand — one row per (chunk × question) ---------
            expanded       = []
            texts_to_embed = []

            for rec in all_records:
                questions = rec.questions or [""]
                for q in questions:
                    combined = f"{q}\n\n{rec.text}" if q else rec.text
                    expanded.append({
                        "text":        rec.text,
                        "question":    q,
                        "page_number": rec.page_number,
                    })
                    texts_to_embed.append(combined)

            logger.info(f"{len(all_records)} records → {len(expanded)} rows after expansion")

            # ---- STEP 3: embed -------------------------------------------
            logger.info(f"Generating embeddings for {len(texts_to_embed)} rows...")
            vectors = self.embedder.embed_documents(texts_to_embed)
            logger.info("Embeddings done.")

            # ---- STEP 4: build Milvus entities ---------------------------
            entities = []
            for idx, row in enumerate(expanded):
                entities.append({
                    "text":         row["text"],
                    "question":     row["question"],
                    "dense_vector": vectors[idx],
                    "client_id":    ins_client,
                    "project_id":   ins_project,
                    "page_number":  row["page_number"],
                })

            # ---- STEP 5: insert in batches -------------------------------
            bs             = cfg.milvus_batch_size
            total_inserted = 0
            total_batches  = (len(entities) + bs - 1) // bs

            for batch_num, start in enumerate(range(0, len(entities), bs), 1):
                batch = entities[start: start + bs]
                logger.info(
                    f"Inserting batch {batch_num}/{total_batches} ({len(batch)} rows)..."
                )
                self.collection.insert(batch)
                self.collection.flush()
                total_inserted += len(batch)

            logger.info(f"Inserted {total_inserted} rows into '{self.collection_name}'.")

            return {
                "status":          "success",
                "rows_inserted":   total_inserted,
                "chunk_records":   len(all_records),
                "files_processed": len(file_paths),
                "collection":      self.collection_name,
                "client_id":       ins_client,
                "project_id":      ins_project,
            }

        except FileNotFoundError as e:
            return {"status": "error", "error": str(e), "error_type": "file_not_found"}
        except ValueError as e:
            return {"status": "error", "error": str(e), "error_type": "unsupported_file_type"}
        except EnvironmentError as e:
            return {"status": "error", "error": str(e), "error_type": "env_error"}
        except RuntimeError as e:
            return {"status": "error", "error": str(e), "error_type": "runtime_error"}
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_project_data(self, client_id: str = None, project_id: str = None) -> bool:
        """Delete all Milvus rows for a client + project combination."""
        qclient  = client_id  or self.client_id
        qproject = project_id or self.project_id
        try:
            self.collection.delete(
                expr=f'client_id == "{qclient}" && project_id == "{qproject}"'
            )
            logger.info(f"Deleted rows: client={qclient}, project={qproject}")
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False