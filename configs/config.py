import os
from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_yaml() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {_CONFIG_PATH}. "
            "Make sure config.yaml is in the configs/ directory."
        )
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class Config:
    # --- Secrets: from environment variables only -------------------------
    gemini_api_key: str   # env: GEMINI_API_KEY
    milvus_uri: str       # env: MILVUS_URI
    milvus_token: str     # env: MILVUS_TOKEN

    # --- Non-sensitive: from config.yaml ----------------------------------
    # Gemini models
    gemini_chunking_model: str
    gemini_question_model: str
    gemini_embedding_model: str
    gemini_generation_model: str      # used by generation.py
    gemini_embedding_dim: int
    gemini_max_concurrent_requests: int
    gemini_embedding_batch_size: int

    # Milvus
    milvus_collection_name: str
    milvus_batch_size: int

    # Retrieval
    retrieval_top_k: int


def _build_config() -> Config:
    raw = _load_yaml()

    gemini = raw.get("gemini", {})
    milvus = raw.get("milvus", {})
    ret    = raw.get("retrieval", {})

    return Config(
        # Secrets — os.environ only, no fallback to yaml
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        milvus_uri=os.environ.get("MILVUS_URI", ""),
        milvus_token=os.environ.get("MILVUS_TOKEN", ""),

        # Gemini models
        gemini_chunking_model=gemini.get("chunking_model", "gemini-2.0-flash-001"),
        gemini_question_model=gemini.get("question_model", "gemini-2.0-flash-001"),
        gemini_embedding_model=gemini.get("embedding_model", "gemini-embedding-001"),
        gemini_generation_model=gemini.get("generation_model", "gemini-2.0-flash-001"),
        gemini_embedding_dim=int(gemini.get("embedding_dim", 3072)),
        gemini_max_concurrent_requests=int(gemini.get("max_concurrent_requests", 30)),
        gemini_embedding_batch_size=int(gemini.get("embedding_batch_size", 50)),

        # Milvus
        milvus_collection_name=milvus.get("collection_name", "rag_collection"),
        milvus_batch_size=int(milvus.get("batch_size", 500)),

        # Retrieval
        retrieval_top_k=int(ret.get("top_k", 5)),
    )


# Single importable instance
cfg = _build_config()


def validate() -> None:
    """
    Call at startup to catch missing env vars early.
    Raises EnvironmentError listing every missing required secret.
    """
    missing = []

    if not cfg.gemini_api_key:
        missing.append("GEMINI_API_KEY")

    if not cfg.milvus_uri:
        missing.append("MILVUS_URI")

    if not cfg.milvus_token:
        missing.append("MILVUS_TOKEN")

    if missing:
        raise EnvironmentError(
            "Missing required environment variables:\n"
            + "\n".join(f"  export {m}" for m in missing)
        )