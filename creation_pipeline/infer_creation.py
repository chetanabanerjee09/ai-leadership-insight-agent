import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent          # RAG/
_PIPELINE = Path(__file__).parent             # RAG/creation_pipeline/
_CHUNKING = _PIPELINE / "chunking"            # RAG/creation_pipeline/chunking/

for p in [str(_ROOT), str(_PIPELINE), str(_CHUNKING)]:
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infer_creation.py",
        description="Ingest PDF files into Milvus.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        metavar="FILE",
        help="One or more local PDF file paths. Example: --files data/policy.pdf",
    )
    parser.add_argument(
        "--client_id",
        type=str,
        required=True,
        help="Client identifier. Example: --client_id acme",
    )
    parser.add_argument(
        "--project_id",
        type=str,
        required=True,
        help="Project identifier. Example: --project_id proj_1",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Milvus collection name (default: value from config.yaml)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    from doc_ingestion import DocIngestion

    ingestor = DocIngestion(
        collection_name=args.collection,
        client_id=args.client_id,
        project_id=args.project_id,
    )

    result = ingestor.insert_documents(file_paths=args.files)

    print(json.dumps(result, indent=2))

    if result.get("status") != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()