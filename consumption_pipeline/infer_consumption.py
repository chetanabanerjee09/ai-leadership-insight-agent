import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infer.py",
        description="Ask a question and get an answer (+ optional plot) from ingested documents.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help='Question to answer. Example: --question "What was the revenue in 2024?"',
    )
    parser.add_argument(
        "--client_id",
        type=str,
        required=True,
        help="Client identifier used during ingestion.",
    )
    parser.add_argument(
        "--project_id",
        type=str,
        required=True,
        help="Project identifier used during ingestion.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Milvus collection name (default: value from config.yaml).",
    )
    parser.add_argument(
        "--save_context",
        action="store_true",
        default=False,
        help="Save retrieved context to retrieved_context.txt for inspection.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="plots",
        help="Directory to save generated plots (default: plots/).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    from retriever   import Retriever
    from generation  import EnginePipeline
    from visualizer  import generate_plot

    # ---- Step 1: retrieve unique chunks ----------------------------------
    retriever = Retriever(
        collection_name=args.collection,
        client_id=args.client_id,
        project_id=args.project_id,
    )

    chunks = retriever.retrieve(
        query=args.question,
        client_id=args.client_id,
        project_id=args.project_id,
    )

    if not chunks:
        print("\nNo relevant chunks found. Try ingesting documents first.")
        sys.exit(1)

    # ---- Step 2: build context from text only ----------------------------
    # Questions stored in Milvus are NOT passed to LLM —
    # they were only used to improve embedding alignment during ingestion.
    context = "\n\n".join(
        f"[Page {c['page_number']}] {c['text']}" for c in chunks
    )
    logger.info(f"Built context from {len(chunks)} unique chunks.")

    # Optional: save context for debugging
    if args.save_context:
        context_path = Path(__file__).resolve().parent.parent / "retrieved_context.txt"
        with open(context_path, "w", encoding="utf-8") as f:
            for i, c in enumerate(chunks, 1):
                f.write(f"--- Chunk {i} | Page {c['page_number']} | Score {c['score']:.4f} ---\n")
                f.write(c["text"].strip() + "\n\n")
        logger.info(f"Saved retrieved context to {context_path}")

    engine     = EnginePipeline()
    plot_path  = None

    # ---- Step 3: ask Gemini if a plot is needed --------------------------
    # decide_plot() returns validated spec dict or None
    # Numbers are grounded in context — prompt enforces no hallucination
    # _validate_plot_data() guards against malformed / hallucinated values
    decision = engine.decide_plot(
        question=args.question,
        context=context,
    )

    # ---- Step 4: generate plot if decision says yes ----------------------
    if decision:
        # Resolve plot_dir relative to project root so it works from any cwd
        plot_dir = str(Path(__file__).resolve().parent.parent / args.plot_dir)
        plot_path = generate_plot(decision, output_dir=plot_dir)
        if plot_path:
            logger.info(f"Plot saved to: {plot_path}")
        else:
            logger.warning("Plot generation failed — continuing with text answer only.")
    else:
        logger.info("No plot needed for this question.")

    # ---- Step 5: generate text answer ------------------------------------
    answer = engine.generate_answer(
        question=args.question,
        context=context,
    )

    # ---- Step 6: print result --------------------------------------------
    print("\n" + "=" * 60)
    print(f"Question : {args.question}")
    print("=" * 60)
    print(f"Answer   : {answer}")
    if plot_path:
        print(f"Plot     : {plot_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()