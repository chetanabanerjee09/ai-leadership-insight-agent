from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import asyncio
import logging

from llm_chunking import Chunk, LLMChunking
from question_generator import QuestionGenerator

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    """
    One processed unit ready for embedding → Milvus.

    Fields
    ------
    text              : chunk passage
    questions         : list of 4-5 questions
    source            : filename
    chunk_index       : position within the source document
    page_number       : PDF page number or paragraph index (0 = unknown)
    extraction_method : how raw text was extracted
    """
    text: str
    questions: List[str]
    source: str
    chunk_index: int
    page_number: int = 0
    extraction_method: str = "pymupdf"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "questions": self.questions,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "extraction_method": self.extraction_method,
        }


class ChunkingPipeline:
    """
    Orchestrates LLMChunking → QuestionGenerator.
    Has NO knowledge of embeddings or Milvus.
    """

    def __init__(self):
        self.chunker = LLMChunking()
        self.question_generator = QuestionGenerator()
        logger.info("ChunkingPipeline initialised.")

    def process_document(
        self,
        document_text: str,
        source: str,
        page_number: int = 0,
        extraction_method: str = "pymupdf",
    ) -> List[ChunkRecord]:
        """Chunk one document and generate questions for every chunk."""
        logger.info(f"Processing: {source} (page {page_number})")

        chunks: List[Chunk] = self.chunker.chunk_document(
            document_text, source, extraction_method
        )
        if not chunks:
            logger.warning(f"No chunks for {source}")
            return []

        records = []
        for chunk in chunks:
            questions = self.question_generator.generate_questions(chunk.text)
            records.append(ChunkRecord(
                text=chunk.text,
                questions=questions,
                source=source,
                chunk_index=chunk.metadata.get("chunk_index", 0),
                page_number=page_number,
                extraction_method=extraction_method,
            ))

        logger.info(f"Done: {source} → {len(records)} ChunkRecords")
        return records

    async def process_documents_async(
        self,
        documents: List[Tuple[str, str, str, int]],
        # (page_or_para_id, document_text, source, page_number)
        extraction_method: str = "pymupdf",
    ) -> List[ChunkRecord]:
        """Concurrent chunking + sequential question generation."""
        chunker_input = [(pid, text, src) for pid, text, src, _ in documents]
        chunked_results = await self.chunker.chunk_multiple_documents_async(chunker_input)

        source_map = {pid: src for pid, _, src, _ in documents}
        page_map   = {pid: pg  for pid, _, _,  pg in documents}

        records = []
        for pid, chunk_texts in chunked_results:
            source      = source_map.get(pid, pid)
            page_number = page_map.get(pid, 0)
            for idx, chunk_text in enumerate(chunk_texts):
                if not (isinstance(chunk_text, str) and chunk_text.strip()):
                    continue
                questions = self.question_generator.generate_questions(chunk_text)
                records.append(ChunkRecord(
                    text=chunk_text.strip(),
                    questions=questions,
                    source=source,
                    chunk_index=idx,
                    page_number=page_number,
                    extraction_method=extraction_method,
                ))

        logger.info(f"Async pipeline complete: {len(records)} ChunkRecords")
        return records