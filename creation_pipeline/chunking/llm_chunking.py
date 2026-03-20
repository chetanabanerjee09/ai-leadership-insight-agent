from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import asyncio
import json
import logging

from google import genai
from google.genai import types

from configs.config import cfg

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMChunking:
    """Chunk documents into semantically coherent passages using Gemini."""

    SYSTEM_INSTRUCTION = """
    You are an assistant specialized in understanding the content and break the content into individual and self-explanatory chunks.
    Chunk means - "A medium sized paragraph that contains part of text from the given content"

    OUTPUT FORMAT (Make SURE TO OUTPUT ONLY IN JSON FORMAT ONLY): [<Chunk1>, <Chunk2>, ....]
    Example output:
    ["Diversity and Inclusion Adobe for All is our vision to advance diversity, equity and inclusion across Adobe. We recognize that when people feel respected and included, they can be more creative, innovative and successful. As of November 29, 2024, women represented 35.4% of our global employees, and underrepresented minorities (“URMs,” defined as those who identify as Black/African American, Hispanic/Latinx, Native American, Pacific Islander and/or two or more races) represented 11.6% of our U.S. employees.", 
    "We have a three-pillar strategy to grow the diversity and inclusion of our workforce over time, on which we have continued to drive progress during fiscal 2024: • Workforce: We take action to improve the hiring, retention and promotion of a more diverse workforce that reflects Adobe’s global footprint. We invest in partnerships and events to grow our pipeline and engage candidates across underrepresented communities. We aim to give individuals from nontraditional backgrounds new skills and opportunities to enter technology and design careers through Adobe Digital Academy, in partnership with our educational partners.",
    "HUMAN CAPITAL Our culture is built on the foundation that our people and the way we treat one another promote creativity, innovation and performance, which spur our success. We are continually investing in our global workforce to provide fair and market-competitive pay and benefits to support our employees’ wellbeing, foster their growth and development, and to further drive diversity and inclusion. As of November 29, 2024, we employed 30,709 people, of which 50% were in the United States and 50% were in our international locations. During fiscal 2024, our total attrition rate was 7.8%. We have not experienced work stoppages and believe our employee relations are good. Understanding employee sentiment and listening to employee feedback is important to Adobe. We utilize a variety of feedback mechanisms throughout an employee lifecycle to gather insights that help inform our decision-making regarding employee programs, talent risks, management opportunities, employee networks and more. In fiscal 2024, 81% of our employees participated in our most recent engagement survey."]

    NOTE: MAKE SURE TO COVER THE ENTIRE TEXT FROM THE CONTENT ACROSS THE CHUNKS
    NOTE: DONT OUTPUT ```json```. JUST OUTPUT ONLY LIST OF CHUNKS
    NOTE: DONT LOOSE INFORMATION THAT IS IN THE CONTEXT
    NOTE: REMOVE IRRELEVANT CHARACTERS AND EMOJIs LIKE "●" or "🔒" IN CHUNK
    NOTE: IGNORE UNNECESSARY ICONS OR LOGOS OR TAGS OR ANY SPECIAL SYMBOL IN THE CONTEXT.
    NOTE: IGNORE UNNECESSARY TAGS LIKE "\\n" IN CHUNK
    NOTE: KEEP EVERY CHUNK SIZE BETWEEN 400 TO 700 TOKENS BUT EVERY CHUNK SHOULD CONTAIN SENTENCES WHICH ARE SEMANTICALLY SIMILAR ALSO KEEP ONE TABLE IN ONE CHUNK.
    """

    def __init__(self):
        if not cfg.gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY is not configured.")
        self.client = genai.Client(api_key=cfg.gemini_api_key)
        self.model_name = cfg.gemini_chunking_model
        self.max_concurrent_requests = cfg.gemini_max_concurrent_requests
        logger.info(f"LLMChunking initialised. Model: {self.model_name}")

    def chunk_with_gemini(self, document_text: str) -> List[str]:
        user_prompt = f"This is the document text:\n<document>\n{document_text}\n</document>"
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.SYSTEM_INSTRUCTION,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text.strip())

    async def chunk_with_gemini_async(
        self, document_text: str, page_or_para_id: str
    ) -> Tuple[str, List[str]]:
        try:
            loop = asyncio.get_running_loop()
            chunks = await loop.run_in_executor(None, self.chunk_with_gemini, document_text)
            return (page_or_para_id, chunks)
        except Exception as e:
            logger.error(f"Error chunking {page_or_para_id}: {e}")
            return (page_or_para_id, [])

    async def chunk_multiple_documents_async(
        self,
        documents: List[Tuple[str, str, str]],
    ) -> List[Tuple[str, List[str]]]:
        if not documents:
            return []
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def _run(pid: str, text: str) -> Tuple[str, List[str]]:
            async with semaphore:
                return await self.chunk_with_gemini_async(text, pid)

        tasks = [_run(pid, text) for pid, text, _ in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed {documents[i][0]}: {result}")
                output.append((documents[i][0], []))
            else:
                output.append(result)
        return output

    def chunk_document(
        self,
        document_text: str,
        source: str,
        extraction_method: str = "pymupdf",
    ) -> List[Chunk]:
        if not document_text or not document_text.strip():
            logger.warning(f"Empty document: {source}")
            return []
        try:
            chunk_texts = self.chunk_with_gemini(document_text)
            chunks = []
            for idx, chunk_text in enumerate(chunk_texts):
                if isinstance(chunk_text, str) and chunk_text.strip():
                    chunks.append(Chunk(
                        text=chunk_text.strip(),
                        metadata={
                            "type": "llm_chunk",
                            "source": source,
                            "chunk_index": idx,
                            "chunking_method": "gemini",
                            "extraction_method": extraction_method,
                        },
                    ))
            logger.info(f"Created {len(chunks)} chunks for {source}")
            return chunks
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for {source}: {e}")
        except Exception as e:
            logger.error(f"Chunking error for {source}: {e}")

        return [Chunk(
            text=document_text,
            metadata={
                "type": "fallback_chunk",
                "source": source,
                "chunk_index": 0,
                "chunking_method": "gemini_error",
                "extraction_method": extraction_method,
            },
        )]