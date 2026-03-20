from __future__ import annotations

from typing import List
import json
import logging

from google import genai
from google.genai import types

from configs.config import cfg

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generate 4-5 questions per chunk using Gemini."""

    SYSTEM_INSTRUCTION = """
    You are an assistant specialized in understanding the content and creating reasonable questions from that.

    OUTPUT FORMAT: Output ONLY a valid JSON array of questions. Each question must be separated by a comma.

    Example:
    ["question 1", "question 2", "question 3"]

    CRITICAL RULES:
    - Use valid JSON array syntax with commas between elements
    - Each question must be in double quotes
    - Separate every element of the list with commas
    - Generate questions whose answers are present in the provided content
    - Do not include code block markers or additional text
    - IMPORTANT: Do NOT use quotation marks or apostrophes inside questions - rephrase to avoid them
    - Instead of "What is the definition of 'staff'?" write "What is the definition of staff?"
    - Generate exactly 4 to 5 questions per chunk
    """

    def __init__(self):
        if not cfg.gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY is not configured.")
        self.client = genai.Client(api_key=cfg.gemini_api_key)
        self.model_name = cfg.gemini_question_model
        logger.info(f"QuestionGenerator initialised. Model: {self.model_name}")

    def generate_questions(self, chunk_text: str) -> List[str]:
        if not chunk_text or not chunk_text.strip():
            logger.warning("Empty chunk — skipping question generation")
            return []
        try:
            user_prompt = (
                "This is the document text:\n"
                f"<document>\n{chunk_text}\n</document>\n\n"
                "Generate 4 to 5 questions from this document."
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_INSTRUCTION,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            questions = json.loads(response.text.strip())
            if not isinstance(questions, list):
                return []
            return [q for q in questions if isinstance(q, str) and q.strip()]
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
        except Exception as e:
            logger.error(f"Question generation error: {e}")
        return []