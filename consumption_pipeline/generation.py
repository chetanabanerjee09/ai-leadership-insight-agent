import json
import logging
import re
from typing import Any, Dict, Optional

from google import genai
from google.genai import types

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs.config import cfg

logger = logging.getLogger(__name__)


class EnginePipeline:

    # ------------------------------------------------------------------
    # System instructions
    # ------------------------------------------------------------------

    ANSWER_SYSTEM_INSTRUCTION = """
You are good at understanding the context and given question.
You are also good at creating answers for a given question from the given context only.

Given the context and a question, you need to formulate the answer and you have to use only the given context.
Don't use external knowledge. Try to use as much information as possible from the given context.

OUTPUT FORMAT:
- Use plain english only and in human readable format.
- PARSE ANY HTML TAGS, TABLES, SPECIAL TOKENS INTO HUMAN READABLE FORMAT
- DONT EXPLICITLY GIVE THE CONTEXT OF THE ANSWER GENERATION. JUST OUTPUT ONLY ANSWER

NOTE:
- Understand the context meaning and question meaning semantically before generating the answer.
- If the context assigns different labels to benefits, entitlements, roles, or quantities, treat them as the same unless the context explicitly states they are different.
- Questions might be incomplete, so try to understand the context of the missing questions if possible.
- When a question asks for more than one thing, answer all parts in a single, coherent response.
- State only the value that directly answers the question.
- Don't hallucinate specifically with numerical values it should be from the context only.
"""

    PLOT_DECISION_SYSTEM_INSTRUCTION = """
You are a data analyst assistant. Given a question and retrieved context, decide if a plot/chart would help answer the question visually.
 
A plot is useful ONLY when:
- The question asks about trends, comparisons, growth, performance over time, rankings, or distributions.
- The context contains actual numerical data (numbers, percentages, values) that can be plotted.
 
A plot is NOT needed when:
- The question asks for definitions, explanations, policies, or qualitative information.
- The context does not contain enough numerical data points to plot.
 
CRITICAL RULES:
- Only extract values EXPLICITLY present in the context. Do NOT invent, estimate numbers.
- If you cannot find at least 2 data points in the context, set plot to false.
- NEVER mix values from different populations or bases (e.g. global % vs US % are different bases — do NOT combine them in one chart).
- Output ONLY valid JSON. No explanation, no markdown, no extra text.

Chart selection rules:
- Use a "bar" chart when comparing values across categories (e.g., departments, metrics, groups), especially when values come from different populations or do not sum to 100%.
- Use a "pie" chart only when all values represent parts of the SAME whole (e.g., distribution of a single population) and explicitly sum to 100% in the context.
- Use a "line" chart only when the data represents a trend over time (e.g., revenue across quarters or years).

Examples:
- Bar: comparing attrition rate vs engagement rate
- Pie: workforce distribution across regions (only if totals sum to 100%)
- Line: revenue growth across quarters
 
OUTPUT FORMAT:
{
  "plot": true or false,
  "plot_type": "line" or "bar" or "pie" or null,
  "title": "chart title" or null,
  "x_label": "x axis label" or null,
  "y_label": "y axis label" or null,
  "data": [{"x": "label", "y": numeric_value}, ...] or []
}
"""

    PLOT_DECISION_PROMPT = """Context:
{context}

Question: {question}

Decide if a plot is needed. Extract data ONLY from the context above. Do not hallucinate values."""

    ANSWER_PROMPT = """Context/Information:
{context}

Question: {question}

Answer the question using only the information provided above. Be concise and direct."""

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self):
        if not cfg.gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY is not configured.")
        self.client     = genai.Client(api_key=cfg.gemini_api_key)
        self.model_name = cfg.gemini_generation_model
        logger.info(f"EnginePipeline initialised. Model: {self.model_name}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise_answer(self, text: str) -> str:
        """Normalise raw LLM output: strip prefixes, markdown, and fix punctuation/capitalisation."""
        if not text:
            return ""

        text = text.strip()

        # Remove surrounding quotes
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        # Remove common unwanted prefixes
        unwanted_phrases = [
            "Answer:", "Response:", "Based on the information",
            "According to the document", "According to the context",
            "The context states", "Based on the context",
        ]
        for phrase in unwanted_phrases:
            if text.lower().startswith(phrase.lower()):
                text = text[len(phrase):].strip()
                if text.startswith(":"):
                    text = text[1:].strip()

        # Remove markdown formatting
        text = re.sub(r"[*#_`\[\]()]", "", text).strip()

        # Ensure proper ending punctuation
        if text and text[-1] not in ".!?":
            text += "."

        # Ensure proper capitalisation
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        return text

    @staticmethod
    def _validate_plot_data(data: Any) -> bool:
        """
        Validate that plot data is a non-empty list of dicts
        each containing 'x' and 'y' keys with a numeric 'y' value.
        Returns False if data is invalid — plot will be skipped.
        """
        if not data or not isinstance(data, list):
            return False
        if len(data) < 2:
            logger.warning("Plot data has fewer than 2 points — skipping plot.")
            return False
        for item in data:
            if not isinstance(item, dict):
                return False
            if "x" not in item or "y" not in item:
                return False
            try:
                float(item["y"])   # y must be numeric
            except (TypeError, ValueError):
                logger.warning(f"Non-numeric y value found: {item['y']} — skipping plot.")
                return False
        return True

    # ------------------------------------------------------------------
    # Public: plot decision
    # ------------------------------------------------------------------

    def decide_plot(self, question: str, context: str) -> Optional[Dict[str, Any]]:
        """
        Ask Gemini whether a plot is needed for this question.
        Extracts structured data from context if a plot is warranted.

        Returns
        -------
        dict  — validated plot spec ready for visualizer.py, e.g.:
                {
                    "plot": True,
                    "plot_type": "line",
                    "title": "Revenue Trend",
                    "x_label": "Quarter",
                    "y_label": "Revenue ($B)",
                    "data": [{"x": "Q1 2024", "y": 5.2}, ...]
                }
        None  — if no plot is needed or data is invalid / Gemini fails
        """
        prompt = self.PLOT_DECISION_PROMPT.format(
            context=context,
            question=question,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.PLOT_DECISION_SYSTEM_INSTRUCTION,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )

            decision = json.loads(response.text.strip())

            # Fallback: if Gemini says no plot or returns no data
            if not decision.get("plot"):
                logger.info("Gemini decided no plot is needed.")
                return None

            if not decision.get("data"):
                logger.warning("Gemini said plot=true but returned no data — skipping plot.")
                return None

            # Validate data integrity — guard against hallucinated numbers
            if not self._validate_plot_data(decision["data"]):
                logger.warning("Plot data failed validation — skipping plot.")
                return None

            logger.info(
                f"Plot decision: type={decision.get('plot_type')}, "
                f"points={len(decision['data'])}, title={decision.get('title')}"
            )
            return decision

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plot decision JSON: {e} — skipping plot.")
            return None
        except Exception as e:
            logger.error(f"Plot decision failed: {e} — skipping plot.")
            return None

    # ------------------------------------------------------------------
    # Public: answer generation
    # ------------------------------------------------------------------

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate a plain text answer from context using Gemini.

        Parameters
        ----------
        question : the user's question
        context  : retrieved chunk texts joined into one string

        Returns
        -------
        Normalised answer string.
        """
        user_prompt = self.ANSWER_PROMPT.format(
            context=context,
            question=question,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.ANSWER_SYSTEM_INSTRUCTION,
                    temperature=0.0,
                ),
            )
            raw_answer   = response.text.strip()
            clean_answer = self._normalise_answer(raw_answer)

            if not clean_answer or len(clean_answer.strip()) < 10:
                clean_answer = "I don't have enough information to answer that question."

            logger.info(f"Generated answer: {clean_answer[:120]}...")
            return clean_answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            return "An error occurred during answer generation."