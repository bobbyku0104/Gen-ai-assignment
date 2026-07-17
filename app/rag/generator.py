import time
from typing import List
import google.generativeai as genai
from langchain_core.documents import Document
from app.config import settings
from app.utils.logger import logger

class AnswerGenerator:
    """
    Interfaces with the Google Gemini API to generate grounded answers.
    Enforces strict context grounding instructions to prevent hallucination.
    """

    def __init__(self):
        """Initializes and configures the Google Gemini generative model."""
        try:
            logger.info("Initializing Google Gemini model client.")
            # Configure standard google-generativeai API key
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # Using gemini-1.5-flash as the default cost-efficient, low-latency model
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Google Gemini client configured successfully.")
        except Exception as e:
            logger.error(f"Failed to configure Google Gemini API client: {e}")
            raise RuntimeError(f"Gemini client setup failure: {str(e)}") from e

    def generate_answer(self, question: str, chunks: List[Document]) -> str:
        """
        Generates an answer based strictly on the retrieved document chunks.
        Bypasses LLM API calls (saving cost) if no chunks are provided.

        Args:
            question (str): User's query.
            chunks (List[Document]): Context documents from retriever.

        Returns:
            str: Grounded response answer from the LLM or a fallback statement.
        """
        # Cost-Saving Safeguard: Avoid API calls if no relevant context was retrieved
        if not chunks or len(chunks) == 0:
            logger.info("No context chunks provided. Bypassing LLM call to save API tokens.")
            return "I cannot find relevant context to answer this question."

        # Format context text blocks with citation metadata indexes
        context_str_list = []
        for idx, doc in enumerate(chunks):
            source = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page", "Unknown Page")
            context_str_list.append(
                f"--- Context Block {idx + 1} (Source: {source}, Page: {page}) ---\n"
                f"{doc.page_content}"
            )
        
        context_text = "\n\n".join(context_str_list)

        # Structure strict prompt forcing the model to rely solely on the context
        prompt = (
            "System Instruction:\n"
            "You are a strict, factual assistant. Answer the user's question using ONLY the provided Context.\n"
            "Guidelines:\n"
            "1. Rely ONLY on facts directly stated in the Context. Do not assume or extrapolate.\n"
            "2. If the Context is insufficient to answer the question, or if you cannot find the answer in the Context, "
            "respond exactly with: 'I cannot find relevant context to answer this question.'\n"
            "3. Do not mention outside information or your general knowledge base.\n\n"
            f"Context:\n{context_text}\n\n"
            f"User Question: {question}\n\n"
            "Response:"
        )

        logger.info("Sending prompt request to Google Gemini API.")
        start_time = time.perf_counter()
        
        try:
            response = self.model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0  # Set temperature to 0 for maximum deterministic grounded responses
                )
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"Gemini API generation completed in {latency_ms:.2f} ms.")
            
            answer_text = response.text.strip() if response.text else ""
            if not answer_text:
                return "I cannot find relevant context to answer this question."
            
            return answer_text
        except Exception as e:
            logger.error(f"Error calling Google Gemini API: {e}")
            raise RuntimeError(f"LLM generation call failed: {str(e)}") from e
