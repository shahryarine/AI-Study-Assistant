import logging
from typing import Optional

from rag.embedder import Embedder
from rag.retriever import Retriever
from rag.prompts import PromptBuilder
from rag.generator import OpenRouterClient

logger = logging.getLogger(__name__)

class AnswerEngine:
    """
    The core orchestration engine for the Generation phase of RAG.
    Connects the Retriever, PromptBuilder, and OpenRouterClient safely.
    """

    def __init__(
        self,
        embedder: Embedder,
        retriever: Retriever,
        llm_client: OpenRouterClient,
        prompt_builder: Optional[PromptBuilder] = None
    ):
        self.embedder = embedder
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or PromptBuilder()

    def _build_context_string(self, retrieved_chunks: list) -> str:
        """
        Helper method to format retrieved chunks into a single readable context string.
        """
        context_parts = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            text = chunk.get("text", "").strip()
            meta = chunk.get("metadata", {})
            if text:
                context_parts.append(
                    f"[Source {index} | File: {meta.get('filename', 'Unknown')} | Page: {meta.get('pdf_page', '?')}]\n{text}"
                )
        return "\n\n".join(context_parts)

    def ask_question(self, question: str) -> str:
        """
        Executes the end-to-end question answering workflow.
        """
        logger.info(f"Processing question: {question}")

        # Step 1: Retrieval
        retrieved_chunks = self.retriever.retrieve(query=question, top_k=5)
        context_text = self._build_context_string(retrieved_chunks)
        
        if not context_text:
            return "No relevant information found in your documents to answer this question."

        # Step 2: Prompt Building
        prompts = self.prompt_builder.build(question=question, context=context_text)

        # Step 3: LLM Generation via OpenRouter API
        try:
            return self.llm_client.generate(
                system_prompt=prompts["system"],
                user_prompt=prompts["user"],
                is_json_mode=False
            )
        except Exception as error:
            logger.error(f"Generation failed: {str(error)}")
            return f"Error communicating with the AI model: {str(error)}"