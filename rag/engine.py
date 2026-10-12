import logging
from typing import Optional

from rag.embedder import Embedder
from rag.retriever import Retriever
from rag.prompts import PromptBuilder
from rag.generator import OpenRouterClient
from rag.utils import track_time

logger = logging.getLogger(__name__)

class AnswerEngine:
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
        logger.info(f"Processing user question...")

        with track_time("Retrieving relevant knowledge chunks", logger):
            retrieved_chunks = self.retriever.retrieve(query=question, top_k=5)
            context_text = self._build_context_string(retrieved_chunks)
        
        if not context_text:
            return "No relevant information found in your documents to answer this question."

        with track_time("Formatting prompts", logger):
            prompts = self.prompt_builder.build(question=question, context=context_text)

        with track_time("Generating answer via OpenRouter API", logger):
            try:
                return self.llm_client.generate(
                    system_prompt=prompts["system"],
                    user_prompt=prompts["user"],
                    is_json_mode=False
                )
            except Exception as error:
                raise RuntimeError(f"API Generation Error: {str(error)}")