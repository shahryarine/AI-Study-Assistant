from typing import Dict, Optional

class PromptBuilder:
    """
    Constructs highly structured prompts for the LLM to ensure grounded, 
    context-aware responses while preventing hallucination.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an AI study assistant.\n\n"
        "Answer the user's question using ONLY the provided context.\n\n"
        "Rules:\n"
        "1. Use only information explicitly supported by the context.\n"
        "2. Do not invent or assume information that is not present.\n"
        "3. If the context lacks sufficient information to answer the question, "
        "clearly state that the available sources are insufficient.\n"
        "4. Provide a clear, concise, and structured answer.\n"
        "5. Whenever possible, cite the relevant source using the provided metadata."
    )

    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize with a default system prompt or override it for custom behaviors.
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def build(self, question: str, context: str) -> Dict[str, str]:
        """
        Generate the final system and user prompt dictionary for API consumption.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not context or not context.strip():
            raise ValueError("Context cannot be empty.")

        user_prompt = f"Context:\n\n{context}\n\nQuestion:\n\n{question.strip()}"

        return {
            "system": self.system_prompt,
            "user": user_prompt,
        }