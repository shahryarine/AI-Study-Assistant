class PromptBuilder:
    """
    Builds prompts for the study assistant LLM.

    The LLM is instructed to answer only from the retrieved context.
    """

    SYSTEM_PROMPT = """
You are an AI study assistant.

Answer the user's question using only the provided context.

Rules:
1. Use only information supported by the context.
2. Do not invent or assume information that is not present in the context.
3. If the context does not contain enough information to answer the question,
   clearly say that the available sources do not provide enough information.
4. Give a clear and concise answer.
5. When possible, cite the relevant source using the provided file name and page.
""".strip()

    def build(
        self,
        question: str,
        context: str,
    ) -> dict[str, str]:
        """
        Build a system prompt and user prompt for the LLM.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not context or not context.strip():
            raise ValueError("Context cannot be empty.")

        user_prompt = f"""
Context:

{context}

Question:

{question.strip()}
""".strip()

        return {
            "system": self.SYSTEM_PROMPT,
            "user": user_prompt,
        }