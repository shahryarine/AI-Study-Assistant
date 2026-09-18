import json
from json import JSONDecodeError
from typing import Any, Dict, Optional

class PromptBuilder:
    """
    Builds prompts for the study assistant LLM.
    Supports grounded question answering, multiple-choice exam generation, 
    and structured JSON response parsing.
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

    EXAM_SYSTEM_PROMPT = """
You are an AI study assistant that generates multiple-choice exam questions.

Generate questions using only the provided context.

Rules:
1. Use only information supported by the context.
2. Do not invent facts that are not present in the context.
3. Each question must have exactly four options.
4. Exactly one option must be correct.
5. Include a concise explanation for the correct answer.
6. Include the source page when it is available in the context.
7. Return only valid JSON.
8. The JSON must contain a top-level "questions" array.
9. Each question object must contain:
   - "question"
   - "options"
   - "correct_answer"
   - "explanation"
   - "source_page"
""".strip()

    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize with a default system prompt or override it for custom behaviors.
        """
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT
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

    def build_exam(
        self,
        context: str,
        num_questions: int = 5,
    ) -> dict[str, str]:
        """
        Build prompts for multiple-choice exam generation.
        """

        if not context or not context.strip():
            raise ValueError("Context cannot be empty.")

        if num_questions <= 0:
            raise ValueError("num_questions must be greater than zero.")

        user_prompt = f"""
Context:

{context}

Generate exactly {num_questions} multiple-choice questions.

Return a JSON object with this structure:

{{
  "questions": [
    {{
      "question": "Question text",
      "options": [
        "Option 1",
        "Option 2",
        "Option 3",
        "Option 4"
      ],
      "correct_answer": "The exact correct option text",
      "explanation": "Why the answer is correct",
      "source_page": null
    }}
  ]
}}
""".strip()

        return {
            "system": self.EXAM_SYSTEM_PROMPT,
            "user": user_prompt,
        }

    @staticmethod
    def parse_exam_response(response: str) -> dict[str, Any]:
        """
        Parse and validate an exam response returned by the LLM.

        Raises:
            ValueError: If the response is not valid JSON or does not
                        match the expected exam structure.
        """

        if not response or not response.strip():
            raise ValueError("Exam response cannot be empty.")

        try:
            data = json.loads(response)
        except JSONDecodeError as exc:
            raise ValueError(
                "Exam response is not valid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise ValueError("Exam response must be a JSON object.")

        questions = data.get("questions")

        if not isinstance(questions, list):
            raise ValueError(
                'Exam response must contain a "questions" array.'
            )

        for index, question in enumerate(questions, start=1):
            if not isinstance(question, dict):
                raise ValueError(
                    f"Question {index} must be a JSON object."
                )

            required_fields = (
                "question",
                "options",
                "correct_answer",
                "explanation",
                "source_page",
            )

            for field in required_fields:
                if field not in question:
                    raise ValueError(
                        f'Question {index} is missing "{field}".'
                    )

            options = question["options"]

            if not isinstance(options, list):
                raise ValueError(
                    f"Question {index} options must be an array."
                )

            if len(options) != 4:
                raise ValueError(
                    f"Question {index} must have exactly four options."
                )

            if question["correct_answer"] not in options:
                raise ValueError(
                    f"Question {index} correct_answer must match one "
                    "of the options."
                )

        return data