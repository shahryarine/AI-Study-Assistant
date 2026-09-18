import os
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()


class OpenRouterClient:
    """
    Client for communicating with OpenRouter's Chat Completions API.
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 45,
    ):
        self.model = model or os.getenv(
            "OPENROUTER_MODEL",
            "openrouter/free",
        )
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        if not self.model:
            raise ValueError("Model name cannot be empty.")

        if self.timeout <= 0:
            raise ValueError("Timeout must be greater than zero.")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        is_json_mode: bool = False,
    ) -> str:
        """
        Send a prompt to OpenRouter and return the generated text.

        When is_json_mode is True, request a JSON object response.
        """

        if not system_prompt or not system_prompt.strip():
            raise ValueError("System prompt cannot be empty.")

        if not user_prompt or not user_prompt.strip():
            raise ValueError("User prompt cannot be empty.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0.0,
        }

        if is_json_mode:
            payload["response_format"] = {
                "type": "json_object",
            }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code == 429:
                raise RuntimeError(
                    "OpenRouter rate limit exceeded."
                )

            if response.status_code >= 500:
                raise RuntimeError(
                    f"OpenRouter server error: {response.status_code}"
                )

            response.raise_for_status()

            data = response.json()

            choices = data.get("choices", [])

            if not choices:
                raise RuntimeError(
                    "OpenRouter response does not contain choices."
                )

            content = choices[0].get("message", {}).get("content")

            if not content:
                raise RuntimeError(
                    "OpenRouter response does not contain generated content."
                )

            return content.strip()

        except requests.Timeout as exc:
            raise RuntimeError(
                "OpenRouter request timed out."
            ) from exc

        except requests.RequestException as exc:
            raise RuntimeError(
                f"OpenRouter request failed: {exc}"
            ) from exc