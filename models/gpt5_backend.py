"""
GPT-5 specific backend for handling the responses API.

This backend extends APIBackend specifically for GPT-5 models that use
the new responses.create() API instead of chat.completions.create().
"""

import os
from typing import Optional
from .base import ModelBackend, ForwardResult, GenerationResult


class GPT5Backend(ModelBackend):
    """
    Backend specifically for GPT-5 models using the responses API.

    Note: This backend expects the OpenAI client to support the responses API.
    If you're getting errors about missing 'responses' attribute, you may need
    to update the openai library or use a client that supports GPT-5.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize GPT-5 backend.

        Args:
            model_name: Model identifier (e.g., "gpt-5", "gpt-5-mini")
            api_key: API key for authentication
            base_url: Custom API endpoint URL
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "GPT-5 backend requires the openai library. "
                "Install with: pip install openai>=1.0.0"
            )

        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url

        if not self.api_key:
            raise ValueError(
                "API key is required. "
                "Provide it via api_key parameter or OPENAI_API_KEY environment variable."
            )

        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = AsyncOpenAI(**client_kwargs)

        # Verify the client supports responses API
        if not hasattr(self.client, 'responses'):
            print(
                f"Warning: OpenAI client does not have 'responses' attribute. "
                f"GPT-5 API calls may fail. Client type: {type(self.client)}"
            )

    async def forward_async(
        self,
        prompt: str,
        max_length: int = 2048,
        **kwargs
    ) -> ForwardResult:
        """
        Forward pass is not supported for API backends.

        Raises:
            NotImplementedError: Always raised
        """
        raise NotImplementedError(
            "Forward pass is not supported by GPT-5 backend. "
            "API endpoints do not provide access to model logits."
        )

    async def generate_async(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text using GPT-5 responses API.

        Args:
            prompt: The input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional API-specific parameters

        Returns:
            GenerationResult containing generated text
        """
        # Prepare API parameters
        api_params = {
            "input": [{"role": "user", "content": prompt}],
            "model": self.model_name,
            "store": False,
        }

        # Add any additional kwargs
        api_params.update(kwargs)

        try:
            # Use responses API
            response = await self.client.responses.create(**api_params)

            # Extract generated text
            generated_text = response.output_text.strip()

            # Return result
            result = GenerationResult(
                generated_text=generated_text,
                generated_ids=[],  # Not available from API
                full_text=prompt + "\n" + generated_text,
                full_ids=[],  # Not available from API
                logits=None  # Not available from API
            )

            return result

        except AttributeError as e:
            raise TypeError(
                f"GPT-5 responses API not available. "
                f"Client type: {type(self.client)}. "
                f"Make sure you have the correct OpenAI library version for GPT-5. "
                f"Original error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"GPT-5 API call failed: {str(e)}") from e

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> GenerationResult:
        """Synchronous version of generate_async."""
        import asyncio
        return asyncio.run(
            self.generate_async(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs
            )
        )

    def get_tokenizer(self):
        """
        Get tokenizer is not supported for API backends.

        Raises:
            NotImplementedError: Always raised
        """
        raise NotImplementedError(
            "GPT5Backend does not provide direct tokenizer access. "
            "Tokenization is handled internally by the API service."
        )

    async def shutdown(self):
        """Cleanup resources and shutdown the backend."""
        await self.client.close()
