"""
OpenAI client implementation for intelligent planning.

This module provides OpenAI-specific integration for LLM-powered planning operations.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ...config.settings import get_settings
from .base import LLMClient, LLMConfig, LLMProvider, LLMResponse, PlanningPrompt


class OpenAIClient(LLMClient):
    """OpenAI client for intelligent planning operations."""

    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            # Create config from centralized settings
            settings = get_settings()
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model=settings.llm.openai_default_model,
                api_key=settings.llm.openai_api_key or "",
                temperature=settings.llm.openai_temperature,
                max_tokens=settings.llm.openai_max_tokens,
                timeout=settings.llm.openai_timeout,
                retry_attempts=settings.llm.llm_retry_attempts,
                retry_delay=settings.llm.llm_retry_delay,
                rate_limit_requests_per_minute=settings.llm.llm_rate_limit_requests_per_minute,
            )
        super().__init__(config)
        self.logger = logging.getLogger(f"{__name__}.OpenAIClient")
        if not OPENAI_AVAILABLE:
            self._client = None
        else:
            self._client: Optional[openai.AsyncOpenAI] = None
        self._last_request_time = 0.0
        self._request_count = 0

        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not available. Install with: pip install openai"
            )

        # Validate model
        valid_models = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]
        if config.model not in valid_models:
            self.logger.warning(
                f"Model {config.model} not in validated list: {valid_models}"
            )

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        try:
            # Initialize the OpenAI client
            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key, timeout=self.config.timeout
            )

            # Test the connection with a simple request
            await self._test_connection()

            self.logger.info(
                f"OpenAI client initialized successfully with model {self.config.model}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    async def generate_text(self, prompt: PlanningPrompt) -> LLMResponse:
        """Generate text response from OpenAI."""
        if self._client is None:
            raise RuntimeError("OpenAI client not initialized")

        await self._rate_limit_check()

        start_time = time.time()

        try:
            # Prepare messages for OpenAI chat format
            messages: List[Any] = [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ]

            # Add examples as conversation history if provided
            if prompt.examples:
                for example in prompt.examples:
                    if "user" in example and "assistant" in example:
                        messages.insert(
                            -1, {"role": "user", "content": example["user"]}
                        )
                        messages.insert(
                            -1, {"role": "assistant", "content": example["assistant"]}
                        )

            self.logger.debug(
                f"Sending request to OpenAI with {len(messages)} messages"
            )

            # Make the API call with retries
            response = await self._make_request_with_retry(messages)

            response_time = time.time() - start_time

            # Extract response content
            content = response.choices[0].message.content

            # Build usage information
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens
                if response.usage
                else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            llm_response = LLMResponse(
                content=content,
                usage=usage,
                model=response.model,
                provider="openai",
                response_time=response_time,
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
            )

            self.logger.info(
                f"OpenAI request completed in {response_time:.2f}s, tokens: {usage['total_tokens']}"
            )

            return llm_response

        except Exception as e:
            self.logger.error(f"OpenAI API request failed: {e}")
            raise

    async def _make_request_with_retry(self, messages: List[Any]) -> Any:
        """Make OpenAI API request with retry logic."""
        if self._client is None:
            raise RuntimeError("OpenAI client not initialized")

        last_exception: Optional[Exception] = None

        for attempt in range(self.config.retry_attempts):
            try:
                response = await self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                )
                return response

            except openai.RateLimitError as e:
                self.logger.warning(f"Rate limit hit on attempt {attempt + 1}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    wait_time = self.config.retry_delay * (
                        2**attempt
                    )  # Exponential backoff
                    await asyncio.sleep(wait_time)
                last_exception = e

            except openai.APITimeoutError as e:
                self.logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    wait_time = self.config.retry_delay * (2**attempt)
                    await asyncio.sleep(wait_time)
                last_exception = e

            except openai.APIError as e:
                self.logger.error(f"OpenAI API error on attempt {attempt + 1}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    wait_time = self.config.retry_delay
                    await asyncio.sleep(wait_time)
                last_exception = e

            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                last_exception = e
                break  # Don't retry for unexpected errors

        # If we get here, all attempts failed
        raise last_exception or Exception("All retry attempts failed")

    async def _test_connection(self) -> None:
        """Test the OpenAI connection with a simple request."""
        if self._client is None:
            raise RuntimeError("OpenAI client not initialized")

        try:
            test_messages: List[Any] = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Say 'connection test successful' if you can read this.",
                },
            ]

            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=test_messages,
                max_tokens=10,
                temperature=0.0,
            )

            if response.choices[0].message.content:
                self.logger.debug("OpenAI connection test successful")
            else:
                raise Exception("Empty response from OpenAI")

        except Exception as e:
            self.logger.error(f"OpenAI connection test failed: {e}")
            raise

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for the OpenAI client."""
        stats = super().get_usage_stats()
        stats.update(
            {
                "api_endpoint": "https://api.openai.com",
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
        )
        return stats
