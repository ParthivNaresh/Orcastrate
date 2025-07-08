"""
Anthropic client implementation for intelligent planning.

This module provides Anthropic Claude-specific integration for LLM-powered planning operations.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, cast

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from ...config.settings import get_settings
from .base import LLMClient, LLMConfig, LLMProvider, LLMResponse, PlanningPrompt


class AnthropicClient(LLMClient):
    """Anthropic Claude client for intelligent planning operations."""

    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            # Create config from centralized settings
            settings = get_settings()
            config = LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model=settings.llm.anthropic_default_model,
                api_key=settings.llm.anthropic_api_key or "",
                temperature=settings.llm.anthropic_temperature,
                max_tokens=settings.llm.anthropic_max_tokens,
                timeout=settings.llm.anthropic_timeout,
                retry_attempts=settings.llm.llm_retry_attempts,
                retry_delay=settings.llm.llm_retry_delay,
                rate_limit_requests_per_minute=settings.llm.llm_rate_limit_requests_per_minute,
            )
        super().__init__(config)
        self.logger = logging.getLogger(f"{__name__}.AnthropicClient")
        self._client: Optional[anthropic.AsyncAnthropic] = None

        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic package not available. Install with: pip install anthropic"
            )

        # Validate model
        valid_models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]
        if config.model not in valid_models:
            self.logger.warning(
                f"Model {config.model} not in validated list: {valid_models}"
            )

    async def initialize(self) -> None:
        """Initialize the Anthropic client."""
        try:
            # Initialize the Anthropic client
            self._client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key, timeout=self.config.timeout
            )

            # Test the connection with a simple request
            await self._test_connection()

            self.logger.info(
                f"Anthropic client initialized successfully with model {self.config.model}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic client: {e}")
            raise

    async def generate_text(self, prompt: PlanningPrompt) -> LLMResponse:
        """Generate text response from Anthropic Claude."""
        if self._client is None:
            raise RuntimeError("Anthropic client not initialized")

        await self._rate_limit_check()

        start_time = time.time()

        try:
            # Prepare the prompt for Anthropic format
            # Anthropic uses system parameter and messages format
            messages = []

            # Add examples as conversation history if provided
            if prompt.examples:
                for example in prompt.examples:
                    if "user" in example and "assistant" in example:
                        messages.append({"role": "user", "content": example["user"]})
                        messages.append(
                            {"role": "assistant", "content": example["assistant"]}
                        )

            # Add the main user prompt
            messages.append({"role": "user", "content": prompt.user_prompt})

            self.logger.debug(
                f"Sending request to Anthropic with {len(messages)} messages"
            )

            # Make the API call with retries
            response = await self._make_request_with_retry(
                prompt.system_prompt, messages
            )

            response_time = time.time() - start_time

            # Extract response content
            content = response.content[0].text if response.content else ""

            # Build usage information
            usage = {
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
                "total_tokens": (
                    response.usage.input_tokens + response.usage.output_tokens
                )
                if response.usage
                else 0,
            }

            llm_response = LLMResponse(
                content=content,
                usage=usage,
                model=response.model,
                provider="anthropic",
                response_time=response_time,
                finish_reason=response.stop_reason,
                metadata={
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
            )

            self.logger.info(
                f"Anthropic request completed in {response_time:.2f}s, tokens: {usage['total_tokens']}"
            )

            return llm_response

        except Exception as e:
            self.logger.error(f"Anthropic API request failed: {e}")
            raise

    async def _make_request_with_retry(
        self, system_prompt: str, messages: List[Dict[str, str]]
    ) -> Any:
        """Make Anthropic API request with retry logic."""
        last_exception: Optional[Exception] = None

        for attempt in range(self.config.retry_attempts):
            try:
                if self._client is None:
                    raise RuntimeError("Anthropic client not initialized")
                client_with_messages = cast(Any, self._client)
                response = await client_with_messages.messages.create(
                    model=self.config.model,
                    system=system_prompt,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                )
                return response

            except anthropic.RateLimitError as e:
                self.logger.warning(f"Rate limit hit on attempt {attempt + 1}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    wait_time = self.config.retry_delay * (
                        2**attempt
                    )  # Exponential backoff
                    await asyncio.sleep(wait_time)
                last_exception = e

            except anthropic.APITimeoutError as e:
                self.logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    wait_time = self.config.retry_delay * (2**attempt)
                    await asyncio.sleep(wait_time)
                last_exception = e

            except anthropic.APIError as e:
                self.logger.error(f"Anthropic API error on attempt {attempt + 1}: {e}")
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
        """Test the Anthropic connection with a simple request."""
        if self._client is None:
            raise RuntimeError("Anthropic client not initialized")

        try:
            test_messages = [
                {
                    "role": "user",
                    "content": "Say 'connection test successful' if you can read this.",
                }
            ]

            client_with_messages = cast(Any, self._client)
            response = await client_with_messages.messages.create(
                model=self.config.model,
                system="You are a helpful assistant.",
                messages=test_messages,
                max_tokens=10,
                temperature=0.0,
            )

            if response.content and response.content[0].text:
                self.logger.debug("Anthropic connection test successful")
            else:
                raise Exception("Empty response from Anthropic")

        except Exception as e:
            self.logger.error(f"Anthropic connection test failed: {e}")
            raise

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for the Anthropic client."""
        stats = super().get_usage_stats()
        stats.update(
            {
                "api_endpoint": "https://api.anthropic.com",
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
        )
        return stats
