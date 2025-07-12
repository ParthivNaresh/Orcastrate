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
        self._messages: Optional[Any] = None
        self._initialized: bool = False

        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError(
                "Anthropic library not available. Install with: pip install anthropic"
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

            # Set up messages interface
            self._messages = self._client.messages

            # Test the connection with a simple request
            await self._test_connection()

            self._initialized = True
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
                    (response.usage.input_tokens + response.usage.output_tokens)
                    if response.usage
                    else 0
                ),
            }

            # Handle finish_reason with proper string conversion
            finish_reason = "completed"
            if hasattr(response, "stop_reason"):
                if response.stop_reason is not None:
                    finish_reason = str(response.stop_reason)

            llm_response = LLMResponse(
                content=content,
                usage=usage,
                model=str(response.model),
                provider="anthropic",
                response_time=response_time,
                finish_reason=finish_reason,
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
                if self._client is None or self._messages is None:
                    raise RuntimeError("Anthropic client not initialized")
                response = await self._messages.create(
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
                # Retry on rate limit errors even if they're generic exceptions
                if "rate limit" in str(e).lower():
                    if attempt < self.config.retry_attempts - 1:
                        wait_time = self.config.retry_delay * (2**attempt)
                        await asyncio.sleep(wait_time)
                    last_exception = e
                else:
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

            if self._messages is None:
                raise RuntimeError("Anthropic messages interface not available")
            response = await self._messages.create(
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

    async def _validate_api_key(self) -> bool:
        """Validate the Anthropic API key."""
        try:
            if self._client is None:
                await self.initialize()

            # Simple test request to validate API key
            if self._messages is None:
                raise RuntimeError("Anthropic messages interface not available")
            await self._messages.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                temperature=0.0,
            )
            return True
        except Exception as e:
            self.logger.error(f"API key validation failed: {e}")
            return False

    async def _make_requirements_analysis_request(self, requirements: str) -> str:
        """Make a request for requirements analysis."""
        prompt = f"Analyze these requirements: {requirements}"
        response = await self.generate_text(
            PlanningPrompt(
                system_prompt="You are an expert requirements analyst.",
                user_prompt=prompt,
                context={},
            )
        )
        return response.content

    async def _make_plan_generation_request(self, request_data: Dict[str, Any]) -> str:
        """Make a request for plan generation."""
        prompt = f"Generate a plan for: {request_data}"
        response = await self.generate_text(
            PlanningPrompt(
                system_prompt="You are an expert plan generator.",
                user_prompt=prompt,
                context={},
            )
        )
        return response.content

    async def _make_plan_optimization_request(
        self, request_data: Dict[str, Any]
    ) -> str:
        """Make a request for plan optimization."""
        prompt = f"Optimize this plan: {request_data}"
        response = await self.generate_text(
            PlanningPrompt(
                system_prompt="You are an expert plan optimizer.",
                user_prompt=prompt,
                context={},
            )
        )
        return response.content

    async def _make_plan_validation_request(self, plan_data: Dict[str, Any]) -> str:
        """Make a request for plan validation."""
        prompt = f"Validate this plan: {plan_data}"
        response = await self.generate_text(
            PlanningPrompt(
                system_prompt="You are an expert plan validator.",
                user_prompt=prompt,
                context={},
            )
        )
        return response.content

    async def _make_plan_explanation_request(self, plan_data: Dict[str, Any]) -> str:
        """Make a request for plan explanation."""
        prompt = f"Explain this plan: {plan_data}"
        response = await self.generate_text(
            PlanningPrompt(
                system_prompt="You are an expert plan explainer.",
                user_prompt=prompt,
                context={},
            )
        )
        return response.content

    def _estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the text."""
        # Simple estimation: roughly 4 characters per token for English
        return len(text) // 4

    def _extract_json_from_xml(self, xml_response: str) -> str:
        """Extract JSON content from XML tags."""
        import re

        # Look for content between <json>, <analysis>, <result>, or <response> tags
        json_match = re.search(
            r"<(?:json|result|response|analysis)>(.*?)</(?:json|result|response|analysis)>",
            xml_response,
            re.DOTALL | re.IGNORECASE,
        )
        if json_match:
            return json_match.group(1).strip()
        return xml_response

    def _get_model_capabilities(self) -> Dict[str, Any]:
        """Get capabilities for the current model."""
        return {
            "max_tokens": self.config.max_tokens,
            "supports_streaming": True,
            "supports_function_calling": False,
            "context_window": 200000 if "claude-3" in self.config.model else 100000,
            "model_version": self.config.model,
        }

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response, handling Claude's XML wrapping."""
        import json

        # First try to extract JSON from XML tags
        extracted = self._extract_json_from_xml(response)

        try:
            result = json.loads(extracted)
            return cast(Dict[str, Any], result)
        except json.JSONDecodeError:
            # If parsing fails, return empty dict
            return {}

    def _truncate_prompt_if_needed(self, prompt: PlanningPrompt) -> PlanningPrompt:
        """Truncate prompt if it exceeds context length limits."""
        estimated_tokens = self._estimate_tokens(
            prompt.user_prompt + prompt.system_prompt
        )

        # Claude-3 has approximately 200k context window
        max_tokens = 150000  # Leave room for response

        if estimated_tokens > max_tokens:
            # Calculate how much to truncate
            reduction_ratio = max_tokens / estimated_tokens

            # Truncate user prompt (keep system prompt intact)
            new_length = int(len(prompt.user_prompt) * reduction_ratio)
            truncated_user_prompt = prompt.user_prompt[:new_length] + "..."

            return PlanningPrompt(
                system_prompt=prompt.system_prompt,
                user_prompt=truncated_user_prompt,
                context=prompt.context,
                examples=prompt.examples,
                constraints=prompt.constraints,
            )

        return prompt

    def _format_request(
        self, system_prompt: str, messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Format request for Anthropic API."""
        return {
            "system": system_prompt,
            "messages": messages,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }

    def _get_current_time(self) -> float:
        """Get current time for rate limiting."""
        return time.time()
