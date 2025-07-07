"""
LLM integration package for intelligent planning.

This package provides abstraction layers for different LLM providers
and prompt engineering for development environment planning.
"""

from .anthropic_client import AnthropicClient
from .base import LLMClient, LLMConfig, LLMResponse, PlanningPrompt
from .openai_client import OpenAIClient
from .prompt_templates import PromptTemplateManager, PromptType, TemplateConfig

__all__ = [
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "PlanningPrompt",
    "OpenAIClient",
    "AnthropicClient",
    "PromptTemplateManager",
    "PromptType",
    "TemplateConfig",
]
