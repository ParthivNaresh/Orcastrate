"""
Configuration management for Orcastrate.

This module provides centralized configuration management using Pydantic
for type safety, validation, and environment-based settings.
"""

from .settings import AppSettings, get_settings

__all__ = ["get_settings", "AppSettings"]
