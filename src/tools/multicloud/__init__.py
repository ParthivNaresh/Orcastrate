"""
Multi-cloud abstraction layer for Orcastrate.

This module provides a unified interface for managing resources across
multiple cloud providers (AWS, GCP, Azure) with common operations for
compute, database, storage, and networking resources.
"""

from .base import (
    CloudProvider,
    ComputeSpec,
    DatabaseSpec,
    MultiCloudManager,
    MultiCloudProvider,
    MultiCloudResource,
    NetworkSpec,
    ResourceSpec,
    ResourceType,
    SecurityGroupSpec,
    StorageSpec,
)
from .registry import ProviderRegistry, get_provider

__all__ = [
    "CloudProvider",
    "MultiCloudProvider",
    "MultiCloudResource",
    "MultiCloudManager",
    "ResourceSpec",
    "ComputeSpec",
    "DatabaseSpec",
    "StorageSpec",
    "NetworkSpec",
    "SecurityGroupSpec",
    "ResourceType",
    "ProviderRegistry",
    "get_provider",
]
