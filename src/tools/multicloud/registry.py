"""
Provider registry for managing multi-cloud provider instances.

This module provides a factory pattern for registering and retrieving
cloud provider implementations dynamically.
"""

import logging
from typing import Any, Optional, Type

from .base import CloudProvider, MultiCloudProvider


class ProviderRegistry:
    """Registry for cloud provider implementations."""

    def __init__(self):
        """Initialize the provider registry."""
        self._providers: dict[CloudProvider, Type[MultiCloudProvider]] = {}
        self._instances: dict[str, MultiCloudProvider] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def register(
        self, provider_type: CloudProvider, provider_class: Type[MultiCloudProvider]
    ) -> None:
        """
        Register a cloud provider implementation.

        Args:
            provider_type: The cloud provider type
            provider_class: The provider implementation class
        """
        self._providers[provider_type] = provider_class
        self.logger.info(f"Registered provider: {provider_type.value}")

    def get_provider_class(
        self, provider_type: CloudProvider
    ) -> Optional[Type[MultiCloudProvider]]:
        """
        Get a registered provider class.

        Args:
            provider_type: The cloud provider type

        Returns:
            The provider class or None if not found
        """
        return self._providers.get(provider_type)

    def create_provider(
        self,
        provider_type: CloudProvider,
        config: dict[str, Any],
        instance_id: Optional[str] = None,
    ) -> MultiCloudProvider:
        """
        Create a provider instance with the given configuration.

        Args:
            provider_type: The cloud provider type
            config: Configuration for the provider
            instance_id: Optional instance ID for caching

        Returns:
            A configured provider instance

        Raises:
            ValueError: If provider type is not registered
        """
        provider_class = self.get_provider_class(provider_type)
        if not provider_class:
            raise ValueError(f"Provider {provider_type.value} is not registered")

        # Create instance
        instance = provider_class(config)

        # Cache instance if ID provided
        if instance_id:
            cache_key = f"{provider_type.value}:{instance_id}"
            self._instances[cache_key] = instance
            self.logger.debug(f"Cached provider instance: {cache_key}")

        return instance

    def get_cached_provider(
        self, provider_type: CloudProvider, instance_id: str
    ) -> Optional[MultiCloudProvider]:
        """
        Get a cached provider instance.

        Args:
            provider_type: The cloud provider type
            instance_id: The instance ID

        Returns:
            The cached provider instance or None if not found
        """
        cache_key = f"{provider_type.value}:{instance_id}"
        return self._instances.get(cache_key)

    def list_registered_providers(self) -> list[CloudProvider]:
        """
        List all registered cloud provider types.

        Returns:
            List of registered provider types
        """
        return list(self._providers.keys())

    def is_provider_registered(self, provider_type: CloudProvider) -> bool:
        """
        Check if a provider type is registered.

        Args:
            provider_type: The cloud provider type

        Returns:
            True if provider is registered
        """
        return provider_type in self._providers

    def clear_cache(self) -> None:
        """Clear all cached provider instances."""
        self._instances.clear()
        self.logger.info("Cleared provider instance cache")

    def remove_cached_instance(
        self, provider_type: CloudProvider, instance_id: str
    ) -> bool:
        """
        Remove a specific cached instance.

        Args:
            provider_type: The cloud provider type
            instance_id: The instance ID

        Returns:
            True if instance was found and removed
        """
        cache_key = f"{provider_type.value}:{instance_id}"
        if cache_key in self._instances:
            del self._instances[cache_key]
            self.logger.debug(f"Removed cached instance: {cache_key}")
            return True
        return False


# Global registry instance
_global_registry = ProviderRegistry()


def register_provider(
    provider_type: CloudProvider, provider_class: Type[MultiCloudProvider]
) -> None:
    """
    Register a cloud provider implementation globally.

    Args:
        provider_type: The cloud provider type
        provider_class: The provider implementation class
    """
    _global_registry.register(provider_type, provider_class)


def get_provider(
    provider_type: CloudProvider,
    config: dict[str, Any],
    instance_id: Optional[str] = None,
    use_cache: bool = True,
) -> MultiCloudProvider:
    """
    Get a cloud provider instance.

    Args:
        provider_type: The cloud provider type
        config: Configuration for the provider
        instance_id: Optional instance ID for caching
        use_cache: Whether to use cached instances

    Returns:
        A configured provider instance

    Raises:
        ValueError: If provider type is not registered
    """
    # Try to get cached instance first
    if use_cache and instance_id:
        cached = _global_registry.get_cached_provider(provider_type, instance_id)
        if cached:
            return cached

    # Create new instance
    return _global_registry.create_provider(provider_type, config, instance_id)


def list_providers() -> list[CloudProvider]:
    """
    List all registered cloud provider types.

    Returns:
        List of registered provider types
    """
    return _global_registry.list_registered_providers()


def is_provider_available(provider_type: CloudProvider) -> bool:
    """
    Check if a provider type is available.

    Args:
        provider_type: The cloud provider type

    Returns:
        True if provider is registered
    """
    return _global_registry.is_provider_registered(provider_type)


def clear_provider_cache() -> None:
    """Clear all cached provider instances."""
    _global_registry.clear_cache()


def get_registry() -> ProviderRegistry:
    """
    Get the global provider registry.

    Returns:
        The global provider registry instance
    """
    return _global_registry
