"""
Cloud provider implementations for the multi-cloud abstraction layer.

This package contains provider-specific implementations that adapt
cloud provider APIs to the common multi-cloud interface.
"""

from .aws_provider import AWSProvider

__all__ = [
    "AWSProvider",
]
