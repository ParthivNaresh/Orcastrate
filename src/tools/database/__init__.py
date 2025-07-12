"""
Database tools package for Orcastrate.

This package provides high-performance database connectivity and management tools
for PostgreSQL, MySQL, MongoDB, and Redis with connection pooling, async operations,
and comprehensive monitoring capabilities.
"""

from .base import (
    ConnectionPool,
    ConnectionState,
    DatabaseConfig,
    DatabaseConnection,
    DatabaseTool,
    DatabaseType,
    QueryResult,
    TransactionContext,
)
from .mongodb import MongoDBTool
from .mysql import MySQLTool
from .postgresql import PostgreSQLTool
from .redis import RedisTool

__all__ = [
    # Base classes
    "DatabaseConfig",
    "DatabaseConnection",
    "DatabaseTool",
    "QueryResult",
    "TransactionContext",
    "ConnectionPool",
    "DatabaseType",
    "ConnectionState",
    # Specific implementations
    "PostgreSQLTool",
    "MySQLTool",
    "MongoDBTool",
    "RedisTool",
]
