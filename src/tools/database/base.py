"""
Base abstractions for database tools.

This module provides the foundational classes and interfaces for all database tools,
emphasizing performance, async operations, and proper resource management.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncIterator, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from ..base import Tool, ToolConfig, ToolError


class DatabaseType(Enum):
    """Supported database types."""

    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    SQLITE = "sqlite"


class ConnectionState(Enum):
    """Database connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    host: str
    port: int
    database: str
    username: str
    password: str
    # Performance tuning
    min_connections: int = 1
    max_connections: int = 10
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0
    max_lifetime: float = 3600.0
    # SSL configuration
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    ssl_ca_path: Optional[str] = None
    # Additional options
    options: dict[str, Any] = field(default_factory=dict)


class QueryResult(BaseModel):
    """Result of a database query."""

    success: bool
    rows_affected: int = 0
    rows_returned: int = 0
    data: list[dict[str, Any]] = Field(default_factory=list)
    execution_time: float = 0.0
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class TransactionContext(BaseModel):
    """Context for database transactions."""

    transaction_id: str = Field(default_factory=lambda: str(uuid4()))
    started_at: datetime = Field(default_factory=datetime.utcnow)
    isolation_level: Optional[str] = None
    is_readonly: bool = False
    savepoints: list[str] = Field(default_factory=list)


class DatabaseConnection(ABC):
    """Abstract base class for database connections."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection_id = str(uuid4())
        self.state = ConnectionState.DISCONNECTED
        self.created_at = datetime.utcnow()
        self.last_used = datetime.utcnow()
        self.usage_count = 0
        self.logger = logging.getLogger(
            f"{self.__class__.__name__}:{self.connection_id[:8]}"
        )
        self._native_connection: Optional[Any] = None
        self._transaction_context: Optional[TransactionContext] = None

    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection."""

    @abstractmethod
    async def execute_query(
        self, query: str, params: Optional[dict[str, Any]] = None
    ) -> QueryResult:
        """Execute a database query."""

    @abstractmethod
    async def execute_many(
        self, query: str, param_list: list[dict[str, Any]]
    ) -> QueryResult:
        """Execute a query with multiple parameter sets."""

    @abstractmethod
    async def begin_transaction(
        self, isolation_level: Optional[str] = None
    ) -> TransactionContext:
        """Begin a database transaction."""

    @abstractmethod
    async def commit_transaction(self) -> None:
        """Commit the current transaction."""

    @abstractmethod
    async def rollback_transaction(self) -> None:
        """Rollback the current transaction."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if connection is healthy."""

    async def __aenter__(self) -> "DatabaseConnection":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._transaction_context and exc_type:
            await self.rollback_transaction()
        await self.disconnect()

    @property
    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self.state == ConnectionState.CONNECTED

    @property
    def age(self) -> timedelta:
        """Get connection age."""
        return datetime.utcnow() - self.created_at

    @property
    def idle_time(self) -> timedelta:
        """Get time since last use."""
        return datetime.utcnow() - self.last_used

    def mark_used(self) -> None:
        """Mark connection as recently used."""
        self.last_used = datetime.utcnow()
        self.usage_count += 1


class ConnectionPool:
    """High-performance async connection pool."""

    def __init__(self, config: DatabaseConfig, connection_factory):
        self.config = config
        self.connection_factory = connection_factory
        self.logger = logging.getLogger(f"ConnectionPool:{config.database}")

        # Pool management
        self._pool: set[DatabaseConnection] = set()
        self._available: asyncio.Queue[DatabaseConnection] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._closed = False

        # Metrics
        self.total_connections = 0
        self.active_connections = 0
        self.pool_hits = 0
        self.pool_misses = 0

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        # Create minimum connections
        for _ in range(self.config.min_connections):
            await self._create_connection()

        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def _create_connection(self) -> DatabaseConnection:
        """Create a new database connection."""
        connection: DatabaseConnection = self.connection_factory(self.config)
        await connection.connect()

        self._pool.add(connection)
        await self._available.put(connection)
        self.total_connections += 1

        self.logger.debug(f"Created new connection {connection.connection_id[:8]}")
        return connection

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[DatabaseConnection]:
        """Acquire a connection from the pool."""
        if self._closed:
            raise ToolError("Connection pool is closed")

        connection = await self._get_connection()
        self.active_connections += 1

        try:
            yield connection
        finally:
            await self._return_connection(connection)
            self.active_connections -= 1

    async def _get_connection(self) -> DatabaseConnection:
        """Get a connection from the pool."""
        time.time()

        try:
            # Try to get an available connection
            connection = await asyncio.wait_for(
                self._available.get(), timeout=self.config.connection_timeout
            )

            # Health check
            if await connection.health_check():
                connection.mark_used()
                self.pool_hits += 1
                return connection
            else:
                # Connection is unhealthy, create a new one
                await self._remove_connection(connection)
                return await self._create_new_connection()

        except asyncio.TimeoutError:
            # No available connections, try to create new one
            return await self._create_new_connection()

    async def _create_new_connection(self) -> DatabaseConnection:
        """Create a new connection if under limit."""
        async with self._lock:
            if len(self._pool) < self.config.max_connections:
                self.pool_misses += 1
                return await self._create_connection()
            else:
                raise ToolError(
                    f"Connection pool exhausted (max: {self.config.max_connections})"
                )

    async def _return_connection(self, connection: DatabaseConnection) -> None:
        """Return a connection to the pool."""
        if connection in self._pool and not self._closed:
            await self._available.put(connection)

    async def _remove_connection(self, connection: DatabaseConnection) -> None:
        """Remove a connection from the pool."""
        if connection in self._pool:
            self._pool.remove(connection)
            self.total_connections -= 1
            await connection.disconnect()
            self.logger.debug(f"Removed connection {connection.connection_id[:8]}")

    async def _cleanup_loop(self) -> None:
        """Background task to clean up idle connections."""
        while not self._closed:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_idle_connections()
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_idle_connections(self) -> None:
        """Remove idle and expired connections."""
        async with self._lock:
            to_remove = []

            for connection in self._pool:
                # Remove if connection is too old or idle too long
                if (
                    connection.age.total_seconds() > self.config.max_lifetime
                    or connection.idle_time.total_seconds() > self.config.idle_timeout
                ):
                    to_remove.append(connection)

            # Keep minimum connections
            if len(self._pool) - len(to_remove) < self.config.min_connections:
                to_remove = to_remove[: len(self._pool) - self.config.min_connections]

            for connection in to_remove:
                await self._remove_connection(connection)

    async def _monitor_loop(self) -> None:
        """Background task to monitor pool health."""
        while not self._closed:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                self.logger.debug(
                    f"Pool stats: total={self.total_connections}, "
                    f"active={self.active_connections}, "
                    f"available={self._available.qsize()}, "
                    f"hits={self.pool_hits}, misses={self.pool_misses}"
                )
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")

    async def close(self) -> None:
        """Close the connection pool."""
        self.logger.info("Closing connection pool")
        self._closed = True

        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._monitor_task:
            self._monitor_task.cancel()

        # Close all connections
        for connection in list(self._pool):
            await self._remove_connection(connection)

        self.logger.info("Connection pool closed")

    @property
    def stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "available_connections": self._available.qsize(),
            "pool_hits": self.pool_hits,
            "pool_misses": self.pool_misses,
            "hit_ratio": (
                self.pool_hits / (self.pool_hits + self.pool_misses)
                if (self.pool_hits + self.pool_misses) > 0
                else 0
            ),
        }


class DatabaseTool(Tool):
    """Base class for database tools."""

    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.db_config: Optional[DatabaseConfig] = None
        self.connection_pool: Optional[ConnectionPool] = None
        self.database_type: Optional[DatabaseType] = None

        # Performance metrics
        self.query_count = 0
        self.total_query_time = 0.0
        self.error_count = 0

    async def initialize(self) -> None:
        """Initialize the database tool."""
        # Create database config BEFORE calling super().initialize()
        # This ensures db_config is available when _create_client() is called
        self.db_config = self._create_db_config()

        # Now call parent initialization which will call _create_client()
        await super().initialize()

        # Initialize connection pool
        self.connection_pool = ConnectionPool(self.db_config, self._create_connection)
        await self.connection_pool.initialize()

        if self.database_type:
            self.logger.info(
                f"Database tool initialized for {self.database_type.value}"
            )
        else:
            self.logger.info("Database tool initialized")

    @abstractmethod
    def _create_db_config(self) -> DatabaseConfig:
        """Create database configuration from tool config."""

    @abstractmethod
    def _create_connection(self, config: DatabaseConfig) -> DatabaseConnection:
        """Create a database connection."""

    async def cleanup(self) -> None:
        """Clean up database tool resources."""
        if self.connection_pool:
            await self.connection_pool.close()
        # Note: Base Tool class cleanup() is not implemented yet

    async def execute_query(
        self,
        query: str,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> QueryResult:
        """Execute a database query with connection pooling."""
        start_time = time.time()

        try:
            if not self.connection_pool:
                raise ToolError("Database tool not initialized")

            async with self.connection_pool.acquire() as connection:
                if timeout:
                    result = await asyncio.wait_for(
                        connection.execute_query(query, params), timeout=timeout
                    )
                else:
                    result = await connection.execute_query(query, params)

            # Update metrics
            execution_time = time.time() - start_time
            self.query_count += 1
            self.total_query_time += execution_time

            return result

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Query execution failed: {e}")
            raise ToolError(f"Database query failed: {e}")

    @asynccontextmanager
    async def transaction(
        self, isolation_level: Optional[str] = None
    ) -> AsyncIterator[DatabaseConnection]:
        """Context manager for database transactions."""
        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            await connection.begin_transaction(isolation_level)

            try:
                yield connection
                await connection.commit_transaction()
            except Exception:
                await connection.rollback_transaction()
                raise

    @property
    def performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        avg_query_time = (
            self.total_query_time / self.query_count if self.query_count > 0 else 0
        )

        stats: dict[str, Any] = {
            "query_count": self.query_count,
            "total_query_time": self.total_query_time,
            "average_query_time": avg_query_time,
            "error_count": self.error_count,
            "error_rate": (
                self.error_count / self.query_count if self.query_count > 0 else 0
            ),
        }

        if self.connection_pool:
            pool_stats = self.connection_pool.stats
            stats["pool"] = pool_stats
            # Also include individual pool stats for backward compatibility
            stats.update(pool_stats)

        return stats
