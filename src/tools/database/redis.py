"""
Redis database tool implementation.

High-performance Redis connector with async operations, key-value management,
and comprehensive cache operations.
"""

import time
from typing import Any, Dict, List, Optional, Union, cast

import redis.asyncio as aioredis
from redis.asyncio import Redis

from ..base import CostEstimate, ToolConfig, ToolError, ToolSchema
from .base import (
    ConnectionState,
    DatabaseConfig,
    DatabaseConnection,
    DatabaseTool,
    DatabaseType,
    QueryResult,
    TransactionContext,
)


class RedisConnection(DatabaseConnection):
    """High-performance Redis connection with aioredis."""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._redis: Optional[Redis] = None
        self._pipeline: Optional[Any] = None
        self._transaction_active = False

    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            self.state = ConnectionState.CONNECTING
            self.logger.debug(
                f"Connecting to Redis at {self.config.host}:{self.config.port}"
            )

            # Build connection URL
            url = self._build_connection_url()

            # Connect with optimized settings
            self._redis = aioredis.from_url(
                url,
                encoding="utf-8",
                decode_responses=True,
                health_check_interval=30,
                socket_keepalive=True,
                socket_keepalive_options={},
                retry_on_timeout=True,
                socket_connect_timeout=self.config.connection_timeout,
                max_connections=self.config.max_connections,
            )

            # Test connection
            await self._redis.ping()

            self.state = ConnectionState.CONNECTED
            self.logger.debug(f"Connected to Redis {self.connection_id[:8]}")

        except Exception as e:
            self.state = ConnectionState.ERROR
            raise ToolError(f"Redis connection failed: {e}")

    def _build_connection_url(self) -> str:
        """Build Redis connection URL."""
        if self.config.password:
            auth = f":{self.config.password}@"
        else:
            auth = ""

        url_parts = [f"redis://{auth}{self.config.host}:{self.config.port}"]

        # Add database number
        if hasattr(self.config, "database_number"):
            url_parts.append(f"/{getattr(self.config, 'database_number', 0)}")
        else:
            url_parts.append("/0")  # Default to database 0

        # SSL configuration
        if self.config.ssl_enabled:
            url_parts[0] = url_parts[0].replace("redis://", "rediss://")

        return "".join(url_parts)

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            self.state = ConnectionState.CLOSING
            try:
                await self._redis.close()
                self.logger.debug(f"Disconnected from Redis {self.connection_id[:8]}")
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")
            finally:
                self._redis = None
                self.state = ConnectionState.DISCONNECTED

    async def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Execute Redis command (not traditional SQL query)."""
        if not self._redis:
            raise ToolError("Not connected to Redis")

        start_time = time.time()

        try:
            # Parse Redis command from query
            parts = query.strip().split()
            if not parts:
                raise ToolError("Empty Redis command")

            command = parts[0].upper()
            args = parts[1:] if len(parts) > 1 else []

            # Execute command
            result = await self._redis.execute_command(command, *args)

            execution_time = time.time() - start_time

            return QueryResult(
                success=True,
                data=[{"result": result}] if result is not None else [],
                rows_returned=1 if result is not None else 0,
                execution_time=execution_time,
                metadata={
                    "command": command,
                    "connection_id": self.connection_id,
                    "args": args,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Redis command failed: {e}")

            return QueryResult(
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"connection_id": self.connection_id},
            )

    async def get_value(self, key: str) -> QueryResult:
        """Get value by key."""
        if not self._redis:
            raise ToolError("Not connected to Redis")

        start_time = time.time()

        try:
            value = await self._redis.get(key)
            execution_time = time.time() - start_time

            return QueryResult(
                success=True,
                data=[{"key": key, "value": value}] if value is not None else [],
                rows_returned=1 if value is not None else 0,
                execution_time=execution_time,
                metadata={
                    "operation": "get",
                    "connection_id": self.connection_id,
                    "key": key,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"GET operation failed: {e}")

            return QueryResult(
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"connection_id": self.connection_id},
            )

    async def set_value(
        self, key: str, value: Union[str, int, float], ttl: Optional[int] = None
    ) -> QueryResult:
        """Set key-value pair with optional TTL."""
        if not self._redis:
            raise ToolError("Not connected to Redis")

        start_time = time.time()

        try:
            # Set value with optional TTL
            if ttl:
                result = await self._redis.setex(key, ttl, value)
            else:
                result = await self._redis.set(key, value)

            execution_time = time.time() - start_time

            return QueryResult(
                success=result is not False,
                rows_affected=1 if result else 0,
                execution_time=execution_time,
                metadata={
                    "operation": "set",
                    "connection_id": self.connection_id,
                    "key": key,
                    "ttl": ttl,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"SET operation failed: {e}")

            return QueryResult(
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"connection_id": self.connection_id},
            )

    async def delete_key(self, key: str) -> QueryResult:
        """Delete key."""
        if not self._redis:
            raise ToolError("Not connected to Redis")

        start_time = time.time()

        try:
            result = await self._redis.delete(key)
            execution_time = time.time() - start_time

            return QueryResult(
                success=True,
                rows_affected=result,
                execution_time=execution_time,
                metadata={
                    "operation": "delete",
                    "connection_id": self.connection_id,
                    "key": key,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"DELETE operation failed: {e}")

            return QueryResult(
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"connection_id": self.connection_id},
            )

    async def get_keys_pattern(self, pattern: str = "*") -> QueryResult:
        """Get keys matching pattern."""
        if not self._redis:
            raise ToolError("Not connected to Redis")

        start_time = time.time()

        try:
            keys = await self._redis.keys(pattern)
            execution_time = time.time() - start_time

            data = [{"key": key} for key in keys]

            return QueryResult(
                success=True,
                data=data,
                rows_returned=len(keys),
                execution_time=execution_time,
                metadata={
                    "operation": "keys",
                    "connection_id": self.connection_id,
                    "pattern": pattern,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"KEYS operation failed: {e}")

            return QueryResult(
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"connection_id": self.connection_id},
            )

    async def hash_operations(
        self,
        key: str,
        operation: str,
        field: Optional[str] = None,
        value: Optional[str] = None,
    ) -> QueryResult:
        """Hash operations (HGET, HSET, HGETALL, HDEL, etc.)."""
        if not self._redis:
            raise ToolError("Not connected to Redis")

        start_time = time.time()

        try:
            result = None
            operation_upper = operation.upper()

            if operation_upper == "HGET" and field:
                result = await self._redis.hget(key, field)  # type: ignore
            elif operation_upper == "HSET" and field and value:
                result = await self._redis.hset(key, field, value)  # type: ignore
            elif operation_upper == "HGETALL":
                result = await self._redis.hgetall(key)  # type: ignore
            elif operation_upper == "HDEL" and field:
                result = await self._redis.hdel(key, field)  # type: ignore
            elif operation_upper == "HKEYS":
                result = await self._redis.hkeys(key)  # type: ignore
            elif operation_upper == "HVALS":
                result = await self._redis.hvals(key)  # type: ignore
            else:
                raise ToolError(f"Unsupported hash operation: {operation}")

            execution_time = time.time() - start_time

            return QueryResult(
                success=True,
                data=[{"result": result}] if result is not None else [],
                rows_returned=1 if result is not None else 0,
                execution_time=execution_time,
                metadata={
                    "operation": operation_upper,
                    "connection_id": self.connection_id,
                    "key": key,
                    "field": field,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Hash operation failed: {e}")

            return QueryResult(
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"connection_id": self.connection_id},
            )

    async def execute_many(
        self, query: str, param_list: List[Dict[str, Any]]
    ) -> QueryResult:
        """Execute many operations using pipeline."""
        if not self._redis:
            raise ToolError("Not connected to Redis")

        start_time = time.time()

        try:
            pipe = self._redis.pipeline()

            for params in param_list:
                # Build command from query template and params
                formatted_query = query.format(**params)
                parts = formatted_query.strip().split()
                if parts:
                    command = parts[0].upper()
                    args = parts[1:] if len(parts) > 1 else []
                    pipe.execute_command(command, *args)

            results = await pipe.execute()
            execution_time = time.time() - start_time

            return QueryResult(
                success=True,
                data=[{"result": result} for result in results],
                rows_affected=len(results),
                execution_time=execution_time,
                metadata={
                    "operation": "pipeline",
                    "connection_id": self.connection_id,
                    "commands_count": len(param_list),
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Pipeline operation failed: {e}")

            return QueryResult(
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"connection_id": self.connection_id},
            )

    async def begin_transaction(
        self, isolation_level: Optional[str] = None
    ) -> TransactionContext:
        """Begin Redis transaction (MULTI/EXEC)."""
        if not self._redis:
            raise ToolError("Not connected to Redis")

        try:
            self._pipeline = self._redis.pipeline()
            await self._pipeline.multi()
            self._transaction_active = True

            context = TransactionContext()
            self._transaction_context = context

            return context

        except Exception as e:
            self.logger.error(f"Failed to begin transaction: {e}")
            raise ToolError(f"Transaction begin failed: {e}")

    async def commit_transaction(self) -> None:
        """Commit Redis transaction (EXEC)."""
        if not self._pipeline:
            raise ToolError("No active transaction")

        try:
            await self._pipeline.execute()
            self._pipeline = None
            self._transaction_active = False
            self._transaction_context = None
        except Exception as e:
            self.logger.error(f"Failed to commit transaction: {e}")
            raise ToolError(f"Transaction commit failed: {e}")

    async def rollback_transaction(self) -> None:
        """Rollback Redis transaction (DISCARD)."""
        if not self._pipeline:
            raise ToolError("No active transaction")

        try:
            await self._pipeline.discard()
            self._pipeline = None
            self._transaction_active = False
            self._transaction_context = None
        except Exception as e:
            self.logger.error(f"Failed to rollback transaction: {e}")
            raise ToolError(f"Transaction rollback failed: {e}")

    async def health_check(self) -> bool:
        """Check Redis connection health."""
        if not self._redis:
            return False

        try:
            await self._redis.ping()
            return True
        except Exception:
            return False


class RedisTool(DatabaseTool):
    """High-performance Redis database tool."""

    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.database_type = DatabaseType.REDIS

    def _create_db_config(self) -> DatabaseConfig:
        """Create Redis configuration from tool config."""
        env = self.config.environment

        # Create config with database_number for Redis
        config = DatabaseConfig(
            host=env.get("host", "localhost"),
            port=int(env.get("port", 6379)),
            database=env.get("database", "0"),  # Redis database number
            username=env.get("username", ""),
            password=env.get("password", ""),
            min_connections=int(env.get("min_connections", 1)),
            max_connections=int(env.get("max_connections", 10)),
            connection_timeout=float(env.get("connection_timeout", 30.0)),
            idle_timeout=float(env.get("idle_timeout", 300.0)),
            max_lifetime=float(env.get("max_lifetime", 3600.0)),
            ssl_enabled=bool(env.get("ssl_enabled", False)),
            ssl_cert_path=env.get("ssl_cert_path"),
            ssl_key_path=env.get("ssl_key_path"),
            ssl_ca_path=env.get("ssl_ca_path"),
            options=env.get("options", {}),
        )

        # Add Redis-specific database number
        setattr(config, "database_number", env.get("database_number", 0))

        return config

    def _create_connection(self, config: DatabaseConfig) -> DatabaseConnection:
        """Create Redis connection."""
        return RedisConnection(config)

    async def get_schema(self) -> ToolSchema:
        """Return Redis tool schema."""
        return ToolSchema(
            name="redis",
            description="High-performance Redis database tool",
            version=self.config.version,
            actions={
                "get": {
                    "description": "Get value by key",
                    "parameters": {
                        "key": {"type": "string", "required": True},
                    },
                },
                "set": {
                    "description": "Set key-value pair",
                    "parameters": {
                        "key": {"type": "string", "required": True},
                        "value": {"type": "string", "required": True},
                        "ttl": {
                            "type": "integer",
                            "description": "Time to live in seconds",
                        },
                    },
                },
                "delete": {
                    "description": "Delete key",
                    "parameters": {
                        "key": {"type": "string", "required": True},
                    },
                },
                "keys": {
                    "description": "Get keys matching pattern",
                    "parameters": {
                        "pattern": {"type": "string", "default": "*"},
                    },
                },
                "hash_get": {
                    "description": "Get field from hash",
                    "parameters": {
                        "key": {"type": "string", "required": True},
                        "field": {"type": "string", "required": True},
                    },
                },
                "hash_set": {
                    "description": "Set field in hash",
                    "parameters": {
                        "key": {"type": "string", "required": True},
                        "field": {"type": "string", "required": True},
                        "value": {"type": "string", "required": True},
                    },
                },
                "hash_getall": {
                    "description": "Get all fields from hash",
                    "parameters": {
                        "key": {"type": "string", "required": True},
                    },
                },
                "hash_delete": {
                    "description": "Delete field from hash",
                    "parameters": {
                        "key": {"type": "string", "required": True},
                        "field": {"type": "string", "required": True},
                    },
                },
                "execute_command": {
                    "description": "Execute Redis command",
                    "parameters": {
                        "command": {"type": "string", "required": True},
                    },
                },
                "get_statistics": {
                    "description": "Get Redis server statistics",
                    "parameters": {},
                },
            },
        )

    async def _execute_action(
        self, action: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Redis action."""
        if action == "get":
            return await self._get_action(params)
        elif action == "set":
            return await self._set_action(params)
        elif action == "delete":
            return await self._delete_action(params)
        elif action == "keys":
            return await self._keys_action(params)
        elif action == "hash_get":
            return await self._hash_get_action(params)
        elif action == "hash_set":
            return await self._hash_set_action(params)
        elif action == "hash_getall":
            return await self._hash_getall_action(params)
        elif action == "hash_delete":
            return await self._hash_delete_action(params)
        elif action == "execute_command":
            return await self._execute_command_action(params)
        elif action == "get_statistics":
            return await self._get_statistics_action(params)
        else:
            raise ToolError(f"Unknown action: {action}")

    async def _get_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get value by key."""
        key = params["key"]

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            redis_conn = cast(RedisConnection, connection)
            result = await redis_conn.get_value(key)

        return {
            "success": result.success,
            "value": result.data[0]["value"] if result.data else None,
            "found": len(result.data) > 0,
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _set_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set key-value pair."""
        key = params["key"]
        value = params["value"]
        ttl = params.get("ttl")

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            redis_conn = cast(RedisConnection, connection)
            result = await redis_conn.set_value(key, value, ttl)

        return {
            "success": result.success,
            "key": key,
            "value": value,
            "ttl": ttl,
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _delete_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete key."""
        key = params["key"]

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            redis_conn = cast(RedisConnection, connection)
            result = await redis_conn.delete_key(key)

        return {
            "success": result.success,
            "key": key,
            "deleted": result.rows_affected > 0,
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _keys_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get keys matching pattern."""
        pattern = params.get("pattern", "*")

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            redis_conn = cast(RedisConnection, connection)
            result = await redis_conn.get_keys_pattern(pattern)

        return {
            "success": result.success,
            "keys": [item["key"] for item in result.data],
            "count": result.rows_returned,
            "pattern": pattern,
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _hash_get_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get field from hash."""
        key = params["key"]
        field = params["field"]

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            redis_conn = cast(RedisConnection, connection)
            result = await redis_conn.hash_operations(key, "HGET", field)

        return {
            "success": result.success,
            "key": key,
            "field": field,
            "value": result.data[0]["result"] if result.data else None,
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _hash_set_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set field in hash."""
        key = params["key"]
        field = params["field"]
        value = params["value"]

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            redis_conn = cast(RedisConnection, connection)
            result = await redis_conn.hash_operations(key, "HSET", field, value)

        return {
            "success": result.success,
            "key": key,
            "field": field,
            "value": value,
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _hash_getall_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get all fields from hash."""
        key = params["key"]

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            redis_conn = cast(RedisConnection, connection)
            result = await redis_conn.hash_operations(key, "HGETALL")

        return {
            "success": result.success,
            "key": key,
            "hash": result.data[0]["result"] if result.data else {},
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _hash_delete_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete field from hash."""
        key = params["key"]
        field = params["field"]

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            redis_conn = cast(RedisConnection, connection)
            result = await redis_conn.hash_operations(key, "HDEL", field)

        return {
            "success": result.success,
            "key": key,
            "field": field,
            "deleted": result.data[0]["result"] if result.data else 0,
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _execute_command_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Redis command."""
        command = params["command"]

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            redis_conn = cast(RedisConnection, connection)
            result = await redis_conn.execute_query(command)

        # Extract the result value
        result_value = result.data[0]["result"] if result.data else None

        # Convert boolean True to "OK" for SET commands (redis-py converts "OK" to True)
        if result_value is True:
            command_parts = command.strip().split()
            if command_parts and command_parts[0].upper() == "SET":
                result_value = "OK"

        return {
            "success": result.success,
            "result": result_value,
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _get_statistics_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get Redis server statistics."""
        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        try:
            async with self.connection_pool.acquire() as connection:
                redis_conn = cast(RedisConnection, connection)
                if not redis_conn._redis:
                    raise ToolError("Not connected to Redis")

                # Get Redis INFO
                info = await redis_conn._redis.info()

                # Extract key statistics
                stats = {
                    "redis_version": info.get("redis_version"),
                    "uptime_in_seconds": info.get("uptime_in_seconds"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory": info.get("used_memory"),
                    "used_memory_human": info.get("used_memory_human"),
                    "total_commands_processed": info.get("total_commands_processed"),
                    "keyspace_hits": info.get("keyspace_hits"),
                    "keyspace_misses": info.get("keyspace_misses"),
                    "db_keys": {},
                }

                # Add database key counts
                for key, value in info.items():
                    if key.startswith("db"):
                        stats["db_keys"][key] = value

                return {
                    "success": True,
                    "statistics": stats,
                    "tool_stats": self.performance_stats,
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def estimate_cost(self, action: str, params: Dict[str, Any]) -> CostEstimate:
        """Estimate cost of Redis operations."""
        # Redis is typically free/open source, so cost is mainly computational
        base_cost = 0.0

        if action in ["get", "hash_get"]:
            base_cost = 0.0001  # Very cheap read operations
        elif action in ["set", "hash_set", "delete", "hash_delete"]:
            base_cost = 0.0002  # Slightly more expensive write operations
        elif action == "keys":
            # KEYS operation can be expensive on large datasets
            base_cost = 0.005
        elif action == "hash_getall":
            base_cost = 0.001  # Depends on hash size
        elif action == "execute_command":
            base_cost = 0.001  # General command execution

        return CostEstimate(estimated_cost=base_cost, currency="USD", confidence=0.7)

    async def _get_supported_actions(self) -> List[str]:
        """Get list of supported Redis actions."""
        return [
            "get",
            "set",
            "delete",
            "keys",
            "hash_get",
            "hash_set",
            "hash_getall",
            "hash_delete",
            "execute_command",
            "get_statistics",
        ]

    async def _execute_rollback(self, execution_id: str) -> Dict[str, Any]:
        """Execute rollback operation for Redis."""
        # TODO: Implement rollback logic
        return {"message": f"Rollback not implemented for execution {execution_id}"}

    async def _create_client(self) -> RedisConnection:
        """Create Redis connection client."""
        if self.db_config is None:
            raise ToolError("Database configuration not initialized")
        return RedisConnection(self.db_config)

    async def _create_validator(self) -> None:
        """Create parameter validator (not needed for database tools)."""
        return None
