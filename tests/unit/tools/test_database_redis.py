"""
Unit tests for Redis database tool.

Tests the Redis tool functionality with mocked Redis connections.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.base import ToolConfig, ToolError
from src.tools.database.base import DatabaseConfig, DatabaseType
from src.tools.database.redis import RedisConnection, RedisTool


class TestRedisConnection:
    """Test Redis connection functionality."""

    @pytest.fixture
    def redis_config(self):
        """Create test Redis configuration."""
        return DatabaseConfig(
            host="localhost",
            port=6379,
            database="0",
            username="",
            password="test_password",
            min_connections=1,
            max_connections=5,
            connection_timeout=10.0,
            ssl_enabled=False,
        )

    @pytest.fixture
    def redis_connection(self, redis_config):
        """Create Redis connection instance."""
        return RedisConnection(redis_config)

    def test_connection_initialization(self, redis_connection, redis_config):
        """Test Redis connection initialization."""
        assert redis_connection.config == redis_config
        assert redis_connection._redis is None
        assert redis_connection._pipeline is None
        assert redis_connection._transaction_active is False

    def test_build_connection_url_with_password(self, redis_connection):
        """Test Redis connection URL building with password."""
        url = redis_connection._build_connection_url()
        assert url == "redis://:test_password@localhost:6379/0"

    def test_build_connection_url_without_password(self, redis_config):
        """Test Redis connection URL building without password."""
        redis_config.password = ""
        connection = RedisConnection(redis_config)
        url = connection._build_connection_url()
        assert url == "redis://localhost:6379/0"

    def test_build_connection_url_with_ssl(self, redis_config):
        """Test Redis connection URL building with SSL."""
        redis_config.ssl_enabled = True
        connection = RedisConnection(redis_config)
        url = connection._build_connection_url()
        assert url == "rediss://:test_password@localhost:6379/0"

    @pytest.mark.asyncio
    @patch("src.tools.database.redis.aioredis.from_url")
    async def test_connect_success(self, mock_from_url, redis_connection):
        """Test successful Redis connection."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_from_url.return_value = mock_redis

        # Test connection
        await redis_connection.connect()

        # Assertions
        mock_from_url.assert_called_once()
        mock_redis.ping.assert_called_once()
        assert redis_connection._redis == mock_redis
        assert redis_connection.is_connected

    @pytest.mark.asyncio
    @patch("src.tools.database.redis.aioredis.from_url")
    async def test_connect_failure(self, mock_from_url, redis_connection):
        """Test Redis connection failure."""
        # Setup mock to raise exception
        mock_from_url.side_effect = Exception("Connection failed")

        # Test connection failure
        with pytest.raises(ToolError, match="Redis connection failed"):
            await redis_connection.connect()

        assert not redis_connection.is_connected

    @pytest.mark.asyncio
    async def test_disconnect(self, redis_connection):
        """Test Redis disconnection."""
        # Setup mock Redis connection
        mock_redis = AsyncMock()
        redis_connection._redis = mock_redis

        # Test disconnection
        await redis_connection.disconnect()

        # Assertions
        mock_redis.close.assert_called_once()
        assert redis_connection._redis is None
        assert not redis_connection.is_connected

    @pytest.mark.asyncio
    async def test_execute_query_success(self, redis_connection):
        """Test successful Redis command execution."""
        # Setup mock Redis connection
        mock_redis = AsyncMock()
        mock_redis.execute_command = AsyncMock(return_value="OK")
        redis_connection._redis = mock_redis

        # Test query execution
        result = await redis_connection.execute_query("SET test_key test_value")

        # Assertions
        assert result.success
        assert result.data == [{"result": "OK"}]
        assert result.rows_returned == 1
        mock_redis.execute_command.assert_called_once_with(
            "SET", "test_key", "test_value"
        )

    @pytest.mark.asyncio
    async def test_execute_query_failure(self, redis_connection):
        """Test Redis command execution failure."""
        # Setup mock Redis connection
        mock_redis = AsyncMock()
        mock_redis.execute_command = AsyncMock(side_effect=Exception("Command failed"))
        redis_connection._redis = mock_redis

        # Test query execution failure
        result = await redis_connection.execute_query("INVALID COMMAND")

        # Assertions
        assert not result.success
        assert "Command failed" in result.error

    @pytest.mark.asyncio
    async def test_get_value_success(self, redis_connection):
        """Test successful GET operation."""
        # Setup mock Redis connection
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="test_value")
        redis_connection._redis = mock_redis

        # Test GET operation
        result = await redis_connection.get_value("test_key")

        # Assertions
        assert result.success
        assert result.data == [{"key": "test_key", "value": "test_value"}]
        assert result.rows_returned == 1
        mock_redis.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_get_value_not_found(self, redis_connection):
        """Test GET operation for non-existent key."""
        # Setup mock Redis connection
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        redis_connection._redis = mock_redis

        # Test GET operation
        result = await redis_connection.get_value("nonexistent_key")

        # Assertions
        assert result.success
        assert result.data == []
        assert result.rows_returned == 0

    @pytest.mark.asyncio
    async def test_set_value_without_ttl(self, redis_connection):
        """Test SET operation without TTL."""
        # Setup mock Redis connection
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        redis_connection._redis = mock_redis

        # Test SET operation
        result = await redis_connection.set_value("test_key", "test_value")

        # Assertions
        assert result.success
        assert result.rows_affected == 1
        mock_redis.set.assert_called_once_with("test_key", "test_value")

    @pytest.mark.asyncio
    async def test_set_value_with_ttl(self, redis_connection):
        """Test SET operation with TTL."""
        # Setup mock Redis connection
        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock(return_value=True)
        redis_connection._redis = mock_redis

        # Test SET operation with TTL
        result = await redis_connection.set_value("test_key", "test_value", ttl=300)

        # Assertions
        assert result.success
        assert result.rows_affected == 1
        mock_redis.setex.assert_called_once_with("test_key", 300, "test_value")

    @pytest.mark.asyncio
    async def test_delete_key(self, redis_connection):
        """Test DELETE operation."""
        # Setup mock Redis connection
        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock(return_value=1)
        redis_connection._redis = mock_redis

        # Test DELETE operation
        result = await redis_connection.delete_key("test_key")

        # Assertions
        assert result.success
        assert result.rows_affected == 1
        mock_redis.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_get_keys_pattern(self, redis_connection):
        """Test KEYS pattern matching."""
        # Setup mock Redis connection
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(return_value=["key1", "key2", "key3"])
        redis_connection._redis = mock_redis

        # Test KEYS operation
        result = await redis_connection.get_keys_pattern("test_*")

        # Assertions
        assert result.success
        assert result.rows_returned == 3
        expected_data = [{"key": "key1"}, {"key": "key2"}, {"key": "key3"}]
        assert result.data == expected_data
        mock_redis.keys.assert_called_once_with("test_*")

    @pytest.mark.asyncio
    async def test_hash_operations_hget(self, redis_connection):
        """Test HGET operation."""
        # Setup mock Redis connection
        mock_redis = AsyncMock()
        mock_redis.hget = AsyncMock(return_value="field_value")
        redis_connection._redis = mock_redis

        # Test HGET operation
        result = await redis_connection.hash_operations("test_hash", "HGET", "field1")

        # Assertions
        assert result.success
        assert result.data == [{"result": "field_value"}]
        mock_redis.hget.assert_called_once_with("test_hash", "field1")

    @pytest.mark.asyncio
    async def test_hash_operations_hset(self, redis_connection):
        """Test HSET operation."""
        # Setup mock Redis connection
        mock_redis = AsyncMock()
        mock_redis.hset = AsyncMock(return_value=1)
        redis_connection._redis = mock_redis

        # Test HSET operation
        result = await redis_connection.hash_operations(
            "test_hash", "HSET", "field1", "value1"
        )

        # Assertions
        assert result.success
        assert result.data == [{"result": 1}]
        mock_redis.hset.assert_called_once_with("test_hash", "field1", "value1")

    @pytest.mark.asyncio
    async def test_hash_operations_hgetall(self, redis_connection):
        """Test HGETALL operation."""
        # Setup mock Redis connection
        mock_redis = AsyncMock()
        mock_redis.hgetall = AsyncMock(
            return_value={"field1": "value1", "field2": "value2"}
        )
        redis_connection._redis = mock_redis

        # Test HGETALL operation
        result = await redis_connection.hash_operations("test_hash", "HGETALL")

        # Assertions
        assert result.success
        expected_result = {"field1": "value1", "field2": "value2"}
        assert result.data == [{"result": expected_result}]
        mock_redis.hgetall.assert_called_once_with("test_hash")

    @pytest.mark.asyncio
    async def test_execute_many_pipeline(self, redis_connection):
        """Test pipeline execution with multiple commands."""
        # Setup mock Redis connection and pipeline
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=["OK", "1", "value"])
        mock_redis = AsyncMock()
        mock_redis.pipeline = MagicMock(return_value=mock_pipeline)
        redis_connection._redis = mock_redis

        # Test pipeline execution
        query = "SET key{i} value{i}"
        param_list = [{"i": 1}, {"i": 2}, {"i": 3}]
        result = await redis_connection.execute_many(query, param_list)

        # Assertions
        assert result.success
        assert result.rows_affected == 3
        assert len(result.data) == 3

    @pytest.mark.asyncio
    async def test_begin_transaction(self, redis_connection):
        """Test transaction begin."""
        # Setup mock Redis connection and pipeline
        mock_pipeline = AsyncMock()
        mock_pipeline.multi = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.pipeline = MagicMock(return_value=mock_pipeline)
        redis_connection._redis = mock_redis

        # Test transaction begin
        context = await redis_connection.begin_transaction()

        # Assertions
        assert redis_connection._pipeline == mock_pipeline
        assert redis_connection._transaction_active
        assert context is not None
        mock_pipeline.multi.assert_called_once()

    @pytest.mark.asyncio
    async def test_commit_transaction(self, redis_connection):
        """Test transaction commit."""
        # Setup mock pipeline
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock()
        redis_connection._pipeline = mock_pipeline
        redis_connection._transaction_active = True

        # Test transaction commit
        await redis_connection.commit_transaction()

        # Assertions
        assert redis_connection._pipeline is None
        assert not redis_connection._transaction_active
        mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback_transaction(self, redis_connection):
        """Test transaction rollback."""
        # Setup mock pipeline
        mock_pipeline = AsyncMock()
        mock_pipeline.discard = AsyncMock()
        redis_connection._pipeline = mock_pipeline
        redis_connection._transaction_active = True

        # Test transaction rollback
        await redis_connection.rollback_transaction()

        # Assertions
        assert redis_connection._pipeline is None
        assert not redis_connection._transaction_active
        mock_pipeline.discard.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, redis_connection):
        """Test health check with healthy connection."""
        # Setup mock Redis connection
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        redis_connection._redis = mock_redis

        # Test health check
        is_healthy = await redis_connection.health_check()

        # Assertions
        assert is_healthy
        mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, redis_connection):
        """Test health check with unhealthy connection."""
        # Test health check without connection
        is_healthy = await redis_connection.health_check()
        assert not is_healthy

        # Test health check with failing ping
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=Exception("Connection lost"))
        redis_connection._redis = mock_redis

        is_healthy = await redis_connection.health_check()
        assert not is_healthy


class TestRedisTool:
    """Test Redis tool functionality."""

    @pytest.fixture
    def redis_tool_config(self):
        """Create test Redis tool configuration."""
        return ToolConfig(
            name="redis",
            version="1.0.0",
            environment={
                "host": "localhost",
                "port": 6379,
                "database": "1",
                "password": "test_password",
                "min_connections": 1,
                "max_connections": 5,
                "database_number": 1,
            },
        )

    @pytest.fixture
    def redis_tool(self, redis_tool_config):
        """Create Redis tool instance."""
        return RedisTool(redis_tool_config)

    def test_tool_initialization(self, redis_tool):
        """Test Redis tool initialization."""
        assert redis_tool.database_type == DatabaseType.REDIS
        assert redis_tool.config.name == "redis"

    def test_create_db_config(self, redis_tool):
        """Test database configuration creation."""
        config = redis_tool._create_db_config()

        assert config.host == "localhost"
        assert config.port == 6379
        assert config.database == "1"
        assert config.password == "test_password"
        assert hasattr(config, "database_number")
        assert getattr(config, "database_number") == 1

    def test_create_connection(self, redis_tool):
        """Test connection creation."""
        config = redis_tool._create_db_config()
        connection = redis_tool._create_connection(config)

        assert isinstance(connection, RedisConnection)
        assert connection.config == config

    @pytest.mark.asyncio
    async def test_get_schema(self, redis_tool):
        """Test tool schema retrieval."""
        schema = await redis_tool.get_schema()

        assert schema.name == "redis"
        assert schema.description == "High-performance Redis database tool"
        assert "get" in schema.actions
        assert "set" in schema.actions
        assert "delete" in schema.actions
        assert "hash_get" in schema.actions
        assert "execute_command" in schema.actions

    @pytest.mark.asyncio
    async def test_get_action(self, redis_tool):
        """Test GET action execution."""
        with patch.object(redis_tool, "connection_pool") as mock_pool:
            # Setup mock connection
            mock_connection = AsyncMock()
            mock_connection.get_value = AsyncMock()
            mock_connection.get_value.return_value.success = True
            mock_connection.get_value.return_value.data = [
                {"key": "test_key", "value": "test_value"}
            ]
            mock_connection.get_value.return_value.execution_time = 0.001

            # Setup async context manager
            mock_acquire = AsyncMock()
            mock_acquire.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_acquire.__aexit__ = AsyncMock(return_value=None)
            mock_pool.acquire.return_value = mock_acquire

            # Test GET action
            result = await redis_tool._get_action({"key": "test_key"})

            # Assertions
            assert result["success"]
            assert result["value"] == "test_value"
            assert result["found"]
            mock_connection.get_value.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_set_action(self, redis_tool):
        """Test SET action execution."""
        with patch.object(redis_tool, "connection_pool") as mock_pool:
            # Setup mock connection
            mock_connection = AsyncMock()
            mock_connection.set_value = AsyncMock()
            mock_connection.set_value.return_value.success = True
            mock_connection.set_value.return_value.execution_time = 0.001

            # Setup async context manager
            mock_acquire = AsyncMock()
            mock_acquire.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_acquire.__aexit__ = AsyncMock(return_value=None)
            mock_pool.acquire.return_value = mock_acquire

            # Test SET action
            result = await redis_tool._set_action(
                {"key": "test_key", "value": "test_value", "ttl": 300}
            )

            # Assertions
            assert result["success"]
            assert result["key"] == "test_key"
            assert result["value"] == "test_value"
            assert result["ttl"] == 300
            mock_connection.set_value.assert_called_once_with(
                "test_key", "test_value", 300
            )

    # Note: Additional action tests (delete, keys, hash operations) are covered
    # comprehensively in the integration tests. Unit tests focus on core tool
    # functionality, connection management, and schema validation.

    @pytest.mark.asyncio
    async def test_estimate_cost(self, redis_tool):
        """Test cost estimation."""
        # Test different operation costs
        get_cost = await redis_tool.estimate_cost("get", {"key": "test"})
        assert get_cost.estimated_cost == 0.0001

        set_cost = await redis_tool.estimate_cost(
            "set", {"key": "test", "value": "value"}
        )
        assert set_cost.estimated_cost == 0.0002

        keys_cost = await redis_tool.estimate_cost("keys", {"pattern": "*"})
        assert keys_cost.estimated_cost == 0.005

        hash_cost = await redis_tool.estimate_cost("hash_getall", {"key": "hash"})
        assert hash_cost.estimated_cost == 0.001

    @pytest.mark.asyncio
    async def test_get_supported_actions(self, redis_tool):
        """Test supported actions list."""
        actions = await redis_tool._get_supported_actions()

        expected_actions = [
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

        for action in expected_actions:
            assert action in actions

    @pytest.mark.asyncio
    async def test_execute_rollback(self, redis_tool):
        """Test rollback execution (placeholder)."""
        result = await redis_tool._execute_rollback("test_execution_id")
        assert "Rollback not implemented" in result["message"]

    @pytest.mark.asyncio
    async def test_create_client(self, redis_tool):
        """Test client creation."""
        redis_tool.db_config = redis_tool._create_db_config()
        client = await redis_tool._create_client()

        assert isinstance(client, RedisConnection)
        assert client.config == redis_tool.db_config

    @pytest.mark.asyncio
    async def test_create_validator(self, redis_tool):
        """Test validator creation."""
        validator = await redis_tool._create_validator()
        assert validator is None  # Database tools don't need validators
