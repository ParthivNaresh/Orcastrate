"""
Unit tests for PostgreSQL database tool.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.base import ToolConfig, ToolError
from src.tools.database.base import DatabaseConfig, QueryResult
from src.tools.database.postgresql import PostgreSQLConnection, PostgreSQLTool


class TestPostgreSQLConnection:
    """Test PostgreSQL connection implementation."""

    @pytest.fixture
    def db_config(self):
        """Create test database configuration."""
        return DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass",
            min_connections=1,
            max_connections=5,
        )

    @pytest.fixture
    def connection(self, db_config):
        """Create PostgreSQL connection instance."""
        return PostgreSQLConnection(db_config)

    @pytest.mark.asyncio
    async def test_connection_initialization(self, connection, db_config):
        """Test connection initialization."""
        assert connection.config == db_config
        assert connection.connection_id is not None
        assert connection._connection is None
        assert not connection.is_connected

    @pytest.mark.asyncio
    async def test_build_dsn(self, connection):
        """Test DSN building."""
        dsn = connection._build_dsn()
        assert "postgresql://test_user:test_pass@localhost:5432/test_db" in dsn

    @pytest.mark.asyncio
    async def test_build_dsn_with_ssl(self, db_config):
        """Test DSN building with SSL options."""
        db_config.ssl_enabled = True
        db_config.ssl_cert_path = "/path/to/cert.pem"
        db_config.ssl_key_path = "/path/to/key.pem"

        connection = PostgreSQLConnection(db_config)
        dsn = connection._build_dsn()

        assert "sslmode=require" in dsn
        assert "sslcert=/path/to/cert.pem" in dsn
        assert "sslkey=/path/to/key.pem" in dsn

    @pytest.mark.asyncio
    @patch("src.tools.database.postgresql.asyncpg.connect")
    async def test_connect_success(self, mock_connect, connection):
        """Test successful connection."""
        mock_conn = AsyncMock()
        mock_connect.return_value = mock_conn

        await connection.connect()

        assert connection.is_connected
        assert connection._connection == mock_conn
        mock_connect.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.tools.database.postgresql.asyncpg.connect")
    async def test_connect_failure(self, mock_connect, connection):
        """Test connection failure."""
        mock_connect.side_effect = Exception("Connection failed")

        with pytest.raises(ToolError, match="PostgreSQL connection failed"):
            await connection.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, connection):
        """Test disconnection."""
        mock_conn = AsyncMock()
        connection._connection = mock_conn

        await connection.disconnect()

        mock_conn.close.assert_called_once()
        assert connection._connection is None

    @pytest.mark.asyncio
    async def test_convert_params(self, connection):
        """Test parameter conversion."""
        query = "SELECT * FROM table WHERE id = %(id)s AND name = %(name)s"
        params = {"id": 1, "name": "test"}

        converted_query, args = connection._convert_params(query, params)

        assert "$1" in converted_query
        assert "$2" in converted_query
        assert args == [1, "test"]

    @pytest.mark.asyncio
    async def test_execute_query_select(self, connection):
        """Test SELECT query execution."""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{"id": 1, "name": "test"}]
        connection._connection = mock_conn

        result = await connection.execute_query("SELECT * FROM table")

        assert result.success
        assert result.rows_returned == 1
        assert result.data == [{"id": 1, "name": "test"}]

    @pytest.mark.asyncio
    async def test_execute_query_insert(self, connection):
        """Test INSERT query execution."""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "INSERT 0 1"
        connection._connection = mock_conn

        result = await connection.execute_query("INSERT INTO table VALUES (1, 'test')")

        assert result.success
        assert result.rows_affected == 1

    @pytest.mark.asyncio
    async def test_execute_query_error(self, connection):
        """Test query execution error."""
        mock_conn = AsyncMock()
        mock_conn.fetch.side_effect = Exception("Query failed")
        connection._connection = mock_conn

        result = await connection.execute_query("SELECT * FROM table")

        assert not result.success
        assert "Query failed" in result.error

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, connection):
        """Test health check when connection is healthy."""
        mock_conn = MagicMock()  # Use MagicMock for sync methods
        mock_conn.is_closed.return_value = False  # Mock method call
        mock_conn.fetchval = AsyncMock(return_value=1)  # Mock async method
        connection._connection = mock_conn

        is_healthy = await connection.health_check()

        assert is_healthy
        mock_conn.is_closed.assert_called_once()
        mock_conn.fetchval.assert_called_once_with("SELECT 1")

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, connection):
        """Test health check when connection is unhealthy."""
        mock_conn = AsyncMock()
        mock_conn.is_closed.return_value = True
        connection._connection = mock_conn

        is_healthy = await connection.health_check()

        assert not is_healthy


class TestPostgreSQLTool:
    """Test PostgreSQL tool implementation."""

    @pytest.fixture
    def tool_config(self):
        """Create test tool configuration."""
        return ToolConfig(
            name="postgresql",
            version="1.0.0",
            environment={
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "username": "test_user",
                "password": "test_pass",
            },
        )

    @pytest.fixture
    def tool(self, tool_config):
        """Create PostgreSQL tool instance."""
        return PostgreSQLTool(tool_config)

    def test_tool_initialization(self, tool, tool_config):
        """Test tool initialization."""
        assert tool.config == tool_config
        assert tool.database_type.value == "postgresql"

    def test_create_db_config(self, tool):
        """Test database config creation."""
        db_config = tool._create_db_config()

        assert db_config.host == "localhost"
        assert db_config.port == 5432
        assert db_config.database == "test_db"
        assert db_config.username == "test_user"
        assert db_config.password == "test_pass"

    def test_create_connection(self, tool):
        """Test connection creation."""
        db_config = tool._create_db_config()
        connection = tool._create_connection(db_config)

        assert isinstance(connection, PostgreSQLConnection)
        assert connection.config == db_config

    @pytest.mark.asyncio
    async def test_get_schema(self, tool):
        """Test schema retrieval."""
        schema = await tool.get_schema()

        assert schema.name == "postgresql"
        assert "execute_query" in schema.actions
        assert "create_database" in schema.actions
        assert "create_table" in schema.actions
        assert "create_user" in schema.actions

    @pytest.mark.asyncio
    @patch.object(PostgreSQLTool, "execute_query")
    async def test_execute_query_action(self, mock_execute, tool):
        """Test execute query action."""
        mock_result = QueryResult(
            success=True, rows_returned=1, data=[{"id": 1}], execution_time=0.1
        )
        mock_execute.return_value = mock_result

        result = await tool._execute_query_action(
            {"query": "SELECT * FROM table", "params": {}, "timeout": 30}
        )

        assert result["success"]
        assert result["rows_returned"] == 1
        assert result["data"] == [{"id": 1}]

    @pytest.mark.asyncio
    @patch.object(PostgreSQLTool, "execute_query")
    async def test_create_database_action(self, mock_execute, tool):
        """Test create database action."""
        mock_result = QueryResult(success=True)
        mock_execute.return_value = mock_result

        result = await tool._create_database_action(
            {"database_name": "new_db", "owner": "test_user", "encoding": "UTF8"}
        )

        assert result["success"]
        assert result["database_name"] == "new_db"
        assert "created successfully" in result["message"]

    @pytest.mark.asyncio
    @patch.object(PostgreSQLTool, "execute_query")
    async def test_create_table_action(self, mock_execute, tool):
        """Test create table action."""
        mock_result = QueryResult(success=True)
        mock_execute.return_value = mock_result

        result = await tool._create_table_action(
            {
                "table_name": "users",
                "columns": [
                    {"name": "id", "type": "SERIAL PRIMARY KEY"},
                    {"name": "name", "type": "VARCHAR(100)", "not_null": True},
                ],
            }
        )

        assert result["success"]
        assert result["table_name"] == "users"

    @pytest.mark.asyncio
    @patch.object(PostgreSQLTool, "execute_query")
    async def test_create_user_action(self, mock_execute, tool):
        """Test create user action."""
        mock_result = QueryResult(success=True)
        mock_execute.return_value = mock_result

        result = await tool._create_user_action(
            {
                "username": "new_user",
                "password": "password123",
                "permissions": ["LOGIN"],
                "databases": ["test_db"],
            }
        )

        assert result["success"]
        assert result["username"] == "new_user"
        assert "created successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_estimate_cost(self, tool):
        """Test cost estimation."""
        cost = await tool.estimate_cost(
            "execute_query", {"query": "SELECT * FROM table"}
        )

        assert cost.estimated_cost > 0
        assert cost.currency == "USD"
        assert 0 <= cost.confidence <= 1
