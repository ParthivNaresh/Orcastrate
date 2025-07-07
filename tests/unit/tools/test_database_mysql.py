"""
Unit tests for MySQL database tool.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.base import ToolConfig, ToolError
from src.tools.database.base import DatabaseConfig, QueryResult
from src.tools.database.mysql import MySQLConnection, MySQLTool


class TestMySQLConnection:
    """Test MySQL connection implementation."""

    @pytest.fixture
    def db_config(self):
        """Create test database configuration."""
        return DatabaseConfig(
            host="localhost",
            port=3306,
            database="test_db",
            username="test_user",
            password="test_pass",
            min_connections=1,
            max_connections=5,
        )

    @pytest.fixture
    def connection(self, db_config):
        """Create MySQL connection instance."""
        return MySQLConnection(db_config)

    @pytest.mark.asyncio
    async def test_connection_initialization(self, connection, db_config):
        """Test connection initialization."""
        assert connection.config == db_config
        assert connection.connection_id is not None
        assert connection._connection is None
        assert not connection.is_connected

    @pytest.mark.asyncio
    @patch("src.tools.database.mysql.aiomysql.connect", new_callable=AsyncMock)
    async def test_connect_success(self, mock_connect, connection):
        """Test successful connection."""
        mock_conn = MagicMock()  # Use MagicMock for connection object
        mock_connect.return_value = mock_conn

        await connection.connect()

        assert connection.is_connected
        assert connection._connection == mock_conn
        mock_connect.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.tools.database.mysql.aiomysql.connect")
    async def test_connect_failure(self, mock_connect, connection):
        """Test connection failure."""
        mock_connect.side_effect = Exception("Connection failed")

        with pytest.raises(ToolError, match="MySQL connection failed"):
            await connection.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, connection):
        """Test disconnection."""
        mock_conn = MagicMock()
        connection._connection = mock_conn

        await connection.disconnect()

        mock_conn.close.assert_called_once()
        assert connection._connection is None

    @pytest.mark.asyncio
    async def test_execute_query_select(self, connection):
        """Test SELECT query execution."""
        mock_conn = MagicMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = [{"id": 1, "name": "test"}]
        mock_cursor.rowcount = 0
        # Mock the context manager properly
        mock_conn.cursor.return_value.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__aexit__ = AsyncMock(return_value=None)
        connection._connection = mock_conn

        result = await connection.execute_query("SELECT * FROM table")

        assert result.success
        assert result.rows_returned == 1
        assert result.data == [{"id": 1, "name": "test"}]

    @pytest.mark.asyncio
    async def test_execute_query_insert(self, connection):
        """Test INSERT query execution."""
        mock_conn = MagicMock()
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 1
        mock_cursor.lastrowid = 123
        # Mock the context manager properly
        mock_conn.cursor.return_value.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__aexit__ = AsyncMock(return_value=None)
        connection._connection = mock_conn

        result = await connection.execute_query("INSERT INTO table VALUES (1, 'test')")

        assert result.success
        assert result.rows_affected == 1
        assert result.metadata["last_insert_id"] == 123

    @pytest.mark.asyncio
    async def test_execute_query_error(self, connection):
        """Test query execution error."""
        mock_conn = MagicMock()
        mock_cursor = AsyncMock()
        mock_cursor.execute.side_effect = Exception("Query failed")
        # Mock the context manager properly
        mock_conn.cursor.return_value.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__aexit__ = AsyncMock(return_value=None)
        connection._connection = mock_conn

        result = await connection.execute_query("SELECT * FROM table")

        assert not result.success
        assert "Query failed" in result.error

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, connection):
        """Test health check when connection is healthy."""
        mock_conn = AsyncMock()
        mock_conn.ping.return_value = None  # ping() returns None on success
        connection._connection = mock_conn

        is_healthy = await connection.health_check()

        assert is_healthy

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, connection):
        """Test health check when connection is unhealthy."""
        connection._connection = None

        is_healthy = await connection.health_check()

        assert not is_healthy

    @pytest.mark.asyncio
    async def test_begin_transaction(self, connection):
        """Test transaction begin."""
        mock_conn = MagicMock()
        mock_cursor = AsyncMock()
        # Mock the context manager properly for cursor
        mock_conn.cursor.return_value.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__aexit__ = AsyncMock(return_value=None)
        # Mock other async methods
        mock_conn.select_db = AsyncMock()
        mock_conn.begin = AsyncMock()
        connection._connection = mock_conn

        context = await connection.begin_transaction("read_committed")

        assert context is not None
        assert connection._transaction_active
        mock_conn.begin.assert_called_once()

    @pytest.mark.asyncio
    async def test_commit_transaction(self, connection):
        """Test transaction commit."""
        mock_conn = AsyncMock()
        connection._connection = mock_conn
        connection._transaction_active = True

        await connection.commit_transaction()

        assert not connection._transaction_active
        mock_conn.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback_transaction(self, connection):
        """Test transaction rollback."""
        mock_conn = AsyncMock()
        connection._connection = mock_conn
        connection._transaction_active = True

        await connection.rollback_transaction()

        assert not connection._transaction_active
        mock_conn.rollback.assert_called_once()


class TestMySQLTool:
    """Test MySQL tool implementation."""

    @pytest.fixture
    def tool_config(self):
        """Create test tool configuration."""
        return ToolConfig(
            name="mysql",
            version="1.0.0",
            environment={
                "host": "localhost",
                "port": 3306,
                "database": "test_db",
                "username": "test_user",
                "password": "test_pass",
            },
        )

    @pytest.fixture
    def tool(self, tool_config):
        """Create MySQL tool instance."""
        return MySQLTool(tool_config)

    def test_tool_initialization(self, tool, tool_config):
        """Test tool initialization."""
        assert tool.config == tool_config
        assert tool.database_type.value == "mysql"

    def test_create_db_config(self, tool):
        """Test database config creation."""
        db_config = tool._create_db_config()

        assert db_config.host == "localhost"
        assert db_config.port == 3306
        assert db_config.database == "test_db"
        assert db_config.username == "test_user"
        assert db_config.password == "test_pass"

    def test_create_connection(self, tool):
        """Test connection creation."""
        db_config = tool._create_db_config()
        connection = tool._create_connection(db_config)

        assert isinstance(connection, MySQLConnection)
        assert connection.config == db_config

    @pytest.mark.asyncio
    async def test_get_schema(self, tool):
        """Test schema retrieval."""
        schema = await tool.get_schema()

        assert schema.name == "mysql"
        assert "execute_query" in schema.actions
        assert "create_database" in schema.actions
        assert "create_table" in schema.actions
        assert "create_user" in schema.actions
        assert "optimize_table" in schema.actions

    @pytest.mark.asyncio
    @patch.object(MySQLTool, "execute_query")
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
    @patch.object(MySQLTool, "execute_query")
    async def test_create_database_action(self, mock_execute, tool):
        """Test create database action."""
        mock_result = QueryResult(success=True)
        mock_execute.return_value = mock_result

        result = await tool._create_database_action(
            {
                "database_name": "new_db",
                "charset": "utf8mb4",
                "collation": "utf8mb4_unicode_ci",
            }
        )

        assert result["success"]
        assert result["database_name"] == "new_db"
        assert result["charset"] == "utf8mb4"
        assert "created successfully" in result["message"]

    @pytest.mark.asyncio
    @patch.object(MySQLTool, "execute_query")
    async def test_create_table_action(self, mock_execute, tool):
        """Test create table action."""
        mock_result = QueryResult(success=True)
        mock_execute.return_value = mock_result

        result = await tool._create_table_action(
            {
                "table_name": "users",
                "columns": [
                    {"name": "id", "type": "INT AUTO_INCREMENT PRIMARY KEY"},
                    {"name": "name", "type": "VARCHAR(100)", "not_null": True},
                ],
                "engine": "InnoDB",
            }
        )

        assert result["success"]
        assert result["table_name"] == "users"
        assert result["engine"] == "InnoDB"

    @pytest.mark.asyncio
    @patch.object(MySQLTool, "execute_query")
    async def test_create_user_action(self, mock_execute, tool):
        """Test create user action."""
        mock_result = QueryResult(success=True)
        mock_execute.return_value = mock_result

        result = await tool._create_user_action(
            {
                "username": "new_user",
                "password": "password123",
                "host": "localhost",
                "permissions": ["SELECT", "INSERT"],
                "databases": ["test_db"],
            }
        )

        assert result["success"]
        assert result["username"] == "new_user"
        assert result["host"] == "localhost"
        assert "created successfully" in result["message"]

    @pytest.mark.asyncio
    @patch.object(MySQLTool, "execute_query")
    async def test_optimize_table_action(self, mock_execute, tool):
        """Test optimize table action."""
        mock_result = QueryResult(success=True, data=[{"Msg_text": "OK"}])
        mock_execute.return_value = mock_result

        result = await tool._optimize_table_action({"table_name": "users"})

        assert result["success"]
        assert result["table_name"] == "users"
        assert "optimized successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_estimate_cost(self, tool):
        """Test cost estimation."""
        cost = await tool.estimate_cost(
            "execute_query", {"query": "SELECT * FROM table"}
        )

        assert cost.estimated_cost > 0
        assert cost.currency == "USD"
        assert 0 <= cost.confidence <= 1
