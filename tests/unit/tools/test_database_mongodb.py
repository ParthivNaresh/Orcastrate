"""
Unit tests for MongoDB database tool.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId

from src.tools.base import ToolConfig, ToolError
from src.tools.database.base import DatabaseConfig, QueryResult
from src.tools.database.mongodb import MongoDBConnection, MongoDBTool


class TestMongoDBConnection:
    """Test MongoDB connection implementation."""

    @pytest.fixture
    def db_config(self):
        """Create test database configuration."""
        return DatabaseConfig(
            host="localhost",
            port=27017,
            database="test_db",
            username="test_user",
            password="test_pass",
            min_connections=1,
            max_connections=5,
        )

    @pytest.fixture
    def connection(self, db_config):
        """Create MongoDB connection instance."""
        return MongoDBConnection(db_config)

    @pytest.mark.asyncio
    async def test_connection_initialization(self, connection, db_config):
        """Test connection initialization."""
        assert connection.config == db_config
        assert connection.connection_id is not None
        assert connection._client is None
        assert connection._database is None
        assert not connection.is_connected

    @pytest.mark.asyncio
    async def test_build_uri(self, connection):
        """Test URI building."""
        uri = connection._build_uri()
        assert "mongodb://test_user:test_pass@localhost:27017/test_db" in uri

    @pytest.mark.asyncio
    async def test_build_uri_with_ssl(self, db_config):
        """Test URI building with SSL options."""
        db_config.ssl_enabled = True
        db_config.ssl_cert_path = "/path/to/cert.pem"
        db_config.ssl_ca_path = "/path/to/ca.pem"

        connection = MongoDBConnection(db_config)
        uri = connection._build_uri()

        assert "ssl=true" in uri
        assert "tlsCertificateKeyFile=/path/to/cert.pem" in uri
        assert "tlsCAFile=/path/to/ca.pem" in uri

    @pytest.mark.asyncio
    @patch("src.tools.database.mongodb.AsyncIOMotorClient")
    async def test_connect_success(self, mock_client_class, connection):
        """Test successful connection."""
        mock_client = AsyncMock()
        mock_database = AsyncMock()
        mock_admin = AsyncMock()

        mock_client_class.return_value = mock_client
        mock_client.__getitem__.return_value = mock_database
        mock_client.admin = mock_admin
        mock_admin.command = AsyncMock()

        await connection.connect()

        assert connection.is_connected
        assert connection._client == mock_client
        assert connection._database == mock_database
        mock_admin.command.assert_called_once_with("ping")

    @pytest.mark.asyncio
    @patch("src.tools.database.mongodb.AsyncIOMotorClient")
    async def test_connect_failure(self, mock_client_class, connection):
        """Test connection failure."""
        mock_client_class.side_effect = Exception("Connection failed")

        with pytest.raises(ToolError, match="MongoDB connection failed"):
            await connection.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, connection):
        """Test disconnection."""
        mock_client = MagicMock()
        connection._client = mock_client

        await connection.disconnect()

        mock_client.close.assert_called_once()
        assert connection._client is None
        assert connection._database is None

    @pytest.mark.asyncio
    async def test_serialize_document(self, connection):
        """Test document serialization."""
        doc = {
            "_id": ObjectId("507f1f77bcf86cd799439011"),
            "name": "test",
            "nested": {"obj_id": ObjectId("507f1f77bcf86cd799439012")},
            "list": [ObjectId("507f1f77bcf86cd799439013"), "string"],
        }

        serialized = connection._serialize_document(doc)

        assert isinstance(serialized["_id"], str)
        assert serialized["name"] == "test"
        assert isinstance(serialized["nested"]["obj_id"], str)
        assert isinstance(serialized["list"][0], str)
        assert serialized["list"][1] == "string"

    @pytest.mark.asyncio
    async def test_find_documents(self, connection):
        """Test finding documents."""
        mock_database = MagicMock()  # Use MagicMock for database access
        mock_collection = MagicMock()  # Use MagicMock for collection
        mock_cursor = MagicMock()  # Use MagicMock for cursor since it's not awaited

        connection._database = mock_database
        mock_database.__getitem__.return_value = mock_collection
        mock_collection.find.return_value = mock_cursor
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.skip.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        # Mock to_list as async method
        mock_cursor.to_list = AsyncMock(
            return_value=[{"_id": ObjectId(), "name": "test"}]
        )

        result = await connection.find_documents(
            "test_collection", {"name": "test"}, sort=[("name", 1)], limit=10
        )

        assert result.success
        assert result.rows_returned == 1
        assert len(result.data) == 1
        mock_collection.find.assert_called_once_with({"name": "test"}, None)

    @pytest.mark.asyncio
    async def test_insert_document(self, connection):
        """Test inserting a document."""
        mock_database = AsyncMock()
        mock_collection = AsyncMock()
        mock_result = MagicMock()
        mock_result.inserted_id = ObjectId()

        connection._database = mock_database
        mock_database.__getitem__.return_value = mock_collection
        mock_collection.insert_one.return_value = mock_result

        result = await connection.insert_document("test_collection", {"name": "test"})

        assert result.success
        assert result.rows_affected == 1
        assert "inserted_id" in result.metadata
        mock_collection.insert_one.assert_called_once_with({"name": "test"})

    @pytest.mark.asyncio
    async def test_update_documents(self, connection):
        """Test updating documents."""
        mock_database = AsyncMock()
        mock_collection = AsyncMock()
        mock_result = MagicMock()
        mock_result.modified_count = 2
        mock_result.matched_count = 2

        connection._database = mock_database
        mock_database.__getitem__.return_value = mock_collection
        mock_collection.update_many.return_value = mock_result

        result = await connection.update_documents(
            "test_collection", {"status": "old"}, {"$set": {"status": "new"}}, many=True
        )

        assert result.success
        assert result.rows_affected == 2
        assert result.metadata["matched_count"] == 2
        mock_collection.update_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_documents(self, connection):
        """Test deleting documents."""
        mock_database = AsyncMock()
        mock_collection = AsyncMock()
        mock_result = MagicMock()
        mock_result.deleted_count = 3

        connection._database = mock_database
        mock_database.__getitem__.return_value = mock_collection
        mock_collection.delete_many.return_value = mock_result

        result = await connection.delete_documents(
            "test_collection", {"status": "old"}, many=True
        )

        assert result.success
        assert result.rows_affected == 3
        mock_collection.delete_many.assert_called_once_with({"status": "old"})

    @pytest.mark.asyncio
    async def test_aggregate_documents(self, connection):
        """Test aggregation pipeline."""
        mock_database = MagicMock()  # Use MagicMock for database access
        mock_collection = MagicMock()  # Use MagicMock for collection
        mock_cursor = MagicMock()  # Use MagicMock for cursor

        connection._database = mock_database
        mock_database.__getitem__.return_value = mock_collection
        mock_collection.aggregate.return_value = mock_cursor
        # Mock to_list as async method
        mock_cursor.to_list = AsyncMock(return_value=[{"count": 5}])

        pipeline = [{"$group": {"_id": None, "count": {"$sum": 1}}}]
        result = await connection.aggregate_documents("test_collection", pipeline)

        assert result.success
        assert result.rows_returned == 1
        assert result.data == [{"count": 5}]
        mock_collection.aggregate.assert_called_once_with(pipeline)

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, connection):
        """Test health check when connection is healthy."""
        mock_client = AsyncMock()
        mock_admin = AsyncMock()

        connection._client = mock_client
        mock_client.admin = mock_admin
        mock_admin.command = AsyncMock()

        is_healthy = await connection.health_check()

        assert is_healthy
        mock_admin.command.assert_called_once_with("ping")

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, connection):
        """Test health check when connection is unhealthy."""
        connection._client = None

        is_healthy = await connection.health_check()

        assert not is_healthy


class TestMongoDBTool:
    """Test MongoDB tool implementation."""

    @pytest.fixture
    def tool_config(self):
        """Create test tool configuration."""
        return ToolConfig(
            name="mongodb",
            version="1.0.0",
            environment={
                "host": "localhost",
                "port": 27017,
                "database": "test_db",
                "username": "test_user",
                "password": "test_pass",
            },
        )

    @pytest.fixture
    def tool(self, tool_config):
        """Create MongoDB tool instance."""
        return MongoDBTool(tool_config)

    def test_tool_initialization(self, tool, tool_config):
        """Test tool initialization."""
        assert tool.config == tool_config
        assert tool.database_type.value == "mongodb"

    def test_create_db_config(self, tool):
        """Test database config creation."""
        db_config = tool._create_db_config()

        assert db_config.host == "localhost"
        assert db_config.port == 27017
        assert db_config.database == "test_db"
        assert db_config.username == "test_user"
        assert db_config.password == "test_pass"

    def test_create_connection(self, tool):
        """Test connection creation."""
        db_config = tool._create_db_config()
        connection = tool._create_connection(db_config)

        assert isinstance(connection, MongoDBConnection)
        assert connection.config == db_config

    @pytest.mark.asyncio
    async def test_get_schema(self, tool):
        """Test schema retrieval."""
        schema = await tool.get_schema()

        assert schema.name == "mongodb"
        assert "find_documents" in schema.actions
        assert "insert_document" in schema.actions
        assert "update_document" in schema.actions
        assert "delete_document" in schema.actions
        assert "aggregate" in schema.actions
        assert "create_collection" in schema.actions
        assert "create_user" in schema.actions

    @pytest.mark.asyncio
    async def test_find_documents_action(self, tool):
        """Test find documents action."""
        mock_pool = MagicMock()
        mock_connection = AsyncMock()
        mock_result = QueryResult(
            success=True,
            rows_returned=1,
            data=[{"_id": "507f1f77bcf86cd799439011", "name": "test"}],
        )

        tool.connection_pool = mock_pool
        # Mock the context manager properly
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_connection
        )
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.find_documents.return_value = mock_result

        result = await tool._find_documents_action(
            {"collection": "users", "filter": {"name": "test"}, "limit": 10}
        )

        assert result["success"]
        assert result["count"] == 1
        assert len(result["documents"]) == 1

    @pytest.mark.asyncio
    async def test_insert_document_action(self, tool):
        """Test insert document action."""
        mock_pool = MagicMock()
        mock_connection = AsyncMock()
        mock_result = QueryResult(
            success=True,
            rows_affected=1,
            metadata={"inserted_id": "507f1f77bcf86cd799439011"},
        )

        tool.connection_pool = mock_pool
        # Mock the context manager properly
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_connection
        )
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.insert_document.return_value = mock_result

        result = await tool._insert_document_action(
            {
                "collection": "users",
                "document": {"name": "test", "email": "test@example.com"},
            }
        )

        assert result["success"]
        assert result["inserted_id"] == "507f1f77bcf86cd799439011"

    @pytest.mark.asyncio
    async def test_create_user_action(self, tool):
        """Test create user action."""
        mock_pool = MagicMock()
        mock_connection = AsyncMock()
        mock_client = AsyncMock()
        mock_admin_db = AsyncMock()

        tool.connection_pool = mock_pool
        tool.db_config = MagicMock()
        tool.db_config.database = "test_db"

        # Mock the context manager properly
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_connection
        )
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection._database = MagicMock()
        mock_connection._client = mock_client
        mock_client.__getitem__.return_value = mock_admin_db
        mock_admin_db.command = AsyncMock()

        result = await tool._create_user_action(
            {
                "username": "new_user",
                "password": "password123",
                "roles": ["read", "readWrite"],
                "database": "test_db",
            }
        )

        assert result["success"]
        assert result["username"] == "new_user"
        assert "created successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_estimate_cost(self, tool):
        """Test cost estimation."""
        cost = await tool.estimate_cost("find_documents", {"collection": "users"})

        assert cost.estimated_cost > 0
        assert cost.currency == "USD"
        assert 0 <= cost.confidence <= 1

    @pytest.mark.asyncio
    async def test_estimate_cost_complex_query(self, tool):
        """Test cost estimation for complex operations."""
        cost = await tool.estimate_cost(
            "aggregate",
            {
                "collection": "users",
                "pipeline": [{"$group": {"_id": "$status", "count": {"$sum": 1}}}],
            },
        )

        assert cost.estimated_cost > 0
        assert cost.currency == "USD"
