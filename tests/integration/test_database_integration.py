"""
Integration tests for database tools.

These tests use testcontainers to run real database instances for comprehensive testing.
"""

import asyncio
import time

import pytest
from testcontainers.mongodb import MongoDbContainer
from testcontainers.mysql import MySqlContainer
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

from src.tools.base import ToolConfig
from src.tools.database.mongodb import MongoDBTool
from src.tools.database.mysql import MySQLTool
from src.tools.database.postgresql import PostgreSQLTool
from src.tools.database.redis import RedisTool


@pytest.mark.integration
@pytest.mark.docker_required
class TestPostgreSQLIntegration:
    """Integration tests for PostgreSQL tool."""

    @pytest.fixture(scope="class")
    def postgres_container(self):
        """Start PostgreSQL container for testing."""
        container = PostgresContainer("postgres:15-alpine")
        container.start()
        yield container
        container.stop()

    @pytest.fixture
    async def postgres_tool(self, postgres_container):
        """Create PostgreSQL tool with container connection."""
        config = ToolConfig(
            name="postgresql",
            version="1.0.0",
            environment={
                "host": postgres_container.get_container_host_ip(),
                "port": postgres_container.get_exposed_port(5432),
                "database": postgres_container.dbname,
                "username": postgres_container.username,
                "password": postgres_container.password,
                "min_connections": 1,
                "max_connections": 3,
            },
        )

        tool = PostgreSQLTool(config)
        await tool.initialize()
        yield tool
        await tool.cleanup()

    @pytest.mark.asyncio
    async def test_connection_and_basic_queries(self, postgres_tool):
        """Test basic PostgreSQL operations."""
        # Test simple query
        result = await postgres_tool.execute_query("SELECT 1 as test_value")
        assert result.success
        assert result.data[0]["test_value"] == 1

        # Test current database info
        result = await postgres_tool.execute_query("SELECT current_database()")
        assert result.success
        assert result.data[0]["current_database"] is not None

    @pytest.mark.asyncio
    async def test_database_creation(self, postgres_tool):
        """Test database creation."""
        db_name = f"test_db_{int(time.time())}"

        result = await postgres_tool._create_database_action(
            {"database_name": db_name, "encoding": "UTF8"}
        )

        assert result["success"]
        assert result["database_name"] == db_name

        # Verify database exists
        check_result = await postgres_tool.execute_query(
            "SELECT 1 FROM pg_database WHERE datname = %s", {"datname": db_name}
        )
        assert check_result.success
        assert len(check_result.data) == 1

    @pytest.mark.asyncio
    async def test_table_operations(self, postgres_tool):
        """Test table creation and manipulation."""
        table_name = f"test_table_{int(time.time())}"

        # Create table
        result = await postgres_tool._create_table_action(
            {
                "table_name": table_name,
                "columns": [
                    {"name": "id", "type": "SERIAL PRIMARY KEY"},
                    {"name": "name", "type": "VARCHAR(100)", "not_null": True},
                    {"name": "email", "type": "VARCHAR(255)"},
                ],
            }
        )

        assert result["success"]
        assert result["table_name"] == table_name

        # Insert data
        insert_result = await postgres_tool.execute_query(
            f'INSERT INTO "{table_name}" (name, email) VALUES (%s, %s) RETURNING id',
            {"name": "John Doe", "email": "john@example.com"},
        )
        assert insert_result.success
        assert insert_result.rows_affected == 1

        # Query data
        select_result = await postgres_tool.execute_query(
            f'SELECT * FROM "{table_name}" WHERE name = %s', {"name": "John Doe"}
        )
        assert select_result.success
        assert len(select_result.data) == 1
        assert select_result.data[0]["name"] == "John Doe"

    @pytest.mark.asyncio
    async def test_user_management(self, postgres_tool):
        """Test user creation and permissions."""
        username = f"test_user_{int(time.time())}"

        # Create user
        result = await postgres_tool._create_user_action(
            {
                "username": username,
                "password": "test_password_123",
                "permissions": ["LOGIN"],
            }
        )

        assert result["success"]
        assert result["username"] == username

        # Verify user exists
        check_result = await postgres_tool.execute_query(
            "SELECT 1 FROM pg_user WHERE usename = %s", {"usename": username}
        )
        assert check_result.success
        assert len(check_result.data) == 1

    @pytest.mark.asyncio
    async def test_performance_analysis(self, postgres_tool):
        """Test query performance analysis."""
        # Create a test table first
        await postgres_tool.execute_query(
            "CREATE TABLE IF NOT EXISTS perf_test (id SERIAL, data TEXT)"
        )

        result = await postgres_tool._analyze_performance_action(
            {"query": "SELECT COUNT(*) FROM perf_test"}
        )

        assert result["success"]
        assert "execution_time" in result
        assert "planning_time" in result
        assert isinstance(result["recommendations"], list)

    @pytest.mark.asyncio
    async def test_connection_pooling(self, postgres_tool):
        """Test connection pooling functionality."""
        # Execute multiple concurrent queries
        tasks = []
        for i in range(5):
            task = postgres_tool.execute_query(f"SELECT {i} as query_id")
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All queries should succeed
        for i, result in enumerate(results):
            assert result.success
            assert result.data[0]["query_id"] == i

        # Check pool statistics
        stats = postgres_tool.performance_stats
        assert stats["query_count"] >= 5
        assert "pool" in stats


@pytest.mark.integration
@pytest.mark.docker_required
class TestMySQLIntegration:
    """Integration tests for MySQL tool."""

    @pytest.fixture(scope="class")
    def mysql_container(self):
        """Start MySQL container for testing."""
        container = MySqlContainer("mysql:8.0")
        container.start()
        yield container
        container.stop()

    @pytest.fixture
    async def mysql_tool(self, mysql_container):
        """Create MySQL tool with container connection."""
        config = ToolConfig(
            name="mysql",
            version="1.0.0",
            environment={
                "host": mysql_container.get_container_host_ip(),
                "port": mysql_container.get_exposed_port(3306),
                "database": mysql_container.dbname,
                "username": mysql_container.username,
                "password": mysql_container.password,
                "min_connections": 1,
                "max_connections": 3,
            },
        )

        tool = MySQLTool(config)
        await tool.initialize()
        yield tool
        await tool.cleanup()

    @pytest.mark.asyncio
    async def test_connection_and_basic_queries(self, mysql_tool):
        """Test basic MySQL operations."""
        # Test simple query
        result = await mysql_tool.execute_query("SELECT 1 as test_value")
        assert result.success
        assert result.data[0]["test_value"] == 1

        # Test MySQL version
        result = await mysql_tool.execute_query("SELECT VERSION() as version")
        assert result.success
        assert "8.0" in result.data[0]["version"]

    @pytest.mark.asyncio
    async def test_database_creation(self, mysql_tool):
        """Test database creation."""
        db_name = f"test_db_{int(time.time())}"

        result = await mysql_tool._create_database_action(
            {
                "database_name": db_name,
                "charset": "utf8mb4",
                "collation": "utf8mb4_unicode_ci",
            }
        )

        assert result["success"]
        assert result["database_name"] == db_name
        assert result["charset"] == "utf8mb4"

    @pytest.mark.asyncio
    async def test_table_operations(self, mysql_tool):
        """Test table creation and manipulation."""
        table_name = f"test_table_{int(time.time())}"

        # Create table
        result = await mysql_tool._create_table_action(
            {
                "table_name": table_name,
                "columns": [
                    {
                        "name": "id",
                        "type": "INT",
                        "auto_increment": True,
                        "primary_key": True,
                    },
                    {"name": "name", "type": "VARCHAR(100)", "not_null": True},
                    {"name": "email", "type": "VARCHAR(255)"},
                ],
                "engine": "InnoDB",
            }
        )

        assert result["success"]
        assert result["table_name"] == table_name

        # Insert data
        insert_result = await mysql_tool.execute_query(
            f"INSERT INTO `{table_name}` (name, email) VALUES (%s, %s)",
            {"name": "Jane Doe", "email": "jane@example.com"},
        )
        assert insert_result.success
        assert insert_result.rows_affected == 1

        # Query data
        select_result = await mysql_tool.execute_query(
            f"SELECT * FROM `{table_name}` WHERE name = %s", {"name": "Jane Doe"}
        )
        assert select_result.success
        assert len(select_result.data) == 1
        assert select_result.data[0]["name"] == "Jane Doe"

    @pytest.mark.asyncio
    async def test_user_management(self, mysql_tool):
        """Test user creation and permissions."""
        username = f"test_user_{int(time.time())}"

        # Create user
        result = await mysql_tool._create_user_action(
            {
                "username": username,
                "password": "test_password_123",
                "host": "%",
                "permissions": ["SELECT", "INSERT"],
                "databases": ["test"],
            }
        )

        assert result["success"]
        assert result["username"] == username

    @pytest.mark.asyncio
    async def test_table_optimization(self, mysql_tool):
        """Test table optimization features."""
        # Create a test table
        table_name = "optimization_test"
        await mysql_tool.execute_query(
            f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                data VARCHAR(255)
            ) ENGINE=InnoDB
        """
        )

        # Test table optimization
        result = await mysql_tool._optimize_table_action({"table_name": table_name})

        assert result["success"]
        assert result["table_name"] == table_name

        # Test table analysis
        analyze_result = await mysql_tool._analyze_table_action(
            {"table_name": table_name}
        )

        assert analyze_result["success"]
        assert analyze_result["table_name"] == table_name


@pytest.mark.integration
@pytest.mark.docker_required
class TestMongoDBIntegration:
    """Integration tests for MongoDB tool."""

    @pytest.fixture(scope="class")
    def mongodb_container(self):
        """Start MongoDB container for testing."""
        container = MongoDbContainer("mongo:7.0")
        container.start()
        yield container
        container.stop()

    @pytest.fixture
    async def mongodb_tool(self, mongodb_container):
        """Create MongoDB tool with container connection."""
        config = ToolConfig(
            name="mongodb",
            version="1.0.0",
            environment={
                "host": mongodb_container.get_container_host_ip(),
                "port": mongodb_container.get_exposed_port(27017),
                "database": "test_db",
                "username": "",
                "password": "",
                "min_connections": 1,
                "max_connections": 3,
            },
        )

        tool = MongoDBTool(config)
        await tool.initialize()
        yield tool
        await tool.cleanup()

    @pytest.mark.asyncio
    async def test_connection_and_basic_operations(self, mongodb_tool):
        """Test basic MongoDB operations."""
        collection_name = f"test_collection_{int(time.time())}"

        # Insert document
        result = await mongodb_tool._insert_document_action(
            {
                "collection": collection_name,
                "document": {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "age": 30,
                },
            }
        )

        assert result["success"]
        assert result["inserted_id"] is not None

        # Find documents
        find_result = await mongodb_tool._find_documents_action(
            {"collection": collection_name, "filter": {"name": "John Doe"}}
        )

        assert find_result["success"]
        assert find_result["count"] == 1
        assert find_result["documents"][0]["name"] == "John Doe"

    @pytest.mark.asyncio
    async def test_bulk_operations(self, mongodb_tool):
        """Test bulk insert and operations."""
        collection_name = f"bulk_test_{int(time.time())}"

        # Insert multiple documents
        documents = [
            {"name": f"User {i}", "age": 20 + i, "status": "active"} for i in range(10)
        ]

        result = await mongodb_tool._insert_many_action(
            {"collection": collection_name, "documents": documents}
        )

        assert result["success"]
        assert result["inserted_count"] == 10

        # Update many documents
        update_result = await mongodb_tool._update_many_action(
            {
                "collection": collection_name,
                "filter": {"age": {"$gte": 25}},
                "update": {"$set": {"status": "senior"}},
            }
        )

        assert update_result["success"]
        assert update_result["modified_count"] >= 5

        # Find updated documents
        find_result = await mongodb_tool._find_documents_action(
            {"collection": collection_name, "filter": {"status": "senior"}}
        )

        assert find_result["success"]
        assert find_result["count"] >= 5

    @pytest.mark.asyncio
    async def test_aggregation_pipeline(self, mongodb_tool):
        """Test MongoDB aggregation pipeline."""
        collection_name = f"agg_test_{int(time.time())}"

        # Insert test data
        documents = [
            {"category": "A", "value": 10},
            {"category": "A", "value": 20},
            {"category": "B", "value": 15},
            {"category": "B", "value": 25},
        ]

        await mongodb_tool._insert_many_action(
            {"collection": collection_name, "documents": documents}
        )

        # Run aggregation
        pipeline = [
            {
                "$group": {
                    "_id": "$category",
                    "total": {"$sum": "$value"},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id": 1}},
        ]

        result = await mongodb_tool._aggregate_action(
            {"collection": collection_name, "pipeline": pipeline}
        )

        assert result["success"]
        assert result["count"] == 2

        # Check aggregation results
        agg_data = result["documents"]
        category_a = next(doc for doc in agg_data if doc["_id"] == "A")
        category_b = next(doc for doc in agg_data if doc["_id"] == "B")

        assert category_a["total"] == 30
        assert category_a["count"] == 2
        assert category_b["total"] == 40
        assert category_b["count"] == 2

    @pytest.mark.asyncio
    async def test_collection_management(self, mongodb_tool):
        """Test collection creation and indexing."""
        collection_name = f"managed_collection_{int(time.time())}"

        # Create collection
        result = await mongodb_tool._create_collection_action(
            {"collection": collection_name, "options": {}}
        )

        assert result["success"]
        assert result["collection"] == collection_name

        # Create index
        index_result = await mongodb_tool._create_index_action(
            {
                "collection": collection_name,
                "keys": {"name": 1, "email": 1},
                "options": {"unique": True},
            }
        )

        assert index_result["success"]
        assert index_result["collection"] == collection_name

    @pytest.mark.asyncio
    async def test_statistics(self, mongodb_tool):
        """Test database statistics retrieval."""
        result = await mongodb_tool._get_statistics_action({})

        assert result["success"]
        assert "statistics" in result
        assert "database_info" in result["statistics"]
        assert "tool_stats" in result["statistics"]


@pytest.mark.integration
@pytest.mark.docker_required
class TestRedisIntegration:
    """Integration tests for Redis tool."""

    @pytest.fixture(scope="class")
    def redis_container(self):
        """Start Redis container for testing."""
        container = RedisContainer("redis:7-alpine")
        container.start()
        yield container
        container.stop()

    @pytest.fixture
    async def redis_tool(self, redis_container):
        """Create Redis tool with container connection."""
        config = ToolConfig(
            name="redis",
            version="1.0.0",
            environment={
                "host": redis_container.get_container_host_ip(),
                "port": redis_container.get_exposed_port(6379),
                "database": "0",
                "username": "",
                "password": "",
                "min_connections": 1,
                "max_connections": 3,
                "database_number": 0,
            },
        )

        tool = RedisTool(config)
        await tool.initialize()
        yield tool
        await tool.cleanup()

    @pytest.mark.asyncio
    async def test_connection_and_basic_operations(self, redis_tool):
        """Test basic Redis operations."""
        key_name = f"test_key_{int(time.time())}"

        # Set value
        set_result = await redis_tool._set_action(
            {"key": key_name, "value": "test_value", "ttl": 300}
        )

        assert set_result["success"]
        assert set_result["key"] == key_name
        assert set_result["value"] == "test_value"
        assert set_result["ttl"] == 300

        # Get value
        get_result = await redis_tool._get_action({"key": key_name})

        assert get_result["success"]
        assert get_result["value"] == "test_value"
        assert get_result["found"]

        # Delete value
        delete_result = await redis_tool._delete_action({"key": key_name})

        assert delete_result["success"]
        assert delete_result["deleted"]

        # Verify deletion
        get_after_delete = await redis_tool._get_action({"key": key_name})
        assert get_after_delete["success"]
        assert not get_after_delete["found"]

    @pytest.mark.asyncio
    async def test_hash_operations(self, redis_tool):
        """Test Redis hash operations."""
        hash_key = f"test_hash_{int(time.time())}"

        # Set hash field
        hset_result = await redis_tool._hash_set_action(
            {"key": hash_key, "field": "field1", "value": "value1"}
        )

        assert hset_result["success"]
        assert hset_result["key"] == hash_key
        assert hset_result["field"] == "field1"
        assert hset_result["value"] == "value1"

        # Get hash field
        hget_result = await redis_tool._hash_get_action(
            {"key": hash_key, "field": "field1"}
        )

        assert hget_result["success"]
        assert hget_result["value"] == "value1"

        # Set multiple fields
        await redis_tool._hash_set_action(
            {"key": hash_key, "field": "field2", "value": "value2"}
        )
        await redis_tool._hash_set_action(
            {"key": hash_key, "field": "field3", "value": "value3"}
        )

        # Get all fields
        hgetall_result = await redis_tool._hash_getall_action({"key": hash_key})

        assert hgetall_result["success"]
        hash_data = hgetall_result["hash"]
        assert hash_data["field1"] == "value1"
        assert hash_data["field2"] == "value2"
        assert hash_data["field3"] == "value3"

        # Delete hash field
        hdel_result = await redis_tool._hash_delete_action(
            {"key": hash_key, "field": "field1"}
        )

        assert hdel_result["success"]
        assert hdel_result["deleted"] == 1

    @pytest.mark.asyncio
    async def test_keys_pattern_matching(self, redis_tool):
        """Test Redis KEYS pattern matching."""
        # Set multiple keys with pattern
        test_keys = [f"pattern_test_{i}" for i in range(5)]
        for key in test_keys:
            await redis_tool._set_action({"key": key, "value": f"value_{key}"})

        # Get keys matching pattern
        keys_result = await redis_tool._keys_action({"pattern": "pattern_test_*"})

        assert keys_result["success"]
        assert keys_result["count"] >= 5
        assert keys_result["pattern"] == "pattern_test_*"

        # Verify all our test keys are in the results
        returned_keys = keys_result["keys"]
        for key in test_keys:
            assert key in returned_keys

        # Clean up
        for key in test_keys:
            await redis_tool._delete_action({"key": key})

    @pytest.mark.asyncio
    async def test_command_execution(self, redis_tool):
        """Test direct Redis command execution."""
        test_key = f"cmd_test_{int(time.time())}"

        # Execute SET command
        set_result = await redis_tool._execute_command_action(
            {"command": f"SET {test_key} command_value"}
        )

        assert set_result["success"]
        assert set_result["result"] == "OK"

        # Execute GET command
        get_result = await redis_tool._execute_command_action(
            {"command": f"GET {test_key}"}
        )

        assert get_result["success"]
        assert get_result["result"] == "command_value"

        # Execute EXISTS command
        exists_result = await redis_tool._execute_command_action(
            {"command": f"EXISTS {test_key}"}
        )

        assert exists_result["success"]
        assert exists_result["result"] == 1

        # Clean up
        await redis_tool._delete_action({"key": test_key})

    @pytest.mark.asyncio
    async def test_statistics(self, redis_tool):
        """Test Redis statistics retrieval."""
        result = await redis_tool._get_statistics_action({})

        assert result["success"]
        assert "statistics" in result
        assert "tool_stats" in result

        stats = result["statistics"]
        assert "redis_version" in stats
        assert "uptime_in_seconds" in stats
        assert "connected_clients" in stats
        assert "used_memory" in stats

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, redis_tool):
        """Test concurrent Redis operations."""
        # Execute multiple concurrent operations
        tasks = []
        for i in range(10):
            task = redis_tool._set_action(
                {"key": f"concurrent_{i}", "value": f"value_{i}"}
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All operations should succeed
        for result in results:
            assert result["success"]

        # Verify all keys exist
        get_tasks = []
        for i in range(10):
            task = redis_tool._get_action({"key": f"concurrent_{i}"})
            get_tasks.append(task)

        get_results = await asyncio.gather(*get_tasks)

        for i, result in enumerate(get_results):
            assert result["success"]
            assert result["value"] == f"value_{i}"

        # Clean up
        delete_tasks = []
        for i in range(10):
            task = redis_tool._delete_action({"key": f"concurrent_{i}"})
            delete_tasks.append(task)

        await asyncio.gather(*delete_tasks)
