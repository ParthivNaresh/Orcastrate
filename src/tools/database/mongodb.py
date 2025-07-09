"""
MongoDB database tool implementation.

High-performance MongoDB connector with async operations, document management,
aggregation pipelines, and comprehensive database operations.
"""

import time
from typing import Any, Optional, cast

from bson import ObjectId
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorClientSession,
    AsyncIOMotorDatabase,
)

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


class MongoDBConnection(DatabaseConnection):
    """High-performance MongoDB connection with Motor (async PyMongo)."""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._client: Optional[AsyncIOMotorClient] = None
        self._database: Optional[AsyncIOMotorDatabase] = None
        self._session: Optional[AsyncIOMotorClientSession] = None

    async def connect(self) -> None:
        """Establish MongoDB connection."""
        try:
            self.state = ConnectionState.CONNECTING
            self.logger.debug(
                f"Connecting to MongoDB at {self.config.host}:{self.config.port}"
            )

            # Build connection URI
            uri = self._build_uri()

            # Connect with optimized settings
            self._client = AsyncIOMotorClient(
                uri,
                serverSelectionTimeoutMS=int(self.config.connection_timeout * 1000),
                connectTimeoutMS=int(self.config.connection_timeout * 1000),
                socketTimeoutMS=60000,
                maxPoolSize=self.config.max_connections,
                minPoolSize=self.config.min_connections,
                maxIdleTimeMS=int(self.config.idle_timeout * 1000),
                appname="orcastrate_db_tool",
            )

            # Get database
            self._database = self._client[self.config.database]

            # Test connection
            await self._client.admin.command("ping")

            self.state = ConnectionState.CONNECTED
            self.logger.debug(f"Connected to MongoDB {self.connection_id[:8]}")

        except Exception as e:
            self.state = ConnectionState.ERROR
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            raise ToolError(f"MongoDB connection failed: {e}")

    def _build_uri(self) -> str:
        """Build MongoDB connection URI."""
        if self.config.username and self.config.password:
            auth = f"{self.config.username}:{self.config.password}@"
        else:
            auth = ""

        uri_parts = [
            f"mongodb://{auth}{self.config.host}:{self.config.port}/{self.config.database}"
        ]

        params = []

        # Add authSource for authenticated connections
        if self.config.username and self.config.password:
            params.append("authSource=admin")

        # SSL configuration
        if self.config.ssl_enabled:
            params.append("ssl=true")
            if self.config.ssl_cert_path:
                params.append(f"tlsCertificateKeyFile={self.config.ssl_cert_path}")
            if self.config.ssl_ca_path:
                params.append(f"tlsCAFile={self.config.ssl_ca_path}")

        # Additional options
        for key, value in self.config.options.items():
            params.append(f"{key}={value}")

        if params:
            uri_parts.append("?" + "&".join(params))

        return "".join(uri_parts)

    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self.state = ConnectionState.CLOSING
            try:
                self._client.close()
                self.logger.debug(f"Disconnected from MongoDB {self.connection_id[:8]}")
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")
            finally:
                self._client = None
                self._database = None
                self.state = ConnectionState.DISCONNECTED

    async def execute_query(
        self, query: str, params: Optional[dict[str, Any]] = None
    ) -> QueryResult:
        """Execute MongoDB operation (not traditional SQL query)."""
        # MongoDB operations are handled through specific methods
        # This is kept for interface compatibility
        return QueryResult(
            success=False,
            error="Use specific MongoDB methods instead of execute_query",
        )

    async def find_documents(
        self,
        collection_name: str,
        filter_query: dict[str, Any],
        projection: Optional[dict[str, Any]] = None,
        sort: Optional[list[tuple[str, int]]] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> QueryResult:
        """Find documents in a MongoDB collection."""
        if self._database is None:
            raise ToolError("Not connected to database")

        start_time = time.time()

        try:
            collection = self._database[collection_name]

            # Build find query
            cursor = collection.find(filter_query, projection)
            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)

            # Execute and convert to list
            documents = await cursor.to_list(length=limit)

            # Convert ObjectId to string for JSON serialization
            serialized_docs = []
            for doc in documents:
                serialized_doc = self._serialize_document(doc)
                serialized_docs.append(serialized_doc)

            execution_time = time.time() - start_time

            return QueryResult(
                success=True,
                rows_returned=len(serialized_docs),
                data=serialized_docs,
                execution_time=execution_time,
                metadata={
                    "collection": collection_name,
                    "connection_id": self.connection_id,
                    "filter": filter_query,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Find operation failed: {e}")

            return QueryResult(
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"connection_id": self.connection_id},
            )

    async def insert_document(
        self,
        collection_name: str,
        document: dict[str, Any],
    ) -> QueryResult:
        """Insert a single document."""
        if self._database is None:
            raise ToolError("Not connected to database")

        start_time = time.time()

        try:
            collection = self._database[collection_name]
            result = await collection.insert_one(document)

            execution_time = time.time() - start_time

            return QueryResult(
                success=True,
                rows_affected=1,
                execution_time=execution_time,
                metadata={
                    "collection": collection_name,
                    "connection_id": self.connection_id,
                    "inserted_id": str(result.inserted_id),
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Insert operation failed: {e}")

            return QueryResult(
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"connection_id": self.connection_id},
            )

    async def insert_many_documents(
        self,
        collection_name: str,
        documents: list[dict[str, Any]],
    ) -> QueryResult:
        """Insert multiple documents."""
        if self._database is None:
            raise ToolError("Not connected to database")

        start_time = time.time()

        try:
            collection = self._database[collection_name]
            result = await collection.insert_many(documents)

            execution_time = time.time() - start_time

            return QueryResult(
                success=True,
                rows_affected=len(result.inserted_ids),
                execution_time=execution_time,
                metadata={
                    "collection": collection_name,
                    "connection_id": self.connection_id,
                    "inserted_ids": [str(id_) for id_ in result.inserted_ids],
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Insert many operation failed: {e}")

            return QueryResult(
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"connection_id": self.connection_id},
            )

    async def update_documents(
        self,
        collection_name: str,
        filter_query: dict[str, Any],
        update_query: dict[str, Any],
        many: bool = False,
    ) -> QueryResult:
        """Update documents in a collection."""
        if self._database is None:
            raise ToolError("Not connected to database")

        start_time = time.time()

        try:
            collection = self._database[collection_name]

            if many:
                result = await collection.update_many(filter_query, update_query)
            else:
                result = await collection.update_one(filter_query, update_query)

            execution_time = time.time() - start_time

            return QueryResult(
                success=True,
                rows_affected=result.modified_count,
                execution_time=execution_time,
                metadata={
                    "collection": collection_name,
                    "connection_id": self.connection_id,
                    "matched_count": result.matched_count,
                    "modified_count": result.modified_count,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Update operation failed: {e}")

            return QueryResult(
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"connection_id": self.connection_id},
            )

    async def delete_documents(
        self,
        collection_name: str,
        filter_query: dict[str, Any],
        many: bool = False,
    ) -> QueryResult:
        """Delete documents from a collection."""
        if self._database is None:
            raise ToolError("Not connected to database")

        start_time = time.time()

        try:
            collection = self._database[collection_name]

            if many:
                result = await collection.delete_many(filter_query)
            else:
                result = await collection.delete_one(filter_query)

            execution_time = time.time() - start_time

            return QueryResult(
                success=True,
                rows_affected=result.deleted_count,
                execution_time=execution_time,
                metadata={
                    "collection": collection_name,
                    "connection_id": self.connection_id,
                    "deleted_count": result.deleted_count,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Delete operation failed: {e}")

            return QueryResult(
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"connection_id": self.connection_id},
            )

    async def aggregate_documents(
        self,
        collection_name: str,
        pipeline: list[dict[str, Any]],
    ) -> QueryResult:
        """Execute an aggregation pipeline."""
        if self._database is None:
            raise ToolError("Not connected to database")

        start_time = time.time()

        try:
            collection = self._database[collection_name]
            cursor = collection.aggregate(pipeline)

            documents = await cursor.to_list(length=None)
            serialized_docs = [self._serialize_document(doc) for doc in documents]

            execution_time = time.time() - start_time

            return QueryResult(
                success=True,
                rows_returned=len(serialized_docs),
                data=serialized_docs,
                execution_time=execution_time,
                metadata={
                    "collection": collection_name,
                    "connection_id": self.connection_id,
                    "pipeline_stages": len(pipeline),
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Aggregation operation failed: {e}")

            return QueryResult(
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"connection_id": self.connection_id},
            )

    def _serialize_document(self, doc: Any) -> Any:
        """Serialize MongoDB document for JSON compatibility."""
        if isinstance(doc, dict):
            serialized: dict[str, Any] = {}
            for key, value in doc.items():
                if isinstance(value, ObjectId):
                    serialized[key] = str(value)
                elif isinstance(value, dict):
                    serialized[key] = self._serialize_document(value)
                elif isinstance(value, list):
                    serialized_list: list[Any] = []
                    for item in value:
                        if isinstance(item, dict):
                            serialized_list.append(self._serialize_document(item))
                        elif isinstance(item, ObjectId):
                            serialized_list.append(str(item))
                        else:
                            serialized_list.append(item)
                    serialized[key] = serialized_list
                else:
                    serialized[key] = value
            return serialized
        else:
            return doc

    async def execute_many(
        self, query: str, param_list: list[dict[str, Any]]
    ) -> QueryResult:
        """Execute many operations (not applicable for MongoDB)."""
        return QueryResult(
            success=False,
            error="Use specific MongoDB bulk operations instead",
        )

    async def begin_transaction(
        self, isolation_level: Optional[str] = None
    ) -> TransactionContext:
        """Begin MongoDB transaction (requires replica set)."""
        if not self._client:
            raise ToolError("Not connected to database")

        try:
            if self._client:
                session = await self._client.start_session()
                session.start_transaction()
                self._session = session
            else:
                raise ToolError("Client not initialized")

            context = TransactionContext()
            self._transaction_context = context

            return context

        except Exception as e:
            self.logger.error(f"Failed to begin transaction: {e}")
            raise ToolError(f"Transaction begin failed: {e}")

    async def commit_transaction(self) -> None:
        """Commit MongoDB transaction."""
        if not self._session:
            raise ToolError("No active transaction")

        try:
            await self._session.commit_transaction()
            self._session = None
            self._transaction_context = None
        except Exception as e:
            self.logger.error(f"Failed to commit transaction: {e}")
            raise ToolError(f"Transaction commit failed: {e}")

    async def rollback_transaction(self) -> None:
        """Rollback MongoDB transaction."""
        if not self._session:
            raise ToolError("No active transaction")

        try:
            await self._session.abort_transaction()
            self._session = None
            self._transaction_context = None
        except Exception as e:
            self.logger.error(f"Failed to rollback transaction: {e}")
            raise ToolError(f"Transaction rollback failed: {e}")

    async def health_check(self) -> bool:
        """Check MongoDB connection health."""
        if not self._client:
            return False

        try:
            await self._client.admin.command("ping")
            return True
        except Exception:
            return False


class MongoDBTool(DatabaseTool):
    """High-performance MongoDB database tool."""

    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.database_type = DatabaseType.MONGODB

    def _create_db_config(self) -> DatabaseConfig:
        """Create MongoDB configuration from tool config."""
        env = self.config.environment

        return DatabaseConfig(
            host=env.get("host", "localhost"),
            port=int(env.get("port", 27017)),
            database=env.get("database", "test"),
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

    def _create_connection(self, config: DatabaseConfig) -> DatabaseConnection:
        """Create MongoDB connection."""
        return MongoDBConnection(config)

    async def get_schema(self) -> ToolSchema:
        """Return MongoDB tool schema."""
        return ToolSchema(
            name="mongodb",
            description="High-performance MongoDB database tool",
            version=self.config.version,
            actions={
                "find_documents": {
                    "description": "Find documents in a collection",
                    "parameters": {
                        "collection": {"type": "string", "required": True},
                        "filter": {"type": "object", "default": {}},
                        "projection": {"type": "object"},
                        "sort": {"type": "array"},
                        "limit": {"type": "integer"},
                        "skip": {"type": "integer"},
                    },
                },
                "insert_document": {
                    "description": "Insert a single document",
                    "parameters": {
                        "collection": {"type": "string", "required": True},
                        "document": {"type": "object", "required": True},
                    },
                },
                "insert_many": {
                    "description": "Insert multiple documents",
                    "parameters": {
                        "collection": {"type": "string", "required": True},
                        "documents": {"type": "array", "required": True},
                    },
                },
                "update_document": {
                    "description": "Update a single document",
                    "parameters": {
                        "collection": {"type": "string", "required": True},
                        "filter": {"type": "object", "required": True},
                        "update": {"type": "object", "required": True},
                    },
                },
                "update_many": {
                    "description": "Update multiple documents",
                    "parameters": {
                        "collection": {"type": "string", "required": True},
                        "filter": {"type": "object", "required": True},
                        "update": {"type": "object", "required": True},
                    },
                },
                "delete_document": {
                    "description": "Delete a single document",
                    "parameters": {
                        "collection": {"type": "string", "required": True},
                        "filter": {"type": "object", "required": True},
                    },
                },
                "delete_many": {
                    "description": "Delete multiple documents",
                    "parameters": {
                        "collection": {"type": "string", "required": True},
                        "filter": {"type": "object", "required": True},
                    },
                },
                "aggregate": {
                    "description": "Execute an aggregation pipeline",
                    "parameters": {
                        "collection": {"type": "string", "required": True},
                        "pipeline": {"type": "array", "required": True},
                    },
                },
                "create_collection": {
                    "description": "Create a new collection",
                    "parameters": {
                        "collection": {"type": "string", "required": True},
                        "options": {"type": "object", "default": {}},
                    },
                },
                "create_index": {
                    "description": "Create an index on a collection",
                    "parameters": {
                        "collection": {"type": "string", "required": True},
                        "keys": {"type": "object", "required": True},
                        "options": {"type": "object", "default": {}},
                    },
                },
                "get_statistics": {
                    "description": "Get database and collection statistics",
                    "parameters": {
                        "collection": {"type": "string"},
                    },
                },
                "create_user": {
                    "description": "Create a new MongoDB user",
                    "parameters": {
                        "username": {"type": "string", "required": True},
                        "password": {"type": "string", "required": True},
                        "roles": {"type": "array", "default": []},
                        "database": {"type": "string"},
                    },
                },
                "grant_role": {
                    "description": "Grant a role to a MongoDB user",
                    "parameters": {
                        "username": {"type": "string", "required": True},
                        "role": {"type": "string", "required": True},
                        "database": {"type": "string"},
                    },
                },
            },
        )

    async def _execute_action(
        self, action: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute MongoDB action."""
        if action == "find_documents":
            return await self._find_documents_action(params)
        elif action == "insert_document":
            return await self._insert_document_action(params)
        elif action == "insert_many":
            return await self._insert_many_action(params)
        elif action == "update_document":
            return await self._update_document_action(params)
        elif action == "update_many":
            return await self._update_many_action(params)
        elif action == "delete_document":
            return await self._delete_document_action(params)
        elif action == "delete_many":
            return await self._delete_many_action(params)
        elif action == "aggregate":
            return await self._aggregate_action(params)
        elif action == "create_collection":
            return await self._create_collection_action(params)
        elif action == "create_index":
            return await self._create_index_action(params)
        elif action == "get_statistics":
            return await self._get_statistics_action(params)
        elif action == "create_user":
            return await self._create_user_action(params)
        elif action == "grant_role":
            return await self._grant_role_action(params)
        else:
            raise ToolError(f"Unknown action: {action}")

    async def _find_documents_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Find documents in a collection."""
        collection = params["collection"]
        filter_query = params.get("filter", {})
        projection = params.get("projection")
        sort = params.get("sort")
        limit = params.get("limit")
        skip = params.get("skip")

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            mongo_conn = cast(MongoDBConnection, connection)
            result = await mongo_conn.find_documents(
                collection, filter_query, projection, sort, limit, skip
            )

        return {
            "success": result.success,
            "documents": result.data,
            "count": result.rows_returned,
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _insert_document_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Insert a single document."""
        collection = params["collection"]
        document = params["document"]

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            mongo_conn = cast(MongoDBConnection, connection)
            result = await mongo_conn.insert_document(collection, document)

        return {
            "success": result.success,
            "inserted_id": (
                result.metadata.get("inserted_id") if result.success else None
            ),
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _insert_many_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Insert multiple documents."""
        collection = params["collection"]
        documents = params["documents"]

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            mongo_conn = cast(MongoDBConnection, connection)
            result = await mongo_conn.insert_many_documents(collection, documents)

        return {
            "success": result.success,
            "inserted_count": result.rows_affected,
            "inserted_ids": (
                result.metadata.get("inserted_ids") if result.success else None
            ),
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _update_document_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Update a single document."""
        collection = params["collection"]
        filter_query = params["filter"]
        update_query = params["update"]

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            mongo_conn = cast(MongoDBConnection, connection)
            result = await mongo_conn.update_documents(
                collection, filter_query, update_query, many=False
            )

        return {
            "success": result.success,
            "matched_count": result.metadata.get("matched_count", 0),
            "modified_count": result.rows_affected,
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _update_many_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Update multiple documents."""
        collection = params["collection"]
        filter_query = params["filter"]
        update_query = params["update"]

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            mongo_conn = cast(MongoDBConnection, connection)
            result = await mongo_conn.update_documents(
                collection, filter_query, update_query, many=True
            )

        return {
            "success": result.success,
            "matched_count": result.metadata.get("matched_count", 0),
            "modified_count": result.rows_affected,
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _delete_document_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Delete a single document."""
        collection = params["collection"]
        filter_query = params["filter"]

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            mongo_conn = cast(MongoDBConnection, connection)
            result = await mongo_conn.delete_documents(
                collection, filter_query, many=False
            )

        return {
            "success": result.success,
            "deleted_count": result.rows_affected,
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _delete_many_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Delete multiple documents."""
        collection = params["collection"]
        filter_query = params["filter"]

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            mongo_conn = cast(MongoDBConnection, connection)
            result = await mongo_conn.delete_documents(
                collection, filter_query, many=True
            )

        return {
            "success": result.success,
            "deleted_count": result.rows_affected,
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _aggregate_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute an aggregation pipeline."""
        collection = params["collection"]
        pipeline = params["pipeline"]

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            mongo_conn = cast(MongoDBConnection, connection)
            result = await mongo_conn.aggregate_documents(collection, pipeline)

        return {
            "success": result.success,
            "documents": result.data,
            "count": result.rows_returned,
            "execution_time": result.execution_time,
            "error": result.error,
        }

    async def _create_collection_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create a new collection."""
        collection_name = params["collection"]
        options = params.get("options", {})

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            mongo_conn = cast(MongoDBConnection, connection)
            if mongo_conn._database is None:
                raise ToolError("Not connected to database")

            try:
                await mongo_conn._database.create_collection(collection_name, **options)
                return {
                    "success": True,
                    "collection": collection_name,
                    "message": f"Collection '{collection_name}' created successfully",
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                }

    async def _create_index_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create an index on a collection."""
        collection_name = params["collection"]
        keys = params["keys"]
        options = params.get("options", {})

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            mongo_conn = cast(MongoDBConnection, connection)
            if mongo_conn._database is None:
                raise ToolError("Not connected to database")

            try:
                collection = mongo_conn._database[collection_name]
                index_name = await collection.create_index(keys.items(), **options)

                return {
                    "success": True,
                    "collection": collection_name,
                    "index_name": index_name,
                    "message": f"Index '{index_name}' created successfully",
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                }

    async def _get_statistics_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get database and collection statistics."""
        collection_name = params.get("collection")

        stats = {
            "tool_stats": self.performance_stats,
            "database_info": await self._get_database_info(),
        }

        if collection_name:
            stats["collection_info"] = await self._get_collection_info(collection_name)

        return {"success": True, "statistics": stats}

    async def _get_database_info(self) -> dict[str, Any]:
        """Get MongoDB database information."""
        if not self.connection_pool:
            return {"error": "Database tool not initialized"}

        try:
            async with self.connection_pool.acquire() as connection:
                mongo_conn = cast(MongoDBConnection, connection)
                if not mongo_conn._database:
                    return {"error": "Not connected to database"}

                stats = await mongo_conn._database.command("dbStats")

                return {
                    "database_name": stats.get("db"),
                    "collections": stats.get("collections", 0),
                    "data_size": stats.get("dataSize", 0),
                    "storage_size": stats.get("storageSize", 0),
                    "indexes": stats.get("indexes", 0),
                    "index_size": stats.get("indexSize", 0),
                    "objects": stats.get("objects", 0),
                }
        except Exception as e:
            return {"error": str(e)}

    async def _get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get MongoDB collection information."""
        if not self.connection_pool:
            return {"error": "Database tool not initialized"}

        try:
            async with self.connection_pool.acquire() as connection:
                mongo_conn = cast(MongoDBConnection, connection)
                if mongo_conn._database is None:
                    return {"error": "Not connected to database"}

                stats = await mongo_conn._database.command("collStats", collection_name)

                return {
                    "collection_name": collection_name,
                    "document_count": stats.get("count", 0),
                    "size": stats.get("size", 0),
                    "storage_size": stats.get("storageSize", 0),
                    "total_index_size": stats.get("totalIndexSize", 0),
                    "indexes": len(stats.get("indexSizes", {})),
                }
        except Exception as e:
            return {"error": str(e)}

    async def _create_user_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create a new MongoDB user."""
        username = params["username"]
        password = params["password"]
        roles = params.get("roles", [])
        database = params.get(
            "database", self.db_config.database if self.db_config else "admin"
        )

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        try:
            async with self.connection_pool.acquire() as connection:
                mongo_conn = cast(MongoDBConnection, connection)
                if not mongo_conn._database:
                    raise ToolError("Not connected to database")

                # Use admin database for user creation
                if mongo_conn._client:
                    admin_db = mongo_conn._client["admin"]
                else:
                    raise ToolError("Client not initialized")

                # Prepare roles list
                formatted_roles = []
                for role in roles:
                    if isinstance(role, str):
                        formatted_roles.append({"role": role, "db": database})
                    else:
                        formatted_roles.append(role)

                # Create user
                await admin_db.command(
                    "createUser", username, pwd=password, roles=formatted_roles
                )

                return {
                    "success": True,
                    "username": username,
                    "database": database,
                    "roles": formatted_roles,
                    "message": f"User '{username}' created successfully",
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def _grant_role_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Grant a role to a MongoDB user."""
        username = params["username"]
        role = params["role"]
        database = params.get(
            "database", self.db_config.database if self.db_config else "admin"
        )

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        try:
            async with self.connection_pool.acquire() as connection:
                mongo_conn = cast(MongoDBConnection, connection)
                if not mongo_conn._database:
                    raise ToolError("Not connected to database")

                # Use admin database for role management
                if mongo_conn._client:
                    admin_db = mongo_conn._client["admin"]
                else:
                    raise ToolError("Client not initialized")

                # Grant role
                await admin_db.command(
                    "grantRolesToUser", username, roles=[{"role": role, "db": database}]
                )

                return {
                    "success": True,
                    "username": username,
                    "role": role,
                    "database": database,
                    "message": f"Role '{role}' granted to '{username}' on database '{database}'",
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def estimate_cost(self, action: str, params: dict[str, Any]) -> CostEstimate:
        """Estimate cost of MongoDB operations."""

        # MongoDB is typically free/open source, so cost is mainly computational
        base_cost = 0.0

        if action in ["find_documents", "aggregate"]:
            # Read operations - cost based on complexity
            if params.get("pipeline") or params.get("sort"):
                base_cost = 0.01  # Complex query
            else:
                base_cost = 0.001  # Simple query
        elif action in ["insert_document", "update_document", "delete_document"]:
            base_cost = 0.002  # Single document operations
        elif action in ["insert_many", "update_many", "delete_many"]:
            # Batch operations
            count = len(params.get("documents", [])) if "documents" in params else 10
            base_cost = 0.001 * count
        elif action in ["create_collection", "create_index"]:
            base_cost = 0.05  # Schema operations

        return CostEstimate(estimated_cost=base_cost, currency="USD", confidence=0.7)

    async def _get_supported_actions(self) -> list[str]:
        """Get list of supported MongoDB actions."""
        return [
            "find_documents",
            "insert_document",
            "update_document",
            "delete_document",
            "insert_many",
            "update_many",
            "delete_many",
            "create_collection",
            "create_index",
            "aggregate",
            "get_statistics",
            "create_user",
            "grant_roles",
        ]

    async def _execute_rollback(self, execution_id: str) -> dict[str, Any]:
        """Execute rollback operation for MongoDB."""
        # TODO: Implement rollback logic
        return {"message": f"Rollback not implemented for execution {execution_id}"}

    async def _create_client(self) -> MongoDBConnection:
        """Create MongoDB connection client."""
        if self.db_config is None:
            raise ToolError("Database configuration not initialized")
        return MongoDBConnection(self.db_config)

    async def _create_validator(self) -> None:
        """Create parameter validator (not needed for database tools)."""
        return None
