"""
MySQL database tool implementation.

High-performance MySQL connector with async operations, connection pooling,
and comprehensive database management capabilities.
"""

import time
from typing import Any, Optional

import aiomysql
from aiomysql import Connection
from pymysql.err import Error as MySQLError

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


class MySQLConnection(DatabaseConnection):
    """High-performance MySQL connection with aiomysql."""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._connection: Optional[Connection] = None
        self._transaction_active = False

    async def connect(self) -> None:
        """Establish MySQL connection."""
        try:
            self.state = ConnectionState.CONNECTING
            self.logger.debug(
                f"Connecting to MySQL at {self.config.host}:{self.config.port}"
            )

            # Build connection parameters
            conn_params = {
                "host": self.config.host,
                "port": self.config.port,
                "user": self.config.username,
                "password": self.config.password,
                "db": self.config.database,
                "connect_timeout": int(self.config.connection_timeout),
                "autocommit": True,
                "charset": "utf8mb4",
                "use_unicode": True,
                "sql_mode": "TRADITIONAL",
            }

            # SSL configuration
            if self.config.ssl_enabled:
                ssl_context = {}
                if self.config.ssl_cert_path:
                    ssl_context["cert"] = self.config.ssl_cert_path
                if self.config.ssl_key_path:
                    ssl_context["key"] = self.config.ssl_key_path
                if self.config.ssl_ca_path:
                    ssl_context["ca"] = self.config.ssl_ca_path
                conn_params["ssl"] = ssl_context

            # Additional options
            conn_params.update(self.config.options)

            self._connection = await aiomysql.connect(**conn_params)

            self.state = ConnectionState.CONNECTED
            self.logger.debug(f"Connected to MySQL {self.connection_id[:8]}")

        except Exception as e:
            self.state = ConnectionState.ERROR
            self.logger.error(f"Failed to connect to MySQL: {e}")
            raise ToolError(f"MySQL connection failed: {e}")

    async def disconnect(self) -> None:
        """Close MySQL connection."""
        if self._connection:
            self.state = ConnectionState.CLOSING
            try:
                self._connection.close()
                self.logger.debug(f"Disconnected from MySQL {self.connection_id[:8]}")
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")
            finally:
                self._connection = None
                self.state = ConnectionState.DISCONNECTED

    async def execute_query(
        self, query: str, params: Optional[dict[str, Any]] = None
    ) -> QueryResult:
        """Execute MySQL query with optimized performance."""
        if not self._connection:
            raise ToolError("Not connected to database")

        start_time = time.time()

        try:
            async with self._connection.cursor(aiomysql.DictCursor) as cursor:
                # Execute query
                await cursor.execute(query, params or {})

                # Handle different query types
                if (
                    query.strip()
                    .upper()
                    .startswith(("SELECT", "SHOW", "DESCRIBE", "EXPLAIN"))
                ):
                    # Read operation
                    data = await cursor.fetchall()
                    rows_returned = len(data)
                    rows_affected = 0
                else:
                    # Write operation
                    rows_affected = cursor.rowcount
                    data = []
                    rows_returned = 0

                execution_time = time.time() - start_time

                return QueryResult(
                    success=True,
                    rows_affected=rows_affected,
                    rows_returned=rows_returned,
                    data=data,
                    execution_time=execution_time,
                    metadata={
                        "query_type": query.strip().split()[0].upper(),
                        "connection_id": self.connection_id,
                        "last_insert_id": (
                            cursor.lastrowid if hasattr(cursor, "lastrowid") else None
                        ),
                    },
                )

        except MySQLError as e:
            execution_time = time.time() - start_time
            self.logger.error(f"MySQL query execution failed: {e}")

            return QueryResult(
                success=False,
                execution_time=execution_time,
                error=(
                    f"MySQL Error {e.args[0]}: {e.args[1]}"
                    if len(e.args) >= 2
                    else str(e)
                ),
                metadata={"connection_id": self.connection_id},
            )
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Query execution failed: {e}")

            return QueryResult(
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"connection_id": self.connection_id},
            )

    async def execute_many(
        self, query: str, param_list: list[dict[str, Any]]
    ) -> QueryResult:
        """Execute query with multiple parameter sets efficiently."""
        if not self._connection:
            raise ToolError("Not connected to database")

        start_time = time.time()
        total_affected = 0

        try:
            async with self._connection.cursor() as cursor:
                await cursor.executemany(query, param_list)
                total_affected = cursor.rowcount

            execution_time = time.time() - start_time

            return QueryResult(
                success=True,
                rows_affected=total_affected,
                execution_time=execution_time,
                metadata={
                    "batch_size": len(param_list),
                    "connection_id": self.connection_id,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Batch execution failed: {e}")

            return QueryResult(
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"connection_id": self.connection_id},
            )

    async def begin_transaction(
        self, isolation_level: Optional[str] = None
    ) -> TransactionContext:
        """Begin MySQL transaction."""
        if not self._connection:
            raise ToolError("Not connected to database")

        try:
            # Set isolation level if specified
            if isolation_level:
                isolation_map = {
                    "read_uncommitted": "READ UNCOMMITTED",
                    "read_committed": "READ COMMITTED",
                    "repeatable_read": "REPEATABLE READ",
                    "serializable": "SERIALIZABLE",
                }
                level = isolation_map.get(isolation_level, isolation_level.upper())
                await self._connection.select_db(self.config.database)
                async with self._connection.cursor() as cursor:
                    await cursor.execute(
                        f"SET SESSION TRANSACTION ISOLATION LEVEL {level}"
                    )

            # Begin transaction
            await self._connection.begin()
            self._transaction_active = True

            context = TransactionContext(
                isolation_level=isolation_level,
            )
            self._transaction_context = context

            return context

        except Exception as e:
            self.logger.error(f"Failed to begin transaction: {e}")
            raise ToolError(f"Transaction begin failed: {e}")

    async def commit_transaction(self) -> None:
        """Commit MySQL transaction."""
        if not self._connection:
            raise ToolError("Not connected to database")

        try:
            await self._connection.commit()
            self._transaction_active = False
            self._transaction_context = None
        except Exception as e:
            self.logger.error(f"Failed to commit transaction: {e}")
            raise ToolError(f"Transaction commit failed: {e}")

    async def rollback_transaction(self) -> None:
        """Rollback MySQL transaction."""
        if not self._connection:
            raise ToolError("Not connected to database")

        try:
            await self._connection.rollback()
            self._transaction_active = False
            self._transaction_context = None
        except Exception as e:
            self.logger.error(f"Failed to rollback transaction: {e}")
            raise ToolError(f"Transaction rollback failed: {e}")

    async def health_check(self) -> bool:
        """Check MySQL connection health."""
        if not self._connection:
            return False

        try:
            await self._connection.ping()
            return True
        except Exception:
            return False


class MySQLTool(DatabaseTool):
    """High-performance MySQL database tool."""

    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.database_type = DatabaseType.MYSQL

    def _create_db_config(self) -> DatabaseConfig:
        """Create MySQL configuration from tool config."""
        env = self.config.environment

        return DatabaseConfig(
            host=env.get("host", "localhost"),
            port=int(env.get("port", 3306)),
            database=env.get("database", "mysql"),
            username=env.get("username", "root"),
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
        """Create MySQL connection."""
        return MySQLConnection(config)

    async def get_schema(self) -> ToolSchema:
        """Return MySQL tool schema."""
        return ToolSchema(
            name="mysql",
            description="High-performance MySQL database tool",
            version=self.config.version,
            actions={
                "execute_query": {
                    "description": "Execute a MySQL query",
                    "parameters": {
                        "query": {"type": "string", "required": True},
                        "params": {"type": "object", "default": {}},
                        "timeout": {"type": "number", "default": 30},
                    },
                },
                "execute_many": {
                    "description": "Execute query with multiple parameter sets",
                    "parameters": {
                        "query": {"type": "string", "required": True},
                        "param_list": {"type": "array", "required": True},
                    },
                },
                "create_database": {
                    "description": "Create a new MySQL database",
                    "parameters": {
                        "database_name": {"type": "string", "required": True},
                        "charset": {"type": "string", "default": "utf8mb4"},
                        "collation": {
                            "type": "string",
                            "default": "utf8mb4_unicode_ci",
                        },
                    },
                },
                "create_table": {
                    "description": "Create a new table",
                    "parameters": {
                        "table_name": {"type": "string", "required": True},
                        "columns": {"type": "array", "required": True},
                        "engine": {"type": "string", "default": "InnoDB"},
                        "charset": {"type": "string", "default": "utf8mb4"},
                    },
                },
                "create_index": {
                    "description": "Create an index on a table",
                    "parameters": {
                        "index_name": {"type": "string", "required": True},
                        "table_name": {"type": "string", "required": True},
                        "columns": {"type": "array", "required": True},
                        "unique": {"type": "boolean", "default": False},
                        "index_type": {
                            "type": "string",
                            "enum": ["BTREE", "HASH"],
                            "default": "BTREE",
                        },
                    },
                },
                "optimize_table": {
                    "description": "Optimize a MySQL table",
                    "parameters": {
                        "table_name": {"type": "string", "required": True},
                    },
                },
                "analyze_table": {
                    "description": "Analyze table statistics",
                    "parameters": {
                        "table_name": {"type": "string", "required": True},
                    },
                },
                "get_statistics": {
                    "description": "Get database and connection pool statistics",
                    "parameters": {},
                },
                "create_user": {
                    "description": "Create a new MySQL user",
                    "parameters": {
                        "username": {"type": "string", "required": True},
                        "password": {"type": "string", "required": True},
                        "host": {"type": "string", "default": "%"},
                        "permissions": {"type": "array", "default": []},
                        "databases": {"type": "array", "default": []},
                    },
                },
                "grant_permissions": {
                    "description": "Grant permissions to a user",
                    "parameters": {
                        "username": {"type": "string", "required": True},
                        "host": {"type": "string", "default": "%"},
                        "permissions": {"type": "array", "required": True},
                        "database": {"type": "string"},
                        "table": {"type": "string"},
                    },
                },
            },
        )

    async def _execute_action(
        self, action: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute MySQL action."""
        if action == "execute_query":
            return await self._execute_query_action(params)
        elif action == "execute_many":
            return await self._execute_many_action(params)
        elif action == "create_database":
            return await self._create_database_action(params)
        elif action == "create_table":
            return await self._create_table_action(params)
        elif action == "create_index":
            return await self._create_index_action(params)
        elif action == "optimize_table":
            return await self._optimize_table_action(params)
        elif action == "analyze_table":
            return await self._analyze_table_action(params)
        elif action == "get_statistics":
            return await self._get_statistics_action(params)
        elif action == "create_user":
            return await self._create_user_action(params)
        elif action == "grant_permissions":
            return await self._grant_permissions_action(params)
        else:
            raise ToolError(f"Unknown action: {action}")

    async def _execute_query_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a MySQL query."""
        query = params["query"]
        query_params = params.get("params", {})
        timeout = params.get("timeout", 30)

        result = await self.execute_query(query, query_params, timeout)

        return {
            "success": result.success,
            "rows_affected": result.rows_affected,
            "rows_returned": result.rows_returned,
            "data": result.data,
            "execution_time": result.execution_time,
            "query_id": result.query_id,
            "last_insert_id": result.metadata.get("last_insert_id"),
            "error": result.error,
        }

    async def _execute_many_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute query with multiple parameter sets."""
        query = params["query"]
        param_list = params["param_list"]

        if not self.connection_pool:
            raise ToolError("Database tool not initialized")

        async with self.connection_pool.acquire() as connection:
            result = await connection.execute_many(query, param_list)

        return {
            "success": result.success,
            "rows_affected": result.rows_affected,
            "execution_time": result.execution_time,
            "batch_size": len(param_list),
            "error": result.error,
        }

    async def _create_database_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create a new MySQL database."""
        database_name = params["database_name"]
        charset = params.get("charset", "utf8mb4")
        collation = params.get("collation", "utf8mb4_unicode_ci")

        query = f"CREATE DATABASE `{database_name}` CHARACTER SET {charset} COLLATE {collation}"

        result = await self.execute_query(query)

        return {
            "success": result.success,
            "database_name": database_name,
            "charset": charset,
            "collation": collation,
            "message": (
                f"Database '{database_name}' created successfully"
                if result.success
                else None
            ),
            "error": result.error,
        }

    async def _create_table_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create a new MySQL table."""
        table_name = params["table_name"]
        columns = params["columns"]
        engine = params.get("engine", "InnoDB")
        charset = params.get("charset", "utf8mb4")

        # Build CREATE TABLE query
        column_defs = []
        for col in columns:
            col_def = f"`{col['name']}` {col['type']}"
            if col.get("not_null"):
                col_def += " NOT NULL"
            if col.get("auto_increment"):
                col_def += " AUTO_INCREMENT"
            if col.get("default") is not None:
                col_def += f" DEFAULT {col['default']}"
            if col.get("primary_key"):
                col_def += " PRIMARY KEY"
            column_defs.append(col_def)

        query = (
            f"CREATE TABLE `{table_name}` (\n"
            f"  {',  '.join(column_defs)}\n"
            f") ENGINE={engine} CHARACTER SET {charset}"
        )

        result = await self.execute_query(query)

        return {
            "success": result.success,
            "table_name": table_name,
            "engine": engine,
            "charset": charset,
            "message": (
                f"Table '{table_name}' created successfully" if result.success else None
            ),
            "error": result.error,
        }

    async def _create_index_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create an index on a MySQL table."""
        index_name = params["index_name"]
        table_name = params["table_name"]
        columns = params["columns"]
        unique = params.get("unique", False)
        index_type = params.get("index_type", "BTREE")

        # Build CREATE INDEX query
        query_parts = ["CREATE"]
        if unique:
            query_parts.append("UNIQUE")
        query_parts.append("INDEX")

        column_list = ", ".join(f"`{col}`" for col in columns)
        query_parts.extend(
            [
                f"`{index_name}`",
                "ON",
                f"`{table_name}`",
                f"({column_list})",
                f"USING {index_type}",
            ]
        )

        query = " ".join(query_parts)

        result = await self.execute_query(query)

        return {
            "success": result.success,
            "index_name": index_name,
            "table_name": table_name,
            "unique": unique,
            "index_type": index_type,
            "message": (
                f"Index '{index_name}' created successfully" if result.success else None
            ),
            "error": result.error,
        }

    async def _optimize_table_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Optimize a MySQL table."""
        table_name = params["table_name"]

        query = f"OPTIMIZE TABLE `{table_name}`"
        result = await self.execute_query(query)

        return {
            "success": result.success,
            "table_name": table_name,
            "message": (
                f"Table '{table_name}' optimized successfully"
                if result.success
                else None
            ),
            "optimization_result": result.data if result.success else None,
            "error": result.error,
        }

    async def _analyze_table_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Analyze MySQL table statistics."""
        table_name = params["table_name"]

        query = f"ANALYZE TABLE `{table_name}`"
        result = await self.execute_query(query)

        return {
            "success": result.success,
            "table_name": table_name,
            "message": (
                f"Table '{table_name}' analyzed successfully"
                if result.success
                else None
            ),
            "analysis_result": result.data if result.success else None,
            "error": result.error,
        }

    async def _get_statistics_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get database and connection pool statistics."""
        stats = {
            "tool_stats": self.performance_stats,
            "database_info": await self._get_database_info(),
        }

        return {"success": True, "statistics": stats}

    async def _get_database_info(self) -> dict[str, Any]:
        """Get MySQL database information."""
        queries = {
            "version": "SELECT VERSION() as version",
            "database_size": """
                SELECT ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS size_mb
                FROM information_schema.tables
                WHERE table_schema = DATABASE()
            """,
            "connection_count": "SHOW STATUS LIKE 'Threads_connected'",
            "table_count": """
                SELECT COUNT(*) as count
                FROM information_schema.tables
                WHERE table_schema = DATABASE()
            """,
        }

        info = {}
        for key, query in queries.items():
            try:
                result = await self.execute_query(query)
                if result.success and result.data:
                    if key == "connection_count":
                        info[key] = result.data[0]["Value"]
                    else:
                        info[key] = result.data[0][list(result.data[0].keys())[0]]
            except Exception as e:
                info[key] = f"Error: {e}"

        return info

    async def _create_user_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create a new MySQL user."""
        username = params["username"]
        password = params["password"]
        host = params.get("host", "%")
        permissions = params.get("permissions", [])
        databases = params.get("databases", [])

        try:
            # Create user
            query = f"CREATE USER '{username}'@'{host}' IDENTIFIED BY '{password}'"
            result = await self.execute_query(query)

            if not result.success:
                return {
                    "success": False,
                    "error": f"Failed to create user: {result.error}",
                }

            # Grant database access if specified
            for database in databases:
                for permission in permissions if permissions else ["SELECT"]:
                    grant_query = f"GRANT {permission.upper()} ON `{database}`.* TO '{username}'@'{host}'"
                    await self.execute_query(grant_query)

            # Flush privileges
            await self.execute_query("FLUSH PRIVILEGES")

            return {
                "success": True,
                "username": username,
                "host": host,
                "message": f"User '{username}'@'{host}' created successfully",
                "permissions_granted": permissions,
                "databases_granted": databases,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def _grant_permissions_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Grant permissions to a MySQL user."""
        username = params["username"]
        host = params.get("host", "%")
        permissions = params["permissions"]
        database = params.get("database", "*")
        table = params.get("table", "*")

        try:
            granted_perms = []

            for permission in permissions:
                if table and table != "*":
                    # Table-level permissions
                    query = f"GRANT {permission.upper()} ON `{database}`.`{table}` TO '{username}'@'{host}'"
                else:
                    # Database-level permissions
                    query = f"GRANT {permission.upper()} ON `{database}`.* TO '{username}'@'{host}'"

                result = await self.execute_query(query)
                if result.success:
                    granted_perms.append(permission)

            # Flush privileges
            await self.execute_query("FLUSH PRIVILEGES")

            return {
                "success": len(granted_perms) > 0,
                "username": username,
                "host": host,
                "permissions_granted": granted_perms,
                "database": database,
                "table": table,
                "message": f"Granted {len(granted_perms)} permissions to '{username}'@'{host}'",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def estimate_cost(self, action: str, params: dict[str, Any]) -> CostEstimate:
        """Estimate cost of MySQL operations."""

        # MySQL is typically free/open source, so cost is mainly computational
        base_cost = 0.0

        if action in ["execute_query", "execute_many"]:
            # Estimate based on query complexity
            query = params.get("query", "")
            if any(
                keyword in query.upper() for keyword in ["JOIN", "GROUP BY", "ORDER BY"]
            ):
                base_cost = 0.01  # Complex query
            else:
                base_cost = 0.001  # Simple query
        elif action in ["create_database", "create_table", "create_index"]:
            base_cost = 0.05  # Schema operations
        elif action in ["optimize_table", "analyze_table"]:
            base_cost = 0.02  # Maintenance operations

        return CostEstimate(estimated_cost=base_cost, currency="USD", confidence=0.7)

    async def _get_supported_actions(self) -> list[str]:
        """Get list of supported MySQL actions."""
        return [
            "execute_query",
            "execute_many",
            "create_database",
            "create_table",
            "create_index",
            "optimize_table",
            "analyze_table",
            "get_statistics",
            "create_user",
            "grant_roles",
        ]

    async def _execute_rollback(self, execution_id: str) -> dict[str, Any]:
        """Execute rollback operation for MySQL."""
        # TODO: Implement rollback logic
        return {"message": f"Rollback not implemented for execution {execution_id}"}

    async def _create_client(self) -> MySQLConnection:
        """Create MySQL connection client."""
        if self.db_config is None:
            raise ToolError("Database configuration not initialized")
        return MySQLConnection(self.db_config)

    async def _create_validator(self) -> None:
        """Create parameter validator (not needed for database tools)."""
        return None
