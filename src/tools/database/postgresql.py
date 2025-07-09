"""
PostgreSQL database tool implementation.

High-performance PostgreSQL connector with async operations, connection pooling,
prepared statements, and comprehensive query capabilities.
"""

import time
from typing import Any, Optional

import asyncpg
from asyncpg import Connection

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


class PostgreSQLConnection(DatabaseConnection):
    """High-performance PostgreSQL connection with asyncpg."""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._connection: Optional[Connection] = None
        self._prepared_statements: dict[str, str] = {}

    async def connect(self) -> None:
        """Establish PostgreSQL connection."""
        try:
            self.state = ConnectionState.CONNECTING
            self.logger.debug(
                f"Connecting to PostgreSQL at {self.config.host}:{self.config.port}"
            )

            # Build connection string
            dsn = self._build_dsn()

            # Connect with optimized settings
            self._connection = await asyncpg.connect(
                dsn,
                timeout=self.config.connection_timeout,
                command_timeout=60,
                server_settings={
                    "application_name": "orcastrate_db_tool",
                    "tcp_keepalives_idle": "300",
                    "tcp_keepalives_interval": "30",
                    "tcp_keepalives_count": "3",
                },
            )

            self.state = ConnectionState.CONNECTED
            self.logger.debug(f"Connected to PostgreSQL {self.connection_id[:8]}")

        except Exception as e:
            self.state = ConnectionState.ERROR
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise ToolError(f"PostgreSQL connection failed: {e}")

    def _build_dsn(self) -> str:
        """Build PostgreSQL connection string."""
        dsn_parts = [
            f"postgresql://{self.config.username}:{self.config.password}",
            f"@{self.config.host}:{self.config.port}/{self.config.database}",
        ]

        params = []

        # SSL configuration
        if self.config.ssl_enabled:
            params.append("sslmode=require")
            if self.config.ssl_cert_path:
                params.append(f"sslcert={self.config.ssl_cert_path}")
            if self.config.ssl_key_path:
                params.append(f"sslkey={self.config.ssl_key_path}")
            if self.config.ssl_ca_path:
                params.append(f"sslrootcert={self.config.ssl_ca_path}")

        # Additional options
        for key, value in self.config.options.items():
            params.append(f"{key}={value}")

        if params:
            dsn_parts.append("?" + "&".join(params))

        return "".join(dsn_parts)

    async def disconnect(self) -> None:
        """Close PostgreSQL connection."""
        if self._connection:
            self.state = ConnectionState.CLOSING
            try:
                await self._connection.close()
                self.logger.debug(
                    f"Disconnected from PostgreSQL {self.connection_id[:8]}"
                )
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")
            finally:
                self._connection = None
                self.state = ConnectionState.DISCONNECTED

    async def execute_query(
        self, query: str, params: Optional[dict[str, Any]] = None
    ) -> QueryResult:
        """Execute PostgreSQL query with optimized performance."""
        if not self._connection:
            raise ToolError("Not connected to database")

        start_time = time.time()

        try:
            # Prepare parameters for asyncpg
            if params:
                # Convert named parameters to positional for asyncpg
                query, args = self._convert_params(query, params)
            else:
                args = []

            # Execute query based on type
            if query.strip().upper().startswith(("SELECT", "WITH", "EXPLAIN")):
                # Read operation
                rows = await self._connection.fetch(query, *args)
                data = [dict(row) for row in rows]
                rows_returned = len(data)
                rows_affected = 0
            else:
                # Write operation
                result = await self._connection.execute(query, *args)
                # Parse result string like "INSERT 0 5" or "UPDATE 3"
                if result.startswith(("INSERT", "UPDATE", "DELETE")):
                    parts = result.split()
                    rows_affected = int(parts[-1]) if parts[-1].isdigit() else 0
                else:
                    rows_affected = 1
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
                },
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

    def _convert_params(
        self, query: str, params: dict[str, Any]
    ) -> tuple[str, list[Any]]:
        """Convert named parameters to positional for asyncpg."""
        import re

        args = []
        converted_query = query

        # Handle both %(name)s and %s parameter formats
        if params:
            # First handle named parameters %(name)s
            named_pattern = re.compile(r"%\(([^)]+)\)s")
            matches = named_pattern.findall(converted_query)

            if matches:
                # Named parameters found
                for i, key in enumerate(matches, 1):
                    if key in params:
                        converted_query = converted_query.replace(f"%({key})s", f"${i}")
                        args.append(params[key])
            else:
                # Check for positional parameters %s
                positional_count = converted_query.count("%s")
                if positional_count > 0:
                    # Convert positional %s to $1, $2, etc.
                    for i in range(1, positional_count + 1):
                        converted_query = converted_query.replace("%s", f"${i}", 1)

                    # Use parameter values in the order they appear in the dict
                    args = list(params.values())[:positional_count]

        return converted_query, args

    async def execute_many(
        self, query: str, param_list: list[dict[str, Any]]
    ) -> QueryResult:
        """Execute query with multiple parameter sets efficiently."""
        if not self._connection:
            raise ToolError("Not connected to database")

        start_time = time.time()
        total_affected = 0

        try:
            # Use executemany for better performance
            for params in param_list:
                query_converted, args = self._convert_params(query, params)
                result = await self._connection.execute(query_converted, *args)

                # Parse affected rows
                if result.startswith(("INSERT", "UPDATE", "DELETE")):
                    parts = result.split()
                    if parts[-1].isdigit():
                        total_affected += int(parts[-1])
                else:
                    total_affected += 1

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
        """Begin PostgreSQL transaction."""
        if not self._connection:
            raise ToolError("Not connected to database")

        transaction = self._connection.transaction(
            isolation=isolation_level or "read_committed"
        )
        await transaction.start()

        context = TransactionContext(
            isolation_level=isolation_level,
        )
        self._transaction_context = context

        return context

    async def commit_transaction(self) -> None:
        """Commit PostgreSQL transaction."""
        if not self._connection:
            raise ToolError("Not connected to database")

        # asyncpg handles transaction commit automatically
        self._transaction_context = None

    async def rollback_transaction(self) -> None:
        """Rollback PostgreSQL transaction."""
        if not self._connection:
            raise ToolError("Not connected to database")

        # asyncpg handles transaction rollback automatically
        self._transaction_context = None

    async def health_check(self) -> bool:
        """Check PostgreSQL connection health."""
        if not self._connection or self._connection.is_closed():
            return False

        try:
            await self._connection.fetchval("SELECT 1")
            return True
        except Exception:
            return False


class PostgreSQLTool(DatabaseTool):
    """High-performance PostgreSQL database tool."""

    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.database_type = DatabaseType.POSTGRESQL

    def _create_db_config(self) -> DatabaseConfig:
        """Create PostgreSQL configuration from tool config."""
        env = self.config.environment

        return DatabaseConfig(
            host=env.get("host", "localhost"),
            port=int(env.get("port", 5432)),
            database=env.get("database", "postgres"),
            username=env.get("username", "postgres"),
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
        """Create PostgreSQL connection."""
        return PostgreSQLConnection(config)

    async def get_schema(self) -> ToolSchema:
        """Return PostgreSQL tool schema."""
        return ToolSchema(
            name="postgresql",
            description="High-performance PostgreSQL database tool",
            version=self.config.version,
            actions={
                "execute_query": {
                    "description": "Execute a PostgreSQL query",
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
                    "description": "Create a new PostgreSQL database",
                    "parameters": {
                        "database_name": {"type": "string", "required": True},
                        "owner": {"type": "string"},
                        "encoding": {"type": "string", "default": "UTF8"},
                        "template": {"type": "string", "default": "template1"},
                    },
                },
                "create_table": {
                    "description": "Create a new table",
                    "parameters": {
                        "table_name": {"type": "string", "required": True},
                        "columns": {"type": "array", "required": True},
                        "constraints": {"type": "array", "default": []},
                        "indexes": {"type": "array", "default": []},
                    },
                },
                "create_index": {
                    "description": "Create an index on a table",
                    "parameters": {
                        "index_name": {"type": "string", "required": True},
                        "table_name": {"type": "string", "required": True},
                        "columns": {"type": "array", "required": True},
                        "unique": {"type": "boolean", "default": False},
                        "concurrent": {"type": "boolean", "default": True},
                    },
                },
                "backup_database": {
                    "description": "Create a database backup",
                    "parameters": {
                        "database_name": {"type": "string", "required": True},
                        "backup_path": {"type": "string", "required": True},
                        "format": {
                            "type": "string",
                            "enum": ["custom", "plain", "directory"],
                            "default": "custom",
                        },
                        "compress": {"type": "boolean", "default": True},
                    },
                },
                "restore_database": {
                    "description": "Restore a database from backup",
                    "parameters": {
                        "backup_path": {"type": "string", "required": True},
                        "database_name": {"type": "string", "required": True},
                        "clean": {"type": "boolean", "default": False},
                    },
                },
                "analyze_performance": {
                    "description": "Analyze query performance and get recommendations",
                    "parameters": {
                        "query": {"type": "string", "required": True},
                        "params": {"type": "object", "default": {}},
                    },
                },
                "get_statistics": {
                    "description": "Get database and connection pool statistics",
                    "parameters": {},
                },
                "create_user": {
                    "description": "Create a new PostgreSQL user",
                    "parameters": {
                        "username": {"type": "string", "required": True},
                        "password": {"type": "string", "required": True},
                        "permissions": {"type": "array", "default": []},
                        "databases": {"type": "array", "default": []},
                    },
                },
                "grant_permissions": {
                    "description": "Grant permissions to a user",
                    "parameters": {
                        "username": {"type": "string", "required": True},
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
        """Execute PostgreSQL action."""
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
        elif action == "backup_database":
            return await self._backup_database_action(params)
        elif action == "restore_database":
            return await self._restore_database_action(params)
        elif action == "analyze_performance":
            return await self._analyze_performance_action(params)
        elif action == "get_statistics":
            return await self._get_statistics_action(params)
        elif action == "create_user":
            return await self._create_user_action(params)
        elif action == "grant_permissions":
            return await self._grant_permissions_action(params)
        else:
            raise ToolError(f"Unknown action: {action}")

    async def _execute_query_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a PostgreSQL query."""
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
        """Create a new PostgreSQL database."""
        database_name = params["database_name"]
        owner = params.get("owner")
        encoding = params.get("encoding", "UTF8")
        template = params.get("template", "template1")

        # Build CREATE DATABASE query
        query_parts = [f'CREATE DATABASE "{database_name}"']

        if owner:
            query_parts.append(f'OWNER "{owner}"')
        query_parts.append(f"ENCODING '{encoding}'")
        query_parts.append(f"TEMPLATE {template}")

        query = " ".join(query_parts)

        result = await self.execute_query(query)

        return {
            "success": result.success,
            "database_name": database_name,
            "message": (
                f"Database '{database_name}' created successfully"
                if result.success
                else None
            ),
            "error": result.error,
        }

    async def _create_table_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create a new PostgreSQL table."""
        table_name = params["table_name"]
        columns = params["columns"]
        constraints = params.get("constraints", [])
        indexes = params.get("indexes", [])

        # Build CREATE TABLE query
        column_defs = []
        for col in columns:
            col_def = f'"{col["name"]}" {col["type"]}'
            if col.get("not_null"):
                col_def += " NOT NULL"
            if col.get("default"):
                col_def += f" DEFAULT {col['default']}"
            column_defs.append(col_def)

        # Add constraints
        column_defs.extend(constraints)

        query = f'CREATE TABLE "{table_name}" (\n  {", ".join(column_defs)}\n)'

        result = await self.execute_query(query)

        # Create indexes if specified
        if result.success and indexes:
            for idx in indexes:
                await self._create_index_action(
                    {
                        "index_name": idx["name"],
                        "table_name": table_name,
                        "columns": idx["columns"],
                        "unique": idx.get("unique", False),
                    }
                )

        return {
            "success": result.success,
            "table_name": table_name,
            "message": (
                f"Table '{table_name}' created successfully" if result.success else None
            ),
            "error": result.error,
        }

    async def _create_index_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create an index on a PostgreSQL table."""
        index_name = params["index_name"]
        table_name = params["table_name"]
        columns = params["columns"]
        unique = params.get("unique", False)
        concurrent = params.get("concurrent", True)

        # Build CREATE INDEX query
        query_parts = ["CREATE"]
        if unique:
            query_parts.append("UNIQUE")
        query_parts.append("INDEX")
        if concurrent:
            query_parts.append("CONCURRENTLY")

        column_list = ", ".join(f'"{col}"' for col in columns)
        query_parts.extend(
            [f'"{index_name}"', "ON", f'"{table_name}"', f"({column_list})"]
        )

        query = " ".join(query_parts)

        result = await self.execute_query(query)

        return {
            "success": result.success,
            "index_name": index_name,
            "table_name": table_name,
            "message": (
                f"Index '{index_name}' created successfully" if result.success else None
            ),
            "error": result.error,
        }

    async def _backup_database_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create a PostgreSQL database backup using pg_dump."""
        # This would require subprocess execution - placeholder implementation
        return {
            "success": False,
            "error": "Backup functionality requires subprocess integration (not implemented in this version)",
        }

    async def _restore_database_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Restore a PostgreSQL database from backup."""
        # This would require subprocess execution - placeholder implementation
        return {
            "success": False,
            "error": "Restore functionality requires subprocess integration (not implemented in this version)",
        }

    async def _analyze_performance_action(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze query performance using EXPLAIN ANALYZE."""
        query = params["query"]
        query_params = params.get("params", {})

        # Add EXPLAIN ANALYZE to the query
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"

        result = await self.execute_query(explain_query, query_params)

        if result.success and result.data:
            # Parse execution plan - QUERY PLAN is a JSON string that needs parsing
            import json

            query_plan_str = result.data[0].get("QUERY PLAN", "[]")
            plan_list = json.loads(query_plan_str)
            plan_data = plan_list[0] if plan_list else {}

            return {
                "success": True,
                "execution_time": plan_data.get("Execution Time", 0),
                "planning_time": plan_data.get("Planning Time", 0),
                "total_cost": plan_data.get("Plan", {}).get("Total Cost", 0),
                "rows_returned": plan_data.get("Plan", {}).get("Actual Rows", 0),
                "execution_plan": plan_data,
                "recommendations": self._generate_performance_recommendations(
                    plan_data
                ),
            }
        else:
            return {
                "success": False,
                "error": result.error or "Failed to analyze query performance",
            }

    def _generate_performance_recommendations(
        self, plan_data: dict[str, Any]
    ) -> list[str]:
        """Generate performance recommendations based on execution plan."""
        recommendations = []

        # Simple recommendations based on common patterns
        plan = plan_data.get("Plan", {})

        if plan.get("Node Type") == "Seq Scan":
            recommendations.append("Consider adding an index to avoid sequential scan")

        if plan.get("Total Cost", 0) > 1000:
            recommendations.append("Query has high cost - consider optimization")

        if plan_data.get("Execution Time", 0) > 1000:  # > 1 second
            recommendations.append(
                "Query execution time is high - consider optimization"
            )

        return recommendations

    async def _get_statistics_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get database and connection pool statistics."""
        stats = {
            "tool_stats": self.performance_stats,
            "database_info": await self._get_database_info(),
        }

        return {"success": True, "statistics": stats}

    async def _get_database_info(self) -> dict[str, Any]:
        """Get PostgreSQL database information."""
        queries = {
            "version": "SELECT version()",
            "database_size": "SELECT pg_size_pretty(pg_database_size(current_database()))",
            "connection_count": "SELECT count(*) FROM pg_stat_activity",
            "table_count": "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'",
        }

        info = {}
        for key, query in queries.items():
            try:
                result = await self.execute_query(query)
                if result.success and result.data:
                    info[key] = result.data[0][list(result.data[0].keys())[0]]
            except Exception as e:
                info[key] = f"Error: {e}"

        return info

    async def _create_user_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create a new PostgreSQL user."""
        username = params["username"]
        password = params["password"]
        permissions = params.get("permissions", [])
        databases = params.get("databases", [])

        try:
            # Create user - PostgreSQL doesn't allow parameters for passwords in CREATE USER
            # We need to escape the password string properly
            escaped_password = password.replace("'", "''")  # Escape single quotes
            query = f"CREATE USER \"{username}\" WITH PASSWORD '{escaped_password}'"
            result = await self.execute_query(query, {})

            if not result.success:
                return {
                    "success": False,
                    "error": f"Failed to create user: {result.error}",
                }

            # Grant permissions if specified
            for permission in permissions:
                if permission.upper() in [
                    "SUPERUSER",
                    "CREATEDB",
                    "CREATEROLE",
                    "LOGIN",
                ]:
                    alter_query = f'ALTER USER "{username}" {permission.upper()}'
                    await self.execute_query(alter_query)

            # Grant database access if specified
            for database in databases:
                grant_query = f'GRANT CONNECT ON DATABASE "{database}" TO "{username}"'
                await self.execute_query(grant_query)

            return {
                "success": True,
                "username": username,
                "message": f"User '{username}' created successfully",
                "permissions_granted": permissions,
                "databases_granted": databases,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def _grant_permissions_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Grant permissions to a PostgreSQL user."""
        username = params["username"]
        permissions = params["permissions"]
        database = params.get("database")
        table = params.get("table")

        try:
            granted_perms = []

            for permission in permissions:
                if table:
                    # Table-level permissions
                    query = (
                        f'GRANT {permission.upper()} ON TABLE "{table}" TO "{username}"'
                    )
                elif database:
                    # Database-level permissions
                    if permission.upper() == "CONNECT":
                        query = (
                            f'GRANT CONNECT ON DATABASE "{database}" TO "{username}"'
                        )
                    else:
                        query = f'GRANT {permission.upper()} ON DATABASE "{database}" TO "{username}"'
                else:
                    # System-level permissions
                    query = f'ALTER USER "{username}" {permission.upper()}'

                result = await self.execute_query(query)
                if result.success:
                    granted_perms.append(permission)

            return {
                "success": len(granted_perms) > 0,
                "username": username,
                "permissions_granted": granted_perms,
                "database": database,
                "table": table,
                "message": f"Granted {len(granted_perms)} permissions to '{username}'",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def estimate_cost(self, action: str, params: dict[str, Any]) -> CostEstimate:
        """Estimate cost of PostgreSQL operations."""

        # PostgreSQL is typically free/open source, so cost is mainly computational
        base_cost = 0.0

        if action in ["execute_query", "execute_many"]:
            # Estimate based on query complexity (very simplified)
            query = params.get("query", "")
            if any(
                keyword in query.upper() for keyword in ["JOIN", "GROUP BY", "ORDER BY"]
            ):
                base_cost = 0.01  # Complex query
            else:
                base_cost = 0.001  # Simple query
        elif action in ["create_database", "create_table", "create_index"]:
            base_cost = 0.05  # Schema operations
        elif action in ["backup_database", "restore_database"]:
            base_cost = 0.10  # I/O intensive operations

        return CostEstimate(estimated_cost=base_cost, currency="USD", confidence=0.7)

    async def _get_supported_actions(self) -> list[str]:
        """Get list of supported PostgreSQL actions."""
        return [
            "execute_query",
            "execute_many",
            "create_database",
            "create_table",
            "create_index",
            "backup_database",
            "restore_database",
            "analyze_performance",
            "get_statistics",
            "create_user",
            "grant_permissions",
        ]

    async def _execute_rollback(self, execution_id: str) -> dict[str, Any]:
        """Execute rollback operation for PostgreSQL."""
        # TODO: Implement rollback logic
        return {"message": f"Rollback not implemented for execution {execution_id}"}

    async def _create_client(self) -> PostgreSQLConnection:
        """Create PostgreSQL connection client."""
        if self.db_config is None:
            raise ToolError("Database configuration not initialized")
        return PostgreSQLConnection(self.db_config)

    async def _create_validator(self) -> None:
        """Create parameter validator (not needed for database tools)."""
        return None
