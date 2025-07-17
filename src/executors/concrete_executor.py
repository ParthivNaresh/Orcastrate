"""
Concrete executor implementation for development environment creation.

This executor integrates with actual tools to execute plans and create
real development environments.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict

from ..tools.base import ToolConfig
from ..tools.database import MongoDBTool, MySQLTool, PostgreSQLTool, RedisTool
from ..tools.docker import DockerTool
from ..tools.filesystem import FileSystemTool
from ..tools.git import GitTool
from ..tools.terraform import TerraformTool
from .base import (
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    Executor,
    ExecutorConfig,
    ExecutorError,
    Plan,
    StepExecution,
)


class ConcreteExecutor(Executor):
    """
    Concrete executor that uses real tools to execute plans.

    This executor integrates with Docker, Git, File System, and other tools
    to actually create development environments.
    """

    def __init__(self, config: ExecutorConfig):
        super().__init__(config)
        self._tools: Dict[str, Any] = {}
        self._tool_configs: Dict[str, ToolConfig] = {}

    async def initialize(self) -> None:
        """Initialize the executor and its tools."""
        await super().initialize()

        # Initialize tool configurations
        self._tool_configs = {
            "docker": ToolConfig(
                name="docker",
                version="1.0.0",
                timeout=self.config.step_timeout,
                retry_count=self.config.retry_policy.get("max_retries", 3),
            ),
            "git": ToolConfig(
                name="git",
                version="1.0.0",
                timeout=self.config.step_timeout,
                retry_count=self.config.retry_policy.get("max_retries", 3),
            ),
            "filesystem": ToolConfig(
                name="filesystem",
                version="1.0.0",
                timeout=self.config.step_timeout,
                retry_count=self.config.retry_policy.get("max_retries", 3),
            ),
            "postgresql": ToolConfig(
                name="postgresql",
                version="1.0.0",
                timeout=self.config.step_timeout,
                retry_count=self.config.retry_policy.get("max_retries", 3),
            ),
            "mysql": ToolConfig(
                name="mysql",
                version="1.0.0",
                timeout=self.config.step_timeout,
                retry_count=self.config.retry_policy.get("max_retries", 3),
            ),
            "mongodb": ToolConfig(
                name="mongodb",
                version="1.0.0",
                timeout=self.config.step_timeout,
                retry_count=self.config.retry_policy.get("max_retries", 3),
            ),
            "redis": ToolConfig(
                name="redis",
                version="1.0.0",
                timeout=self.config.step_timeout,
                retry_count=self.config.retry_policy.get("max_retries", 3),
            ),
            "terraform": ToolConfig(
                name="terraform",
                version="1.0.0",
                timeout=self.config.step_timeout,
                retry_count=self.config.retry_policy.get("max_retries", 3),
            ),
        }

        # Initialize tools
        await self._initialize_tools()

    async def _initialize_tools(self) -> None:
        """Initialize all available tools."""
        try:
            # Initialize Docker tool
            docker_tool = DockerTool(self._tool_configs["docker"])
            await docker_tool.initialize()
            self._tools["docker"] = docker_tool
        except Exception as e:
            self.logger.warning(f"Docker tool initialization failed: {e}")
            # Continue without Docker - some plans might not need it

        try:
            # Initialize Git tool
            git_tool = GitTool(self._tool_configs["git"])
            await git_tool.initialize()
            self._tools["git"] = git_tool
        except Exception as e:
            self.logger.warning(f"Git tool initialization failed: {e}")

        try:
            # Initialize File System tool
            filesystem_tool = FileSystemTool(self._tool_configs["filesystem"])
            await filesystem_tool.initialize()
            self._tools["filesystem"] = filesystem_tool
        except Exception as e:
            self.logger.error(f"File System tool initialization failed: {e}")
            # File system is critical - raise error
            raise ExecutorError(f"Critical tool initialization failed: {e}")

        # Initialize database tools (optional - continue without them if they fail)
        await self._initialize_database_tools()

        # Initialize infrastructure tools
        await self._initialize_infrastructure_tools()

        # self.logger.info(
        #     f"Initialized {len(self._tools)} tools: {list(self._tools.keys())}"
        # )

    async def _initialize_database_tools(self) -> None:
        """Initialize database tools with default configurations."""
        # PostgreSQL Tool
        try:
            # Configure environment for PostgreSQL tool
            postgresql_tool_config = self._tool_configs["postgresql"]
            postgresql_tool_config.environment = {
                "host": self.config.get("postgresql", {}).get("host", "localhost"),
                "port": self.config.get("postgresql", {}).get("port", 5432),
                "database": self.config.get("postgresql", {}).get(
                    "database", "postgres"
                ),
                "username": self.config.get("postgresql", {}).get(
                    "username", "postgres"
                ),
                "password": self.config.get("postgresql", {}).get("password", ""),
                "connection_timeout": 30,
                "max_connections": 10,
            }
            postgresql_tool = PostgreSQLTool(postgresql_tool_config)
            await postgresql_tool.initialize()
            self._tools["postgresql"] = postgresql_tool
            self.logger.info("PostgreSQL tool initialized successfully")
        except Exception as e:
            pass

        # MySQL Tool
        try:
            # Configure environment for MySQL tool
            mysql_tool_config = self._tool_configs["mysql"]
            mysql_tool_config.environment = {
                "host": self.config.get("mysql", {}).get("host", "localhost"),
                "port": self.config.get("mysql", {}).get("port", 3306),
                "database": self.config.get("mysql", {}).get("database", "mysql"),
                "username": self.config.get("mysql", {}).get("username", "root"),
                "password": self.config.get("mysql", {}).get("password", ""),
                "connection_timeout": 30,
                "max_connections": 10,
            }
            mysql_tool = MySQLTool(mysql_tool_config)
            await mysql_tool.initialize()
            self._tools["mysql"] = mysql_tool
            self.logger.info("MySQL tool initialized successfully")
        except Exception as e:
            pass

        # MongoDB Tool
        try:
            # Configure environment for MongoDB tool
            mongodb_tool_config = self._tool_configs["mongodb"]
            mongodb_tool_config.environment = {
                "host": self.config.get("mongodb", {}).get("host", "localhost"),
                "port": self.config.get("mongodb", {}).get("port", 27017),
                "database": self.config.get("mongodb", {}).get("database", "admin"),
                "username": self.config.get("mongodb", {}).get("username", ""),
                "password": self.config.get("mongodb", {}).get("password", ""),
                "connection_timeout": 30,
                "max_connections": 10,
            }
            mongodb_tool = MongoDBTool(mongodb_tool_config)
            await mongodb_tool.initialize()
            self._tools["mongodb"] = mongodb_tool
            self.logger.info("MongoDB tool initialized successfully")
        except Exception as e:
            pass

        # Redis Tool
        try:
            redis_tool_config = self._tool_configs["redis"]
            redis_tool_config.environment = {
                "host": self.config.get("redis", {}).get("host", "localhost"),
                "port": self.config.get("redis", {}).get("port", 6379),
                "database": self.config.get("redis", {}).get("database", "0"),
                "username": self.config.get("redis", {}).get("username", ""),
                "password": self.config.get("redis", {}).get("password", ""),
                "connection_timeout": 30,
                "max_connections": 10,
                "database_number": self.config.get("redis", {}).get(
                    "database_number", 0
                ),
            }
            redis_tool = RedisTool(redis_tool_config)
            await redis_tool.initialize()
            self._tools["redis"] = redis_tool
        except Exception as e:
            pass

    async def _initialize_infrastructure_tools(self) -> None:
        """Initialize infrastructure tools with default configurations."""
        # Terraform Tool
        try:
            terraform_tool_config = self._tool_configs["terraform"]
            terraform_tool_config.environment = {
                "working_dir": self.config.get("terraform", {}).get("working_dir"),
                "terraform_version": self.config.get("terraform", {}).get(
                    "terraform_version", "latest"
                ),
                "backend_config": self.config.get("terraform", {}).get(
                    "backend_config", {}
                ),
                "var_files": self.config.get("terraform", {}).get("var_files", []),
                "variables": self.config.get("terraform", {}).get("variables", {}),
                "parallelism": self.config.get("terraform", {}).get("parallelism", 10),
                "auto_approve": self.config.get("terraform", {}).get(
                    "auto_approve", False
                ),
            }
            terraform_tool = TerraformTool(terraform_tool_config)
            await terraform_tool.initialize()
            self._tools["terraform"] = terraform_tool
        except Exception as e:
            pass

    async def _execute_plan_with_strategy(
        self, plan: Plan, context: ExecutionContext
    ) -> ExecutionResult:
        """Execute plan using the configured strategy."""
        start_time = datetime.utcnow()

        try:
            # Create dependency graph and sort steps
            dependency_graph = await self._create_dependency_graph(plan)
            execution_order = await self._topological_sort(dependency_graph)

            # Execute steps in dependency order
            executed_count = 0
            failed_steps = []

            for step_id in execution_order:
                if context.cancellation_requested:
                    break

                # Find the step definition
                step = next((s for s in plan.steps if s["id"] == step_id), None)
                if not step:
                    continue

                # Check if dependencies are satisfied
                if not await self._check_dependencies(step, context):
                    failed_steps.append(step_id)
                    continue

                # Execute the step
                step_execution = await self._execute_step(step, context)
                executed_count += 1

                # Update progress
                progress = executed_count / len(plan.steps)
                context.metrics["progress"] = progress

                # Log step completion
                if step_execution.status == ExecutionStatus.COMPLETED:
                    pass
                elif step_execution.status == ExecutionStatus.FAILED:
                    failed_steps.append(step_id)

                    # Stop on failure unless configured otherwise
                    if not self.config.get("continue_on_failure", False):
                        break

                # Add some delay between steps for stability
                await asyncio.sleep(0.1)

            # Determine overall success
            total_steps = len(plan.steps)
            successful_steps = sum(
                1
                for step_exec in context.steps.values()
                if step_exec.status == ExecutionStatus.COMPLETED
            )

            success = len(failed_steps) == 0 and successful_steps == total_steps

            # Create execution result
            result = ExecutionResult(
                success=success,
                execution_id=context.execution_id,
                artifacts=context.artifacts,
                logs=context.logs,
                metrics={
                    **context.metrics,
                    "total_steps": total_steps,
                    "successful_steps": successful_steps,
                    "failed_steps": len(failed_steps),
                    "execution_time": (datetime.utcnow() - start_time).total_seconds(),
                },
                duration=(datetime.utcnow() - start_time).total_seconds(),
            )

            if success:
                pass
            else:
                result.error = f"Plan execution failed. {successful_steps}/{total_steps} steps completed. Failed steps: {failed_steps}"

            return result

        except Exception as e:
            return ExecutionResult(
                success=False,
                execution_id=context.execution_id,
                error=str(e),
                duration=(datetime.utcnow() - start_time).total_seconds(),
            )

    async def _execute_step(
        self, step: Dict[str, Any], context: ExecutionContext
    ) -> StepExecution:
        """Execute a single step using the appropriate tool."""
        step_id = step["id"]
        step_execution = context.steps[step_id]

        try:

            step_execution.status = ExecutionStatus.RUNNING
            step_execution.start_time = datetime.utcnow()

            # Get the tool for this step
            tool_name = step.get("tool")
            if not tool_name:
                raise ExecutorError(f"No tool specified for step: {step_id}")

            if tool_name not in self._tools:
                raise ExecutorError(f"Tool not available: {tool_name}")

            tool = self._tools[tool_name]
            action = step.get("action")
            parameters = step.get("parameters", {})

            # Execute the tool action
            result = await tool.execute(action, parameters)

            if result.success:
                step_execution.status = ExecutionStatus.COMPLETED
                step_execution.result = {
                    "tool_output": result.output,
                    "duration": result.duration,
                    "metadata": result.metadata,
                }

                # Store any artifacts
                if result.output:
                    context.artifacts[step_id] = result.output

            else:
                step_execution.status = ExecutionStatus.FAILED
                step_execution.error = result.error or "Tool execution failed"

            step_execution.end_time = datetime.utcnow()

        except Exception as e:
            step_execution.status = ExecutionStatus.FAILED
            step_execution.end_time = datetime.utcnow()
            step_execution.error = str(e)

        return step_execution

    async def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available tools."""
        tools_info = {}

        for tool_name, tool in self._tools.items():
            try:
                schema = await tool.get_schema()
                tools_info[tool_name] = {
                    "name": schema.name,
                    "description": schema.description,
                    "version": schema.version,
                    "actions": list(schema.actions.keys()),
                    "status": "available",
                }
            except Exception as e:
                tools_info[tool_name] = {
                    "name": tool_name,
                    "status": "error",
                    "error": str(e),
                }

        return tools_info

    async def validate_plan_requirements(self, plan: Plan) -> Dict[str, Any]:
        """Validate that all plan requirements can be met."""
        validation_results: Dict[str, Any] = {
            "valid": True,
            "missing_tools": [],
            "invalid_actions": [],
            "warnings": [],
        }

        # Check each step's tool and action
        for step in plan.steps:
            tool_name = step.get("tool")
            action = step.get("action")

            if not tool_name:
                validation_results["warnings"].append(
                    f"Step {step['id']} has no tool specified"
                )
                continue

            # Check if tool is available
            if tool_name not in self._tools:
                validation_results["missing_tools"].append(tool_name)
                validation_results["valid"] = False
                continue

            # Check if action is supported
            try:
                tool = self._tools[tool_name]
                supported_actions = await tool._get_supported_actions()
                if action not in supported_actions:
                    validation_results["invalid_actions"].append(
                        {
                            "step": step["id"],
                            "tool": tool_name,
                            "action": action,
                            "supported": supported_actions,
                        }
                    )
                    validation_results["valid"] = False
            except Exception as e:
                validation_results["warnings"].append(
                    f"Could not validate tool {tool_name}: {e}"
                )

        return validation_results

    async def get_execution_summary(self, context: ExecutionContext) -> Dict[str, Any]:
        """Get a summary of the execution."""
        total_steps = len(context.steps)
        completed_steps = sum(
            1
            for step in context.steps.values()
            if step.status == ExecutionStatus.COMPLETED
        )
        failed_steps = sum(
            1
            for step in context.steps.values()
            if step.status == ExecutionStatus.FAILED
        )
        running_steps = sum(
            1
            for step in context.steps.values()
            if step.status == ExecutionStatus.RUNNING
        )

        # Calculate duration
        start_time = context.start_time
        end_time = context.end_time or datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        return {
            "execution_id": context.execution_id,
            "status": context.status.value,
            "progress": context.progress,
            "duration": duration,
            "steps": {
                "total": total_steps,
                "completed": completed_steps,
                "failed": failed_steps,
                "running": running_steps,
                "pending": total_steps - completed_steps - failed_steps - running_steps,
            },
            "artifacts": list(context.artifacts.keys()),
            "metrics": context.metrics,
        }
