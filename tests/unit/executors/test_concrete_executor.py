"""
Unit tests for ConcreteExecutor module.

Tests cover executor initialization, tool management, plan execution,
and validation functionality.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.executors.base import (
    ExecutionContext,
    ExecutionStatus,
    ExecutionStrategy,
    ExecutorConfig,
    ExecutorError,
    StepExecution,
)
from src.executors.concrete_executor import ConcreteExecutor
from src.logging_utils.log_manager import LogManager
from src.logging_utils.progress_tracker import ProgressTracker
from src.tools.base import ToolResult, ToolSchema


class TestConcreteExecutor:
    """Test the ConcreteExecutor class."""

    @pytest.fixture
    def executor_config(self):
        """Create a basic executor configuration."""
        return ExecutorConfig(
            strategy=ExecutionStrategy.SEQUENTIAL,
            max_concurrent_steps=5,
            step_timeout=300,
            retry_policy={"max_retries": 3, "backoff_factor": 2.0, "max_delay": 60},
            enable_rollback=True,
        )

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create a mock progress tracker for testing."""
        mock_log_manager = Mock(spec=LogManager)
        mock_log_manager.emit_event = AsyncMock()

        progress_tracker = Mock(spec=ProgressTracker)
        progress_tracker.log_manager = mock_log_manager
        progress_tracker.update_step_progress = Mock()
        progress_tracker.add_step_message = Mock()
        progress_tracker.log_step_success = Mock()
        progress_tracker.log_step_failure = Mock()
        progress_tracker.log_step_conditional = Mock()
        return progress_tracker

    @pytest.fixture
    def executor(self, executor_config, mock_progress_tracker):
        """Create a ConcreteExecutor instance."""
        return ConcreteExecutor(executor_config, progress_tracker=mock_progress_tracker)

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""
        tool = AsyncMock()
        tool.initialize = AsyncMock()
        tool.get_schema = AsyncMock(
            return_value=ToolSchema(
                name="test_tool",
                description="Test tool",
                version="1.0.0",
                actions={"test_action": {"description": "Test action"}},
            )
        )
        tool.execute = AsyncMock()
        tool.validate = AsyncMock()
        tool.estimate_cost = AsyncMock()
        tool._get_supported_actions = AsyncMock(return_value=["test_action"])
        return tool

    @pytest.fixture
    def sample_plan(self):
        """Create a sample plan for testing."""
        plan = Mock()
        plan.id = "test-plan-123"
        plan.steps = [
            {
                "id": "step-1",
                "name": "Setup Docker",
                "tool": "docker",
                "action": "build_image",
                "parameters": {"tag": "test-image", "path": "/tmp/test"},
                "estimated_duration": 60,
            },
            {
                "id": "step-2",
                "name": "Start Container",
                "tool": "docker",
                "action": "run_container",
                "parameters": {"image": "test-image", "name": "test-container"},
                "estimated_duration": 30,
            },
        ]
        plan.estimated_duration = 90
        plan.estimated_cost = 2.50
        plan.risk_assessment = {"risk_level": "low", "confidence": 0.9}
        return plan

    def test_executor_initialization(self, executor_config):
        """Test executor initialization with config."""
        executor = ConcreteExecutor(executor_config)

        assert executor.config == executor_config
        assert executor._tools == {}
        assert executor._tool_configs == {}

    @pytest.mark.asyncio
    async def test_initialize_success(self, executor):
        """Test successful executor initialization."""
        # Mock all tool classes
        with (
            patch("src.executors.concrete_executor.DockerTool") as mock_docker,
            patch("src.executors.concrete_executor.GitTool") as mock_git,
            patch("src.executors.concrete_executor.FileSystemTool") as mock_fs,
            patch("src.executors.concrete_executor.PostgreSQLTool") as mock_pg,
            patch("src.executors.concrete_executor.MySQLTool") as mock_mysql,
            patch("src.executors.concrete_executor.MongoDBTool") as mock_mongo,
            patch("src.executors.concrete_executor.RedisTool") as mock_redis,
            patch("src.executors.concrete_executor.TerraformTool") as mock_terraform,
        ):

            # Create mock tool instances
            mock_docker_instance = AsyncMock()
            mock_git_instance = AsyncMock()
            mock_fs_instance = AsyncMock()
            mock_pg_instance = AsyncMock()
            mock_mysql_instance = AsyncMock()
            mock_mongo_instance = AsyncMock()
            mock_redis_instance = AsyncMock()
            mock_terraform_instance = AsyncMock()

            # Configure mock classes to return instances
            mock_docker.return_value = mock_docker_instance
            mock_git.return_value = mock_git_instance
            mock_fs.return_value = mock_fs_instance
            mock_pg.return_value = mock_pg_instance
            mock_mysql.return_value = mock_mysql_instance
            mock_mongo.return_value = mock_mongo_instance
            mock_redis.return_value = mock_redis_instance
            mock_terraform.return_value = mock_terraform_instance

            # Mock initialize methods
            mock_docker_instance.initialize = AsyncMock()
            mock_git_instance.initialize = AsyncMock()
            mock_fs_instance.initialize = AsyncMock()
            mock_pg_instance.initialize = AsyncMock()
            mock_mysql_instance.initialize = AsyncMock()
            mock_mongo_instance.initialize = AsyncMock()
            mock_redis_instance.initialize = AsyncMock()
            mock_terraform_instance.initialize = AsyncMock()

            await executor.initialize()

            # Verify tool configurations were created
            assert "docker" in executor._tool_configs
            assert "git" in executor._tool_configs
            assert "filesystem" in executor._tool_configs
            assert "postgresql" in executor._tool_configs
            assert "mysql" in executor._tool_configs
            assert "mongodb" in executor._tool_configs
            assert "redis" in executor._tool_configs
            assert "terraform" in executor._tool_configs

            # Verify tools were initialized
            mock_docker_instance.initialize.assert_called_once()
            mock_git_instance.initialize.assert_called_once()
            mock_fs_instance.initialize.assert_called_once()
            mock_pg_instance.initialize.assert_called_once()
            mock_mysql_instance.initialize.assert_called_once()
            mock_mongo_instance.initialize.assert_called_once()
            mock_redis_instance.initialize.assert_called_once()
            mock_terraform_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_filesystem_failure(self, executor):
        """Test initialization failure when filesystem tool fails."""
        with patch("src.executors.concrete_executor.FileSystemTool") as mock_fs:
            mock_fs.return_value.initialize.side_effect = Exception(
                "FileSystem init failed"
            )

            with pytest.raises(
                ExecutorError, match="Critical tool initialization failed"
            ):
                await executor.initialize()

    @pytest.mark.asyncio
    async def test_initialize_optional_tool_failure(self, executor):
        """Test initialization continues when optional tools fail."""
        with (
            patch("src.executors.concrete_executor.DockerTool") as mock_docker,
            patch("src.executors.concrete_executor.GitTool") as mock_git,
            patch("src.executors.concrete_executor.FileSystemTool") as mock_fs,
            patch("src.executors.concrete_executor.PostgreSQLTool") as mock_pg,
        ):

            # Make Docker fail but FileSystem succeed
            mock_docker.return_value.initialize.side_effect = Exception(
                "Docker not available"
            )
            mock_git.return_value.initialize.side_effect = Exception(
                "Git not available"
            )
            mock_fs.return_value.initialize = AsyncMock()
            mock_pg.return_value.initialize.side_effect = Exception(
                "PostgreSQL not available"
            )

            # Should not raise exception, just continue without optional tools
            await executor.initialize()

            # Verify filesystem tool was still initialized
            mock_fs.return_value.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_available_tools_success(self, executor, mock_tool):
        """Test successful tool information retrieval."""
        # Add mock tool to executor
        executor._tools = {"test_tool": mock_tool}

        result = await executor.get_available_tools()

        assert "test_tool" in result
        assert result["test_tool"]["name"] == "test_tool"
        assert result["test_tool"]["description"] == "Test tool"
        assert result["test_tool"]["version"] == "1.0.0"
        assert result["test_tool"]["status"] == "available"
        assert "test_action" in result["test_tool"]["actions"]

    @pytest.mark.asyncio
    async def test_get_available_tools_with_error(self, executor, mock_tool):
        """Test tool information retrieval with tool error."""
        # Make tool schema fail
        mock_tool.get_schema.side_effect = Exception("Schema error")
        executor._tools = {"test_tool": mock_tool}

        result = await executor.get_available_tools()

        assert "test_tool" in result
        assert result["test_tool"]["status"] == "error"
        assert result["test_tool"]["error"] == "Schema error"

    @pytest.mark.asyncio
    async def test_validate_plan_requirements_success(
        self, executor, mock_tool, sample_plan
    ):
        """Test successful plan validation."""
        # Setup mock tool to validate successfully
        executor._tools = {"docker": mock_tool}
        mock_tool.validate.return_value = {"valid": True}
        mock_tool._get_supported_actions.return_value = ["build_image", "run_container"]

        result = await executor.validate_plan_requirements(sample_plan)

        assert result["valid"] is True
        assert result["missing_tools"] == []
        assert result["invalid_actions"] == []
        assert isinstance(result["warnings"], list)

    @pytest.mark.asyncio
    async def test_validate_plan_requirements_missing_tools(
        self, executor, sample_plan
    ):
        """Test plan validation with missing tools."""
        # No tools available
        executor._tools = {}

        result = await executor.validate_plan_requirements(sample_plan)

        assert result["valid"] is False
        assert "docker" in result["missing_tools"]

    @pytest.mark.asyncio
    async def test_validate_plan_requirements_invalid_actions(
        self, executor, mock_tool, sample_plan
    ):
        """Test plan validation with invalid actions."""
        # Setup mock tool to report invalid actions
        executor._tools = {"docker": mock_tool}
        mock_tool._get_supported_actions.return_value = [
            "invalid_action"
        ]  # doesn't include build_image

        result = await executor.validate_plan_requirements(sample_plan)

        assert result["valid"] is False
        assert len(result["invalid_actions"]) > 0

    @pytest.mark.asyncio
    async def test_execute_step_success(self, executor, mock_tool):
        """Test successful step execution."""
        # Setup mock tool
        executor._tools = {"docker": mock_tool}
        mock_tool.execute.return_value = ToolResult(
            success=True,
            tool_name="docker",
            action="build_image",
            output={
                "message": "Image built successfully",
                "artifacts": ["test-image:latest"],
            },
        )

        step = {
            "id": "step-1",
            "tool": "docker",
            "action": "build_image",
            "parameters": {"tag": "test-image", "path": "/tmp/test"},
        }

        # Create proper execution context
        context = ExecutionContext(
            execution_id="test-exec-123",
            plan_id="test-plan-123",
            strategy=ExecutionStrategy.SEQUENTIAL,
            start_time=datetime.utcnow(),
            steps={"step-1": StepExecution(step_id="step-1")},
        )

        result = await executor._execute_step(step, context)

        assert result.status == ExecutionStatus.COMPLETED
        mock_tool.execute.assert_called_once_with(
            "build_image", {"tag": "test-image", "path": "/tmp/test"}
        )

    @pytest.mark.asyncio
    async def test_execute_step_missing_tool(self, executor):
        """Test step execution with missing tool."""
        # No tools available
        executor._tools = {}

        step = {
            "id": "step-1",
            "tool": "docker",
            "action": "build_image",
            "parameters": {"tag": "test-image"},
        }

        # Create proper execution context
        context = ExecutionContext(
            execution_id="test-exec-123",
            plan_id="test-plan-123",
            strategy=ExecutionStrategy.SEQUENTIAL,
            start_time=datetime.utcnow(),
            steps={"step-1": StepExecution(step_id="step-1")},
        )

        result = await executor._execute_step(step, context)

        assert result.status == ExecutionStatus.FAILED
        assert "Tool not available" in result.error

    @pytest.mark.asyncio
    async def test_execute_step_tool_error(self, executor, mock_tool):
        """Test step execution with tool error."""
        # Setup mock tool to fail
        executor._tools = {"docker": mock_tool}
        mock_tool.execute.side_effect = Exception("Tool execution failed")

        step = {
            "id": "step-1",
            "tool": "docker",
            "action": "build_image",
            "parameters": {"tag": "test-image"},
        }

        # Create proper execution context
        context = ExecutionContext(
            execution_id="test-exec-123",
            plan_id="test-plan-123",
            strategy=ExecutionStrategy.SEQUENTIAL,
            start_time=datetime.utcnow(),
            steps={"step-1": StepExecution(step_id="step-1")},
        )

        result = await executor._execute_step(step, context)

        assert result.status == ExecutionStatus.FAILED
        assert "Tool execution failed" in result.error

    @pytest.mark.asyncio
    async def test_get_execution_summary(self, executor):
        """Test execution summary generation."""
        # Create proper execution context
        context = ExecutionContext(
            execution_id="test-exec-123",
            plan_id="test-plan",
            strategy=ExecutionStrategy.SEQUENTIAL,
            start_time=datetime(2023, 1, 1, 10, 0, 0),
            end_time=datetime(2023, 1, 1, 10, 2, 0),
            steps={
                "step-1": StepExecution(
                    step_id="step-1", status=ExecutionStatus.COMPLETED
                ),
                "step-2": StepExecution(
                    step_id="step-2", status=ExecutionStatus.COMPLETED
                ),
            },
        )

        result = await executor.get_execution_summary(context)

        assert result["execution_id"] == "test-exec-123"
        assert result["steps"]["total"] == 2
        assert result["steps"]["completed"] == 2
        assert result["steps"]["failed"] == 0
        assert result["duration"] == 120.0  # 2 minutes

    @pytest.mark.asyncio
    async def test_get_execution_summary_with_failures(self, executor):
        """Test execution summary with failed steps."""
        # Create proper execution context with failures
        context = ExecutionContext(
            execution_id="test-exec-123",
            plan_id="test-plan",
            strategy=ExecutionStrategy.SEQUENTIAL,
            start_time=datetime(2023, 1, 1, 10, 0, 0),
            end_time=datetime(2023, 1, 1, 10, 1, 0),
            steps={
                "step-1": StepExecution(
                    step_id="step-1", status=ExecutionStatus.COMPLETED
                ),
                "step-2": StepExecution(
                    step_id="step-2", status=ExecutionStatus.FAILED, error="Step failed"
                ),
            },
        )

        result = await executor.get_execution_summary(context)

        assert result["execution_id"] == "test-exec-123"
        assert result["steps"]["total"] == 2
        assert result["steps"]["completed"] == 1
        assert result["steps"]["failed"] == 1
        assert result["duration"] == 60.0  # 1 minute


class TestConcreteExecutorIntegration:
    """Integration tests for ConcreteExecutor workflows."""

    @pytest.fixture
    def executor_config(self):
        """Create executor configuration for integration tests."""
        return ExecutorConfig(
            strategy=ExecutionStrategy.SEQUENTIAL,
            max_concurrent_steps=3,
            step_timeout=120,
            retry_policy={"max_retries": 2, "backoff_factor": 1.5, "max_delay": 30},
            enable_rollback=True,
        )

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create a mock progress tracker for testing."""
        mock_log_manager = Mock(spec=LogManager)
        mock_log_manager.emit_event = AsyncMock()

        progress_tracker = Mock(spec=ProgressTracker)
        progress_tracker.log_manager = mock_log_manager
        progress_tracker.update_step_progress = Mock()
        progress_tracker.add_step_message = Mock()
        progress_tracker.log_step_success = Mock()
        progress_tracker.log_step_failure = Mock()
        progress_tracker.log_step_conditional = Mock()
        return progress_tracker

    @pytest.fixture
    def executor(self, executor_config, mock_progress_tracker):
        """Create ConcreteExecutor for integration tests."""
        return ConcreteExecutor(executor_config, progress_tracker=mock_progress_tracker)

    @pytest.fixture
    def complex_plan(self):
        """Create a complex plan for integration testing."""
        plan = Mock()
        plan.id = "complex-plan-456"
        plan.steps = [
            {
                "id": "step-1",
                "name": "Initialize Git Repository",
                "tool": "git",
                "action": "init",
                "parameters": {"path": "/tmp/test-repo"},
                "estimated_duration": 10,
            },
            {
                "id": "step-2",
                "name": "Create Dockerfile",
                "tool": "filesystem",
                "action": "write_file",
                "parameters": {
                    "path": "/tmp/test-repo/Dockerfile",
                    "content": "FROM node:16\nWORKDIR /app\nCOPY . .\nRUN npm install\nCMD ['npm', 'start']",
                },
                "estimated_duration": 5,
            },
            {
                "id": "step-3",
                "name": "Build Docker Image",
                "tool": "docker",
                "action": "build_image",
                "parameters": {"tag": "test-app:latest", "path": "/tmp/test-repo"},
                "estimated_duration": 120,
            },
            {
                "id": "step-4",
                "name": "Start Database",
                "tool": "postgresql",
                "action": "start_database",
                "parameters": {"name": "test-db", "port": 5432},
                "estimated_duration": 30,
            },
        ]
        plan.estimated_duration = 165
        plan.estimated_cost = 5.00
        plan.risk_assessment = {"risk_level": "medium", "confidence": 0.8}
        return plan

    @pytest.mark.asyncio
    async def test_full_workflow_validation(self, executor, complex_plan):
        """Test complete workflow from validation to execution summary."""
        # Mock all tools
        mock_tools = {}
        tool_actions = {
            "git": ["init", "clone", "commit"],
            "filesystem": ["write_file", "read_file", "create_directory"],
            "docker": ["build_image", "run_container", "stop_container"],
            "postgresql": ["start_database", "stop_database", "create_database"],
        }

        for tool_name in ["git", "filesystem", "docker", "postgresql"]:
            mock_tool = AsyncMock()
            actions = tool_actions[tool_name]
            mock_tool.get_schema.return_value = ToolSchema(
                name=tool_name,
                description=f"{tool_name} tool",
                version="1.0.0",
                actions={
                    action: {"description": f"{action} action"} for action in actions
                },
            )
            mock_tool.validate.return_value = {"valid": True}
            mock_tool._get_supported_actions.return_value = actions
            mock_tools[tool_name] = mock_tool

        executor._tools = mock_tools

        # Test validation
        validation_result = await executor.validate_plan_requirements(complex_plan)
        assert validation_result["valid"] is True

        # Test tool availability
        tools_info = await executor.get_available_tools()
        assert len(tools_info) == 4
        assert all(tool["status"] == "available" for tool in tools_info.values())

    @pytest.mark.asyncio
    async def test_partial_tool_availability(self, executor, complex_plan):
        """Test workflow with only some tools available."""
        # Only make some tools available
        executor._tools = {
            "git": AsyncMock(),
            "filesystem": AsyncMock(),
            # docker and postgresql missing
        }

        # Configure available tools
        for tool_name, tool in executor._tools.items():
            tool.get_schema.return_value = ToolSchema(
                name=tool_name,
                description=f"{tool_name} tool",
                version="1.0.0",
                actions={f"{tool_name}_action": {"description": f"{tool_name} action"}},
            )

        # Test validation should fail due to missing tools
        validation_result = await executor.validate_plan_requirements(complex_plan)
        assert validation_result["valid"] is False
        assert "docker" in validation_result["missing_tools"]
        assert "postgresql" in validation_result["missing_tools"]

    @pytest.mark.asyncio
    async def test_tool_configuration_validation(self, executor):
        """Test tool configuration validation."""
        # Test that tool configs are properly created
        await executor.initialize()

        # All expected tool configs should be present
        expected_tools = [
            "docker",
            "git",
            "filesystem",
            "postgresql",
            "mysql",
            "mongodb",
            "redis",
            "terraform",
        ]

        for tool_name in expected_tools:
            assert tool_name in executor._tool_configs
            config = executor._tool_configs[tool_name]
            assert config.name == tool_name
            assert config.version == "1.0.0"
            assert config.timeout == executor.config.step_timeout
            assert config.retry_count == executor.config.retry_policy.get(
                "max_retries", 3
            )


class TestConcreteExecutorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def executor_config(self):
        """Create executor configuration for edge case tests."""
        return ExecutorConfig(
            strategy=ExecutionStrategy.PARALLEL,
            max_concurrent_steps=10,
            step_timeout=600,
            retry_policy={"max_retries": 5, "backoff_factor": 3.0, "max_delay": 300},
            enable_rollback=False,
        )

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create a mock progress tracker for testing."""
        mock_log_manager = Mock(spec=LogManager)
        mock_log_manager.emit_event = AsyncMock()

        progress_tracker = Mock(spec=ProgressTracker)
        progress_tracker.log_manager = mock_log_manager
        progress_tracker.update_step_progress = Mock()
        progress_tracker.add_step_message = Mock()
        progress_tracker.log_step_success = Mock()
        progress_tracker.log_step_failure = Mock()
        progress_tracker.log_step_conditional = Mock()
        return progress_tracker

    @pytest.fixture
    def executor(self, executor_config, mock_progress_tracker):
        """Create ConcreteExecutor for edge case tests."""
        return ConcreteExecutor(executor_config, progress_tracker=mock_progress_tracker)

    @pytest.mark.asyncio
    async def test_empty_plan_validation(self, executor):
        """Test validation of empty plan."""
        empty_plan = Mock()
        empty_plan.id = "empty-plan"
        empty_plan.steps = []

        result = await executor.validate_plan_requirements(empty_plan)

        assert result["valid"] is True
        assert result["missing_tools"] == []
        assert result["invalid_actions"] == []

    @pytest.mark.asyncio
    async def test_plan_with_unknown_tool(self, executor):
        """Test plan with unknown tool."""
        plan = Mock()
        plan.id = "unknown-tool-plan"
        plan.steps = [
            {
                "id": "step-1",
                "tool": "unknown_tool",
                "action": "unknown_action",
                "parameters": {},
            }
        ]

        result = await executor.validate_plan_requirements(plan)

        assert result["valid"] is False
        assert "unknown_tool" in result["missing_tools"]

    @pytest.mark.asyncio
    async def test_concurrent_initialization(self, executor):
        """Test concurrent initialization calls."""
        # This should not cause issues as initialization should be idempotent
        tasks = [executor.initialize() for _ in range(3)]
        await asyncio.gather(*tasks)

        # Should complete without errors
        assert len(executor._tool_configs) > 0

    @pytest.mark.asyncio
    async def test_tool_timeout_configuration(self, executor):
        """Test tool timeout configuration."""
        await executor.initialize()

        # All tools should have the same timeout as configured
        for tool_name, config in executor._tool_configs.items():
            assert config.timeout == executor.config.step_timeout
            assert config.retry_count == executor.config.retry_policy.get(
                "max_retries", 3
            )

    @pytest.mark.asyncio
    async def test_large_plan_validation(self, executor):
        """Test validation of large plan."""
        large_plan = Mock()
        large_plan.id = "large-plan"
        large_plan.steps = [
            {
                "id": f"step-{i}",
                "tool": "docker",
                "action": "build_image",
                "parameters": {"tag": f"image-{i}"},
            }
            for i in range(100)
        ]

        # Mock docker tool
        mock_docker = AsyncMock()
        mock_docker.validate.return_value = {"valid": True}
        mock_docker._get_supported_actions.return_value = ["build_image"]
        executor._tools = {"docker": mock_docker}

        result = await executor.validate_plan_requirements(large_plan)

        assert result["valid"] is True
        # _get_supported_actions should be called once for each step
        assert mock_docker._get_supported_actions.call_count == 100
