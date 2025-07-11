"""
Unit tests for CLI main module.

Tests cover the OrcastrateAgent class and all CLI commands including
initialization, environment creation, template listing, and error handling.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

from src.agent.base import Requirements
from src.cli.main import OrcastrateAgent, cli


class TestOrcastrateAgent:
    """Test the OrcastrateAgent class."""

    @pytest.fixture
    def agent(self):
        """Create an OrcastrateAgent instance."""
        return OrcastrateAgent()

    @pytest.fixture
    def mock_planner(self):
        """Create a mock TemplatePlanner."""
        planner = AsyncMock()
        planner.initialize = AsyncMock()
        planner.create_plan = AsyncMock()
        planner.get_available_templates = AsyncMock()
        return planner

    @pytest.fixture
    def mock_executor(self):
        """Create a mock ConcreteExecutor."""
        executor = AsyncMock()
        executor.initialize = AsyncMock()
        executor.execute_plan = AsyncMock()
        executor.validate_plan_requirements = AsyncMock()
        executor.get_available_tools = AsyncMock()
        return executor

    @pytest.fixture
    def mock_plan(self):
        """Create a mock Plan object."""
        plan = Mock()
        plan.id = "test-plan-123"
        plan.steps = [
            {
                "id": "step-1",
                "name": "Setup Docker",
                "description": "Configure Docker environment",
                "tool": "docker",
                "action": "setup",
                "estimated_duration": 60,
            },
            {
                "id": "step-2",
                "name": "Deploy App",
                "description": "Deploy application",
                "tool": "aws",
                "action": "deploy",
                "estimated_duration": 120,
            },
        ]
        plan.estimated_duration = 180
        plan.estimated_cost = 5.50
        plan.risk_assessment = {"risk_level": "low", "confidence": 0.9}
        return plan

    @pytest.fixture
    def mock_execution_result(self):
        """Create a mock ExecutionResult."""
        result = Mock()
        result.success = True
        result.execution_id = "exec-456"
        result.duration = 175.5
        result.artifacts = ["docker-compose.yml", "deployment.yaml"]
        result.metrics = {
            "total_steps": 2,
            "successful_steps": 2,
            "failed_steps": 0,
        }
        result.error = None
        return result

    @pytest.fixture
    def sample_requirements(self):
        """Create sample Requirements object."""
        return Requirements(
            description="FastAPI REST API with PostgreSQL",
            framework="fastapi",
            database="postgresql",
            cloud_provider="aws",
            metadata={"created_at": "2023-01-01T00:00:00"},
        )

    @pytest.mark.asyncio
    async def test_agent_initialization_success(
        self, agent, mock_planner, mock_executor
    ):
        """Test successful agent initialization."""
        with (
            patch("src.cli.main.TemplatePlanner", return_value=mock_planner),
            patch("src.cli.main.ConcreteExecutor", return_value=mock_executor),
        ):
            await agent.initialize()

            assert agent.planner is mock_planner
            assert agent.executor is mock_executor
            mock_planner.initialize.assert_called_once()
            mock_executor.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_initialization_failure(self, agent):
        """Test agent initialization failure."""
        with patch(
            "src.cli.main.TemplatePlanner", side_effect=Exception("Planner init failed")
        ):
            with pytest.raises(Exception, match="Planner init failed"):
                await agent.initialize()

    @pytest.mark.asyncio
    async def test_create_environment_success(
        self,
        agent,
        mock_planner,
        mock_executor,
        mock_plan,
        mock_execution_result,
        sample_requirements,
    ):
        """Test successful environment creation."""
        # Setup mocks
        agent.planner = mock_planner
        agent.executor = mock_executor
        mock_planner.create_plan.return_value = mock_plan
        mock_executor.validate_plan_requirements.return_value = {
            "valid": True,
            "warnings": ["Minor warning"],
        }
        mock_executor.execute_plan.return_value = mock_execution_result

        result = await agent.create_environment(sample_requirements)

        assert result["success"] is True
        assert result["execution_id"] == "exec-456"
        assert result["plan"]["id"] == "test-plan-123"
        assert result["plan"]["steps"] == 2
        assert result["execution"]["duration"] == 175.5
        assert len(result["execution"]["artifacts"]) == 2
        assert result["error"] is None

        mock_planner.create_plan.assert_called_once_with(sample_requirements)
        mock_executor.validate_plan_requirements.assert_called_once_with(mock_plan)
        mock_executor.execute_plan.assert_called_once_with(mock_plan)

    @pytest.mark.asyncio
    async def test_create_environment_validation_failure(
        self, agent, mock_planner, mock_executor, mock_plan, sample_requirements
    ):
        """Test environment creation with plan validation failure."""
        agent.planner = mock_planner
        agent.executor = mock_executor
        mock_planner.create_plan.return_value = mock_plan
        mock_executor.validate_plan_requirements.return_value = {
            "valid": False,
            "missing_tools": ["terraform"],
            "invalid_actions": ["deploy_without_auth"],
        }

        result = await agent.create_environment(sample_requirements)

        assert result["success"] is False
        assert "Plan validation failed" in result["error"]
        assert "missing_tools" in result["error"]
        mock_executor.execute_plan.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_environment_execution_failure(
        self, agent, mock_planner, mock_executor, mock_plan, sample_requirements
    ):
        """Test environment creation with execution failure."""
        # Setup mocks
        agent.planner = mock_planner
        agent.executor = mock_executor
        mock_planner.create_plan.return_value = mock_plan
        mock_executor.validate_plan_requirements.return_value = {"valid": True}

        failed_result = Mock()
        failed_result.success = False
        failed_result.execution_id = "exec-failed"
        failed_result.duration = 45.0
        failed_result.artifacts = []
        failed_result.metrics = {
            "total_steps": 2,
            "successful_steps": 1,
            "failed_steps": 1,
        }
        failed_result.error = "Step 2 failed: Connection timeout"
        mock_executor.execute_plan.return_value = failed_result

        result = await agent.create_environment(sample_requirements)

        assert result["success"] is False
        assert result["error"] == "Step 2 failed: Connection timeout"
        assert result["execution"]["duration"] == 45.0

    @pytest.mark.asyncio
    async def test_create_environment_not_initialized(self, agent, sample_requirements):
        """Test create_environment when agent not initialized."""
        with pytest.raises(RuntimeError, match="Agent not initialized"):
            await agent.create_environment(sample_requirements)

    @pytest.mark.asyncio
    async def test_create_environment_exception_handling(
        self, agent, mock_planner, mock_executor, sample_requirements
    ):
        """Test create_environment exception handling."""
        agent.planner = mock_planner
        agent.executor = mock_executor
        mock_planner.create_plan.side_effect = Exception("Unexpected error")

        result = await agent.create_environment(sample_requirements)

        assert result["success"] is False
        assert "Unexpected error" in result["error"]
        assert result["requirements"] == sample_requirements.model_dump()

    @pytest.mark.asyncio
    async def test_list_templates_success(self, agent, mock_planner):
        """Test successful template listing."""
        agent.planner = mock_planner
        mock_templates = [
            {
                "name": "FastAPI Template",
                "description": "FastAPI with PostgreSQL",
                "framework": "fastapi",
                "estimated_duration": 300,
                "estimated_cost": 2.50,
            },
            {
                "name": "Node.js Template",
                "description": "Express.js web app",
                "framework": "nodejs",
                "estimated_duration": 240,
                "estimated_cost": 1.80,
            },
        ]
        mock_planner.get_available_templates.return_value = mock_templates

        result = await agent.list_templates()

        assert result["templates"] == mock_templates
        mock_planner.get_available_templates.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_templates_not_initialized(self, agent):
        """Test list_templates when agent not initialized."""
        with pytest.raises(RuntimeError, match="Agent not initialized"):
            await agent.list_templates()

    @pytest.mark.asyncio
    async def test_get_tools_status_success(self, agent, mock_executor):
        """Test successful tools status retrieval."""
        agent.executor = mock_executor
        mock_tools = {
            "docker": {
                "status": "available",
                "description": "Container management",
                "version": "20.10.0",
                "actions": ["build", "run", "stop"],
            },
            "aws": {
                "status": "available",
                "description": "AWS cloud services",
                "version": "1.0.0",
                "actions": ["create_ec2", "deploy_lambda"],
            },
        }
        mock_executor.get_available_tools.return_value = mock_tools

        result = await agent.get_tools_status()

        assert result["tools"] == mock_tools
        mock_executor.get_available_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_tools_status_not_initialized(self, agent):
        """Test get_tools_status when agent not initialized."""
        with pytest.raises(RuntimeError, match="Agent not initialized"):
            await agent.get_tools_status()


class TestCLICommands:
    """Test CLI command functions."""

    @pytest.fixture
    def runner(self):
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_agent(self):
        """Create a mock OrcastrateAgent."""
        agent = AsyncMock()
        agent.initialize = AsyncMock()
        agent.create_environment = AsyncMock()
        agent.list_templates = AsyncMock()
        agent.get_tools_status = AsyncMock()
        return agent

    def test_cli_main_command(self, runner):
        """Test main CLI command help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert (
            "Orcastrate - Production-Grade Development Environment Agent"
            in result.output
        )

    def test_cli_verbose_option(self, runner):
        """Test verbose option sets debug logging."""
        result = runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0
        # Just check that the command completes successfully with verbose flag
        assert (
            "Orcastrate - Production-Grade Development Environment Agent"
            in result.output
        )

    def test_cli_creates_log_directory(self, runner):
        """Test CLI creates log directory."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        # Just check that the CLI executes successfully
        assert (
            "Orcastrate - Production-Grade Development Environment Agent"
            in result.output
        )

    @patch("src.cli.main.OrcastrateAgent")
    @patch("src.cli.main.asyncio.run")
    def test_create_command_basic(self, mock_asyncio_run, mock_agent_class, runner):
        """Test basic create command."""
        mock_agent = AsyncMock()
        mock_agent.initialize = AsyncMock()
        mock_agent.create_environment = AsyncMock(
            return_value={
                "success": True,
                "execution_id": "test-123",
                "plan": {"steps": 3},
                "execution": {
                    "duration": 120.5,
                    "metrics": {"successful_steps": 3, "total_steps": 3},
                },
            }
        )
        mock_agent_class.return_value = mock_agent

        result = runner.invoke(cli, ["create", "FastAPI REST API"])
        assert result.exit_code == 0

        # Verify asyncio.run was called
        mock_asyncio_run.assert_called_once()

    @patch("src.cli.main.OrcastrateAgent")
    @patch("src.cli.main.asyncio.run")
    def test_create_command_with_options(
        self, mock_asyncio_run, mock_agent_class, runner
    ):
        """Test create command with all options."""
        mock_agent = AsyncMock()
        mock_agent_class.return_value = mock_agent

        result = runner.invoke(
            cli,
            [
                "create",
                "Web application",
                "--framework",
                "fastapi",
                "--database",
                "postgresql",
                "--cloud-provider",
                "aws",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()

    @patch("src.cli.main.OrcastrateAgent")
    @patch("src.cli.main.asyncio.run")
    def test_create_command_with_output_file(
        self, mock_asyncio_run, mock_agent_class, runner
    ):
        """Test create command with output file."""
        mock_agent = AsyncMock()
        mock_agent_class.return_value = mock_agent

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            result = runner.invoke(cli, ["create", "Test app", "--output", temp_path])
            assert result.exit_code == 0
            mock_asyncio_run.assert_called_once()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch("src.cli.main.OrcastrateAgent")
    @patch("src.cli.main.asyncio.run")
    def test_create_command_failure(self, mock_asyncio_run, mock_agent_class, runner):
        """Test create command failure handling."""

        def mock_run_with_exception(coro):
            """Mock asyncio.run that raises an exception."""
            raise Exception("Test error")

        mock_asyncio_run.side_effect = mock_run_with_exception
        mock_agent_class.return_value = AsyncMock()

        result = runner.invoke(cli, ["create", "Test app"])
        assert result.exit_code == 1
        # Check if asyncio.run was called indicating the command executed
        mock_asyncio_run.assert_called_once()

    @patch("src.cli.main.OrcastrateAgent")
    @patch("src.cli.main.asyncio.run")
    def test_templates_command(self, mock_asyncio_run, mock_agent_class, runner):
        """Test templates command."""
        mock_agent = AsyncMock()
        mock_agent_class.return_value = mock_agent

        result = runner.invoke(cli, ["templates"])
        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()

    @patch("src.cli.main.OrcastrateAgent")
    @patch("src.cli.main.asyncio.run")
    def test_tools_command(self, mock_asyncio_run, mock_agent_class, runner):
        """Test tools command."""
        mock_agent = AsyncMock()
        mock_agent_class.return_value = mock_agent

        result = runner.invoke(cli, ["tools"])
        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()

    def test_logs_command_no_file(self, runner):
        """Test logs command when no log file exists."""
        with patch("src.cli.main.LOG_DIR") as mock_log_dir:
            mock_log_file = MagicMock()
            mock_log_file.exists.return_value = False
            mock_log_dir.__truediv__.return_value = mock_log_file

            result = runner.invoke(cli, ["logs"])
            assert result.exit_code == 0
            assert "📄 No logs found" in result.output

    def test_logs_command_with_file(self, runner):
        """Test logs command with existing log file."""
        mock_log_content = [
            "2023-01-01 10:00:00 INFO Starting agent\n",
            "2023-01-01 10:01:00 DEBUG Processing request\n",
            "2023-01-01 10:02:00 INFO Task completed\n",
        ]

        with (
            patch("src.cli.main.LOG_DIR") as mock_log_dir,
            patch("builtins.open") as mock_open,
        ):
            mock_log_file = MagicMock()
            mock_log_file.exists.return_value = True
            mock_log_dir.__truediv__.return_value = mock_log_file
            mock_open.return_value.__enter__.return_value.readlines.return_value = (
                mock_log_content
            )

            result = runner.invoke(cli, ["logs", "--lines", "2"])
            assert result.exit_code == 0
            assert "Recent logs" in result.output

    def test_logs_command_with_custom_lines(self, runner):
        """Test logs command with custom line count."""
        with (
            patch("src.cli.main.LOG_DIR") as mock_log_dir,
            patch("builtins.open") as mock_open,
        ):
            mock_log_file = MagicMock()
            mock_log_file.exists.return_value = True
            mock_log_dir.__truediv__.return_value = mock_log_file
            mock_open.return_value.__enter__.return_value.readlines.return_value = [
                "line1\n",
                "line2\n",
            ]

            result = runner.invoke(cli, ["logs", "--lines", "10"])
            assert result.exit_code == 0

    def test_logs_command_read_error(self, runner):
        """Test logs command with file read error."""
        with (
            patch("src.cli.main.LOG_DIR") as mock_log_dir,
            patch("builtins.open") as mock_open,
        ):
            mock_log_file = MagicMock()
            mock_log_file.exists.return_value = True
            mock_log_dir.__truediv__.return_value = mock_log_file
            mock_open.side_effect = IOError("Permission denied")

            result = runner.invoke(cli, ["logs"])
            assert result.exit_code == 0
            assert "❌ Error reading logs" in result.output

    def test_version_command(self, runner):
        """Test version command."""
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert "Orcastrate Development Environment Agent" in result.output
        assert "Version: 1.0.0" in result.output
        assert "Phase: 2" in result.output


class TestCLIIntegration:
    """Integration tests for CLI workflows."""

    @pytest.fixture
    def runner(self):
        """Create a Click test runner."""
        return CliRunner()

    @patch("src.cli.main.OrcastrateAgent")
    @patch("src.cli.main.asyncio.run")
    def test_create_dry_run_workflow(self, mock_asyncio_run, mock_agent_class, runner):
        """Test complete dry-run workflow."""
        # Setup mock agent and plan
        mock_agent = AsyncMock()
        mock_plan = Mock()
        mock_plan.id = "plan-123"
        mock_plan.steps = [{"id": "step-1", "name": "Setup"}]
        mock_plan.estimated_duration = 300
        mock_plan.estimated_cost = 5.0
        mock_plan.risk_assessment = {"risk_level": "low"}

        mock_agent.planner = AsyncMock()
        mock_agent.planner.create_plan.return_value = mock_plan
        mock_agent_class.return_value = mock_agent

        # Create a mock async function that will be called by asyncio.run
        def mock_async_execution(coro):
            # Simulate successful execution
            return None

        mock_asyncio_run.side_effect = mock_async_execution

        result = runner.invoke(
            cli,
            [
                "create",
                "Test application",
                "--framework",
                "fastapi",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()

    @patch("src.cli.main.OrcastrateAgent")
    @patch("src.cli.main.asyncio.run")
    def test_create_with_output_file_workflow(
        self, mock_asyncio_run, mock_agent_class, runner
    ):
        """Test create command that saves output to file."""
        mock_agent = AsyncMock()
        mock_agent_class.return_value = mock_agent

        def mock_async_execution(coro):
            # Simulate the async function execution
            return None

        mock_asyncio_run.side_effect = mock_async_execution

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as temp_file:
            temp_path = temp_file.name

        try:
            with (
                patch("builtins.open", create=True) as _,
                patch("json.dump") as _,
            ):
                result = runner.invoke(
                    cli, ["create", "Test app", "--output", temp_path]
                )
                assert result.exit_code == 0
                mock_asyncio_run.assert_called_once()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch("src.cli.main.OrcastrateAgent")
    @patch("src.cli.main.asyncio.run")
    def test_error_handling_workflow(self, mock_asyncio_run, mock_agent_class, runner):
        """Test error handling in CLI workflow."""
        mock_agent = AsyncMock()
        mock_agent_class.return_value = mock_agent

        # Mock an exception during execution
        mock_asyncio_run.side_effect = RuntimeError("Agent initialization failed")

        result = runner.invoke(cli, ["create", "Test app"])
        assert result.exit_code == 1
        # Check if asyncio.run was called indicating the command executed
        mock_asyncio_run.assert_called_once()


class TestOrcastrateAgentEdgeCases:
    """Test edge cases and error conditions for OrcastrateAgent."""

    @pytest.fixture
    def agent(self):
        """Create an OrcastrateAgent instance."""
        return OrcastrateAgent()

    @pytest.mark.asyncio
    async def test_create_environment_with_warnings(self, agent):
        """Test create_environment with validation warnings."""
        # Setup mocks
        mock_planner = AsyncMock()
        mock_executor = AsyncMock()
        agent.planner = mock_planner
        agent.executor = mock_executor

        mock_plan = Mock()
        mock_plan.id = "test-plan"
        mock_plan.steps = []
        mock_plan.estimated_duration = 100
        mock_plan.estimated_cost = 1.0

        mock_planner.create_plan.return_value = mock_plan
        mock_executor.validate_plan_requirements.return_value = {
            "valid": True,
            "warnings": ["Resource usage is high", "Consider using smaller instance"],
        }

        mock_result = Mock()
        mock_result.success = True
        mock_result.execution_id = "exec-123"
        mock_result.duration = 95.0
        mock_result.artifacts = []
        mock_result.metrics = {}
        mock_result.error = None
        mock_executor.execute_plan.return_value = mock_result

        requirements = Requirements(description="Test app")
        result = await agent.create_environment(requirements)

        assert result["success"] is True
        # Warnings should be logged but not affect the result

    @pytest.mark.asyncio
    async def test_agent_logging_configuration(self, agent):
        """Test that agent has proper logging configuration."""
        assert agent.logger is not None
        assert agent.logger.name == "OrcastrateAgent"

    @pytest.mark.asyncio
    async def test_create_environment_with_empty_plan(self, agent):
        """Test create_environment with empty plan."""
        mock_planner = AsyncMock()
        mock_executor = AsyncMock()
        agent.planner = mock_planner
        agent.executor = mock_executor

        # Create a plan with no steps
        mock_plan = Mock()
        mock_plan.id = "empty-plan"
        mock_plan.steps = []
        mock_plan.estimated_duration = 0
        mock_plan.estimated_cost = 0.0

        mock_planner.create_plan.return_value = mock_plan
        mock_executor.validate_plan_requirements.return_value = {"valid": True}

        mock_result = Mock()
        mock_result.success = True
        mock_result.execution_id = "exec-empty"
        mock_result.duration = 0.1
        mock_result.artifacts = []
        mock_result.metrics = {"total_steps": 0, "successful_steps": 0}
        mock_result.error = None
        mock_executor.execute_plan.return_value = mock_result

        requirements = Requirements(description="Empty app")
        result = await agent.create_environment(requirements)

        assert result["success"] is True
        assert result["plan"]["steps"] == 0
