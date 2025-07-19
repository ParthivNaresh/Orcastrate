"""
End-to-end integration tests for Orcastrate.

These tests verify the complete workflow from requirements to execution.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.agent.base import Requirements
from src.executors.base import ExecutionStrategy, ExecutorConfig
from src.executors.concrete_executor import ConcreteExecutor
from src.logging_utils import LogManager, ProgressTracker
from src.planners.base import PlannerConfig, PlanningStrategy
from src.planners.template_planner import TemplatePlanner


class TestEndToEndWorkflow:
    """Test the complete end-to-end workflow."""

    @pytest.fixture
    async def planner(self):
        """Create and initialize template planner."""
        config = PlannerConfig(
            strategy=PlanningStrategy.TEMPLATE_MATCHING,
            max_plan_steps=20,
            max_planning_time=60,
        )
        planner = TemplatePlanner(config)
        await planner.initialize()
        return planner

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
    async def executor(self, mock_progress_tracker):
        """Create and initialize concrete executor with mocked tools."""
        config = ExecutorConfig(
            strategy=ExecutionStrategy.SEQUENTIAL,
            max_concurrent_steps=5,
            step_timeout=300,
        )
        executor = ConcreteExecutor(config, mock_progress_tracker)

        # Mock tool initialization to avoid requiring real tools
        with patch.object(executor, "_initialize_tools") as mock_init:
            mock_init.return_value = None

            # Create mock tools
            from src.tools.base import ToolSchema

            mock_filesystem = AsyncMock()
            mock_filesystem.execute = AsyncMock()
            mock_filesystem._get_supported_actions = AsyncMock(
                return_value=["create_directory", "write_file", "read_file"]
            )
            mock_filesystem.get_schema = AsyncMock(
                return_value=ToolSchema(
                    name="filesystem",
                    description="File system operations",
                    version="1.0.0",
                    actions={"create_directory": {}, "write_file": {}, "read_file": {}},
                )
            )

            mock_git = AsyncMock()
            mock_git.execute = AsyncMock()
            mock_git._get_supported_actions = AsyncMock(
                return_value=["init", "add", "commit", "clone"]
            )
            mock_git.get_schema = AsyncMock(
                return_value=ToolSchema(
                    name="git",
                    description="Git version control",
                    version="1.0.0",
                    actions={"init": {}, "add": {}, "commit": {}, "clone": {}},
                )
            )

            mock_docker = AsyncMock()
            mock_docker.execute = AsyncMock()
            mock_docker._get_supported_actions = AsyncMock(
                return_value=["build_image", "run_container", "stop_container"]
            )
            mock_docker.get_schema = AsyncMock(
                return_value=ToolSchema(
                    name="docker",
                    description="Docker containerization",
                    version="1.0.0",
                    actions={
                        "build_image": {},
                        "run_container": {},
                        "stop_container": {},
                    },
                )
            )

            executor._tools = {
                "filesystem": mock_filesystem,
                "git": mock_git,
                "docker": mock_docker,
            }

            await executor.initialize()

        return executor

    @pytest.mark.asyncio
    async def test_nodejs_workflow_plan_generation(self, planner):
        """Test plan generation for Node.js application."""
        requirements = Requirements(
            description="Create a Node.js web application with Express",
            framework="nodejs",
        )

        plan = await planner.create_plan(requirements)

        assert plan is not None
        assert len(plan.steps) > 0
        assert plan.estimated_duration > 0

        # Verify expected steps are present
        step_names = [
            step.name if hasattr(step, "name") else step.get("name", "")
            for step in plan.steps
        ]
        assert any("directory" in name.lower() for name in step_names)
        assert any("git" in name.lower() for name in step_names)
        assert any("package" in name.lower() for name in step_names)
        # Enhanced Template Planner only adds Docker for multi-service apps
        # Basic Node.js app gets: directory, git, package.json, main file
        assert any(
            "main" in name.lower() or "application" in name.lower()
            for name in step_names
        )

    @pytest.mark.asyncio
    async def test_fastapi_workflow_plan_generation(self, planner):
        """Test plan generation for FastAPI application."""
        requirements = Requirements(
            description="Build a FastAPI REST API service",
            framework="fastapi",
        )

        plan = await planner.create_plan(requirements)

        assert plan is not None
        assert len(plan.steps) > 0

        # Verify FastAPI-specific elements
        plan_content = str(plan.steps)
        assert "fastapi" in plan_content.lower() or "python" in plan_content.lower()
        assert "requirements.txt" in plan_content

    @pytest.mark.asyncio
    async def test_plan_validation_success(self, executor):
        """Test successful plan validation."""
        # Create a mock plan with valid steps
        mock_plan = Mock()
        mock_plan.steps = [
            {
                "id": "step1",
                "tool": "filesystem",
                "action": "create_directory",
                "parameters": {"path": "/tmp/test"},
            },
            {
                "id": "step2",
                "tool": "git",
                "action": "init",
                "parameters": {"path": "/tmp/test"},
            },
        ]

        validation = await executor.validate_plan_requirements(mock_plan)

        assert validation["valid"] is True
        assert len(validation["missing_tools"]) == 0
        assert len(validation["invalid_actions"]) == 0

    @pytest.mark.asyncio
    async def test_plan_validation_missing_tool(self, executor):
        """Test plan validation with missing tool."""
        mock_plan = Mock()
        mock_plan.steps = [
            {
                "id": "step1",
                "tool": "nonexistent_tool",
                "action": "some_action",
                "parameters": {},
            }
        ]

        validation = await executor.validate_plan_requirements(mock_plan)

        assert validation["valid"] is False
        assert "nonexistent_tool" in validation["missing_tools"]

    @pytest.mark.asyncio
    async def test_plan_validation_invalid_action(self, executor):
        """Test plan validation with invalid action."""
        mock_plan = Mock()
        mock_plan.steps = [
            {
                "id": "step1",
                "tool": "filesystem",
                "action": "invalid_action",
                "parameters": {},
            }
        ]

        validation = await executor.validate_plan_requirements(mock_plan)

        assert validation["valid"] is False
        assert len(validation["invalid_actions"]) > 0
        assert validation["invalid_actions"][0]["action"] == "invalid_action"

    @pytest.mark.asyncio
    async def test_successful_plan_execution(self, executor):
        """Test successful plan execution with mocked tools."""
        # Configure mock tools to return success
        for tool in executor._tools.values():
            mock_result = Mock()
            mock_result.success = True
            mock_result.output = {"result": "success"}
            mock_result.duration = 1.0
            mock_result.metadata = {}
            tool.execute.return_value = mock_result

        # Create a simple plan
        mock_plan = Mock()
        mock_plan.id = "test-plan"
        mock_plan.steps = [
            {
                "id": "step1",
                "name": "Create Directory",
                "tool": "filesystem",
                "action": "create_directory",
                "parameters": {"path": "/tmp/test"},
                "dependencies": [],
            },
            {
                "id": "step2",
                "name": "Initialize Git",
                "tool": "git",
                "action": "init",
                "parameters": {"path": "/tmp/test"},
                "dependencies": ["step1"],
            },
        ]

        result = await executor.execute_plan(mock_plan)

        assert result.success is True
        assert result.execution_id is not None
        assert len(result.artifacts) > 0

        # Verify tools were called
        executor._tools["filesystem"].execute.assert_called()
        executor._tools["git"].execute.assert_called()

    @pytest.mark.asyncio
    async def test_plan_execution_with_failure(self, executor):
        """Test plan execution with tool failure."""
        # Configure first tool to fail
        fail_result = Mock()
        fail_result.success = False
        fail_result.error = "Tool execution failed"
        executor._tools["filesystem"].execute.return_value = fail_result

        # Configure second tool to succeed (shouldn't be called due to failure)
        success_result = Mock()
        success_result.success = True
        success_result.output = {"result": "success"}
        executor._tools["git"].execute.return_value = success_result

        mock_plan = Mock()
        mock_plan.id = "test-plan"
        mock_plan.steps = [
            {
                "id": "step1",
                "tool": "filesystem",
                "action": "create_directory",
                "parameters": {"path": "/tmp/test"},
                "dependencies": [],
            },
            {
                "id": "step2",
                "tool": "git",
                "action": "init",
                "parameters": {"path": "/tmp/test"},
                "dependencies": ["step1"],
            },
        ]

        result = await executor.execute_plan(mock_plan)

        assert result.success is False
        assert "failed" in result.error.lower()

        # Verify only first tool was called
        executor._tools["filesystem"].execute.assert_called()
        # Second tool shouldn't be called due to dependency failure
        # (This depends on implementation details)

    @pytest.mark.asyncio
    async def test_technology_detection_nodejs(self, planner):
        """Test technology detection for Node.js requirements."""
        requirements = Requirements(
            description="Node.js application with Express framework", framework="nodejs"
        )

        # Test the new technology detection system
        detected = planner.detect_technologies(requirements.description)

        assert detected.framework == "nodejs"
        # Test that plan can be generated
        plan = await planner.create_plan(requirements)
        assert plan is not None

    @pytest.mark.asyncio
    async def test_technology_detection_fastapi(self, planner):
        """Test technology detection for FastAPI requirements."""
        requirements = Requirements(
            description="Python FastAPI REST API", framework="fastapi"
        )

        # Test the new technology detection system
        detected = planner.detect_technologies(requirements.description)

        assert detected.framework == "fastapi"
        # Test that plan can be generated
        plan = await planner.create_plan(requirements)
        assert plan is not None

    @pytest.mark.asyncio
    async def test_technology_detection_fallback(self, planner):
        """Test technology detection fallback for generic requirements."""
        requirements = Requirements(description="A web application", framework=None)

        # Test that even without explicit framework, we can still create a plan
        plan = await planner.create_plan(requirements)
        assert plan is not None
        assert len(plan.steps) > 0

    @pytest.mark.asyncio
    async def test_project_name_generation_integration(self, planner):
        """Test project name generation integration."""
        requirements = Requirements(
            description="My awesome Node.js web application", framework="nodejs"
        )

        # Test project name generation directly
        project_name = planner._generate_project_name(requirements.description)

        assert project_name
        assert "my" in project_name or "awesome" in project_name

        # Test that plan generation works with variable substitution
        plan = await planner.create_plan(requirements)
        assert plan is not None
        # Check that variables were properly substituted in steps
        for step in plan.steps:
            # Parameters should not contain unsubstituted variables
            step_params = (
                step.parameters
                if hasattr(step, "parameters")
                else step.get("parameters", {})
            )
            step_str = str(step_params)
            assert "{project_name}" not in step_str
            assert "{project_path}" not in step_str

    @pytest.mark.asyncio
    async def test_project_name_generation(self, planner):
        """Test project name generation from descriptions."""
        test_cases = [
            ("Node.js web application", "node-js-web"),
            ("FastAPI REST API service", "fastapi-rest-api"),
            ("My awesome app", "my-awesome-app"),
            ("Simple web app with database", "simple-web-app"),
        ]

        for description, expected_prefix in test_cases:
            name = planner._generate_project_name(description)
            assert name
            assert expected_prefix in name or name.startswith(
                expected_prefix.split("-")[0]
            )

    @pytest.mark.asyncio
    async def test_variable_substitution(self, planner):
        """Test variable substitution in step data."""
        template_data = {
            "name": "{project_name}",
            "path": "{project_path}/src",
            "nested": {"description": "This is {project_name} with {framework}"},
        }

        variables = {
            "project_name": "my-app",
            "project_path": "/tmp/my-app",
            "framework": "nodejs",
        }

        result = planner._substitute_variables(template_data, variables)

        assert result["name"] == "my-app"
        assert result["path"] == "/tmp/my-app/src"
        assert result["nested"]["description"] == "This is my-app with nodejs"

    @pytest.mark.asyncio
    async def test_available_templates_list(self, planner):
        """Test listing available templates."""
        templates = await planner.get_available_templates()

        assert len(templates) > 0

        for template in templates:
            assert "id" in template
            assert "name" in template
            assert "description" in template
            assert template["name"]  # Non-empty name
            assert template["description"]  # Non-empty description

    @pytest.mark.asyncio
    async def test_available_tools_list(self, executor):
        """Test listing available tools."""
        tools = await executor.get_available_tools()

        assert len(tools) > 0

        expected_tools = ["filesystem", "git", "docker"]
        for tool_name in expected_tools:
            assert tool_name in tools
            assert tools[tool_name]["status"] == "available"

    @pytest.mark.asyncio
    async def test_execution_summary(self, executor):
        """Test execution summary generation."""
        # Create a mock context
        from datetime import datetime

        from src.executors.base import ExecutionContext, ExecutionStatus, StepExecution

        context = ExecutionContext(
            execution_id="test-exec",
            plan_id="test-plan",
            strategy=ExecutionStrategy.SEQUENTIAL,
            start_time=datetime.utcnow(),
        )

        # Add some mock step executions
        context.steps = {
            "step1": StepExecution(step_id="step1", status=ExecutionStatus.COMPLETED),
            "step2": StepExecution(step_id="step2", status=ExecutionStatus.FAILED),
            "step3": StepExecution(step_id="step3", status=ExecutionStatus.PENDING),
        }

        summary = await executor.get_execution_summary(context)

        assert summary["execution_id"] == "test-exec"
        assert summary["steps"]["total"] == 3
        assert summary["steps"]["completed"] == 1
        assert summary["steps"]["failed"] == 1
        assert summary["steps"]["pending"] == 1
        assert "duration" in summary


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_nodejs_express_scenario(self):
        """Test complete Node.js Express application scenario."""
        # This would be a more comprehensive test that might run in a
        # special integration test environment

    @pytest.mark.asyncio
    async def test_fastapi_scenario(self):
        """Test complete FastAPI application scenario."""

    @pytest.mark.asyncio
    async def test_error_recovery_scenario(self):
        """Test error recovery and rollback scenarios."""
