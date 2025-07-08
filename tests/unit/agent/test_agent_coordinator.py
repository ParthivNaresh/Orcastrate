"""
Tests for agent coordinator functionality.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.agent.base import AgentStatus, ExecutionResult
from src.agent.coordinator import AgentCoordinator, AgentError, ExecutionContext


class TestExecutionContext:
    """Test ExecutionContext class."""

    def test_execution_context_creation(self, sample_requirements):
        """Test basic execution context creation."""
        context = ExecutionContext(
            execution_id="exec_123",
            requirements=sample_requirements,
            start_time=datetime.utcnow(),
            status=AgentStatus.PLANNING,
        )

        assert context.execution_id == "exec_123"
        assert context.requirements == sample_requirements
        assert context.status == AgentStatus.PLANNING
        assert context.plan is None
        assert context.progress == 0.0
        assert context.current_step == "initializing"
        assert context.cancelled is False
        assert isinstance(context.metrics, dict)
        assert isinstance(context.artifacts, dict)
        assert isinstance(context.logs, list)


class TestAgentCoordinator:
    """Test AgentCoordinator class."""

    @pytest.fixture
    def coordinator_config(self):
        """Create coordinator configuration for testing."""
        return {
            "planner_type": "intelligent",
            "executor_type": "parallel",
            "security_enabled": True,
            "monitoring_enabled": True,
        }

    @pytest.fixture
    def coordinator(self, coordinator_config):
        """Create coordinator instance for testing."""
        return AgentCoordinator(coordinator_config)

    def test_coordinator_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert coordinator.config is not None
        assert coordinator.status == AgentStatus.IDLE
        assert len(coordinator._active_executions) == 0
        assert isinstance(coordinator._event_handlers, dict)

    @pytest.mark.asyncio
    async def test_initialize_coordinator(self, coordinator):
        """Test coordinator initialization process."""
        # Mock the component initialization
        with (
            patch.object(
                coordinator, "_initialize_components", new_callable=AsyncMock
            ) as mock_init,
            patch.object(coordinator, "_setup_event_handlers") as mock_setup,
        ):
            await coordinator.initialize()

            mock_init.assert_called_once()
            mock_setup.assert_called_once()
            assert coordinator.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_initialize_coordinator_failure(self, coordinator):
        """Test coordinator initialization failure."""
        with patch.object(
            coordinator, "_initialize_components", new_callable=AsyncMock
        ) as mock_init:
            mock_init.side_effect = Exception("Initialization failed")

            with pytest.raises(AgentError, match="Initialization failed"):
                await coordinator.initialize()

    @pytest.mark.asyncio
    async def test_create_environment_success(
        self,
        coordinator,
        sample_requirements,
        mock_planner,
        mock_executor,
        security_manager,
    ):
        """Test successful environment creation."""
        # Setup mocks
        coordinator.planner = mock_planner
        coordinator.executor = mock_executor
        coordinator.security_manager = security_manager

        with (
            patch.object(
                coordinator, "_validate_requirements", return_value=True
            ) as mock_validate,
            patch.object(
                coordinator, "_emit_event", new_callable=AsyncMock
            ) as mock_emit,
        ):
            result = await coordinator.create_environment(sample_requirements)

            assert result.success is True
            assert result.execution_id is not None
            assert len(coordinator._active_executions) == 0  # Should be cleaned up
            mock_validate.assert_called_once()
            assert mock_emit.call_count >= 4  # Should emit multiple events

    @pytest.mark.asyncio
    async def test_create_environment_validation_failure(
        self, coordinator, sample_requirements
    ):
        """Test environment creation with validation failure."""
        with patch.object(coordinator, "_validate_requirements", return_value=False):
            result = await coordinator.create_environment(sample_requirements)

            assert result.success is False
            assert "Requirements validation failed" in result.error

    @pytest.mark.asyncio
    async def test_create_environment_planning_failure(
        self, coordinator, sample_requirements
    ):
        """Test environment creation with planning failure."""
        mock_planner = Mock()
        mock_planner.create_plan = AsyncMock(side_effect=Exception("Planning failed"))
        coordinator.planner = mock_planner

        with patch.object(coordinator, "_validate_requirements", return_value=True):
            result = await coordinator.create_environment(sample_requirements)

            assert result.success is False
            assert "Planning failed" in result.error

    @pytest.mark.asyncio
    async def test_get_execution_status(self, coordinator, sample_requirements):
        """Test getting execution status."""
        # Create a mock execution context
        context = ExecutionContext(
            execution_id="test_exec",
            requirements=sample_requirements,
            start_time=datetime.utcnow(),
            status=AgentStatus.EXECUTING,
        )
        context.progress = 0.5
        context.current_step = "executing"

        # Add to active executions
        async with coordinator._execution_lock:
            coordinator._active_executions["test_exec"] = context

        status = await coordinator.get_execution_status("test_exec")

        assert status is not None
        assert status["execution_id"] == "test_exec"
        assert status["status"] == "executing"
        assert status["progress"] == 0.5
        assert status["current_step"] == "executing"
        assert "start_time" in status
        assert "elapsed_time" in status

    @pytest.mark.asyncio
    async def test_get_execution_status_not_found(self, coordinator):
        """Test getting status for non-existent execution."""
        status = await coordinator.get_execution_status("nonexistent")
        assert status is None

    @pytest.mark.asyncio
    async def test_cancel_execution(self, coordinator, sample_requirements):
        """Test canceling an execution."""
        # Create a mock execution context
        context = ExecutionContext(
            execution_id="test_exec",
            requirements=sample_requirements,
            start_time=datetime.utcnow(),
            status=AgentStatus.EXECUTING,
        )

        # Add to active executions
        async with coordinator._execution_lock:
            coordinator._active_executions["test_exec"] = context

        result = await coordinator.cancel_execution("test_exec")

        assert result is True
        assert context.cancelled is True

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_execution(self, coordinator):
        """Test canceling non-existent execution."""
        result = await coordinator.cancel_execution("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_workflow_execution_phases(
        self, coordinator, sample_requirements, sample_plan
    ):
        """Test the complete workflow execution phases."""
        # Mock all components
        mock_planner = Mock()
        mock_planner.create_plan = AsyncMock(return_value=sample_plan)
        coordinator.planner = mock_planner

        mock_security = Mock()
        mock_security.validate_plan = AsyncMock(return_value=True)
        coordinator.security_manager = mock_security

        mock_executor = Mock()
        mock_result = ExecutionResult(
            success=True, execution_id="test_exec", artifacts={"result": "success"}
        )
        mock_executor.execute_plan = AsyncMock(return_value=mock_result)
        coordinator.executor = mock_executor

        # Mock context
        context = ExecutionContext(
            execution_id="test_exec",
            requirements=sample_requirements,
            start_time=datetime.utcnow(),
            status=AgentStatus.PLANNING,
        )

        with (
            patch.object(
                coordinator, "_emit_event", new_callable=AsyncMock
            ) as mock_emit,
            patch.object(
                coordinator, "_setup_monitoring", new_callable=AsyncMock
            ) as mock_monitor,
        ):
            result = await coordinator._execute_workflow(context)

            # Verify all phases were called
            mock_planner.create_plan.assert_called_once()
            mock_security.validate_plan.assert_called_once()
            mock_executor.execute_plan.assert_called_once()
            mock_monitor.assert_called_once()

            # Verify events were emitted
            assert mock_emit.call_count >= 4

            # Verify result
            assert result.success is True

    @pytest.mark.asyncio
    async def test_workflow_security_validation_failure(
        self, coordinator, sample_requirements, sample_plan
    ):
        """Test workflow with security validation failure."""
        mock_planner = Mock()
        mock_planner.create_plan = AsyncMock(return_value=sample_plan)
        coordinator.planner = mock_planner

        mock_security = Mock()
        mock_security.validate_plan = AsyncMock(return_value=False)
        coordinator.security_manager = mock_security

        context = ExecutionContext(
            execution_id="test_exec",
            requirements=sample_requirements,
            start_time=datetime.utcnow(),
            status=AgentStatus.PLANNING,
        )

        with patch.object(coordinator, "_emit_event", new_callable=AsyncMock):
            with pytest.raises(AgentError, match="Security validation failed"):
                await coordinator._execute_workflow(context)

    def test_event_subscription(self, coordinator):
        """Test event subscription mechanism."""
        mock_handler = AsyncMock()

        coordinator.subscribe_to_event("test_event", mock_handler)

        assert "test_event" in coordinator._event_handlers
        assert mock_handler in coordinator._event_handlers["test_event"]

    @pytest.mark.asyncio
    async def test_event_emission(self, coordinator, sample_requirements):
        """Test event emission to subscribers."""
        mock_handler1 = AsyncMock()
        mock_handler2 = AsyncMock()
        failing_handler = AsyncMock(side_effect=Exception("Handler failed"))

        coordinator.subscribe_to_event("test_event", mock_handler1)
        coordinator.subscribe_to_event("test_event", mock_handler2)
        coordinator.subscribe_to_event("test_event", failing_handler)

        context = ExecutionContext(
            execution_id="test_exec",
            requirements=sample_requirements,
            start_time=datetime.utcnow(),
            status=AgentStatus.EXECUTING,
        )

        await coordinator._emit_event("test_event", context, extra_data="test")

        # Verify handlers were called
        mock_handler1.assert_called_once()
        mock_handler2.assert_called_once()
        failing_handler.assert_called_once()

        # Verify event data structure
        call_args = mock_handler1.call_args[0][0]
        assert call_args["type"] == "test_event"
        assert call_args["execution_id"] == "test_exec"
        assert call_args["extra_data"] == "test"
        assert "timestamp" in call_args
        assert call_args["context"] is context

    @pytest.mark.asyncio
    async def test_validate_requirements_basic(self, coordinator, sample_requirements):
        """Test basic requirements validation."""
        result = await coordinator._validate_requirements(sample_requirements)
        assert result is True

        # Test with empty description
        invalid_req = sample_requirements.model_copy()
        invalid_req.description = ""
        result = await coordinator._validate_requirements(invalid_req)
        assert result is False

    @pytest.mark.asyncio
    async def test_concurrent_executions(self, coordinator, sample_requirements):
        """Test handling multiple concurrent executions."""
        # Create multiple execution contexts
        contexts = []
        for i in range(3):
            context = ExecutionContext(
                execution_id=f"exec_{i}",
                requirements=sample_requirements,
                start_time=datetime.utcnow(),
                status=AgentStatus.EXECUTING,
            )
            contexts.append(context)

        # Add all to active executions
        async with coordinator._execution_lock:
            for context in contexts:
                coordinator._active_executions[context.execution_id] = context

        # Verify all are tracked
        assert len(coordinator._active_executions) == 3

        # Test getting status for each
        for i in range(3):
            status = await coordinator.get_execution_status(f"exec_{i}")
            assert status is not None
            assert status["execution_id"] == f"exec_{i}"

    @pytest.mark.asyncio
    async def test_execution_cleanup_on_failure(self, coordinator, sample_requirements):
        """Test that executions are cleaned up even on failure."""
        with (
            patch.object(coordinator, "_validate_requirements", return_value=True),
            patch.object(
                coordinator,
                "_execute_workflow",
                side_effect=Exception("Workflow failed"),
            ),
        ):
            result = await coordinator.create_environment(sample_requirements)

            assert result.success is False
            assert "Workflow failed" in result.error
            assert len(coordinator._active_executions) == 0  # Should be cleaned up
