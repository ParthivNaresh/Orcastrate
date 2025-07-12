"""
Tests for executor base classes and functionality.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.agent.base import ExecutionResult
from src.executors.base import (
    ExecutionCancellationError,
    ExecutionContext,
    ExecutionStatus,
    ExecutionStrategy,
    ExecutionTimeoutError,
    Executor,
    ExecutorConfig,
    ExecutorError,
    ParallelExecutor,
    SequentialExecutor,
    StepExecution,
)


class TestExecutorConfig:
    """Test ExecutorConfig model."""

    def test_executor_config_defaults(self):
        """Test executor configuration with defaults."""
        config = ExecutorConfig()

        assert config.strategy == ExecutionStrategy.ADAPTIVE
        assert config.max_concurrent_steps == 10
        assert config.step_timeout == 3600
        assert config.retry_policy == {
            "max_retries": 3,
            "backoff_factor": 2.0,
            "max_delay": 300,
        }
        assert config.enable_rollback is True
        assert config.checkpoint_interval == 10
        assert config.resource_limits == {}
        assert config.monitoring_enabled is True

    def test_executor_config_custom(self):
        """Test executor configuration with custom values."""
        config = ExecutorConfig(
            strategy=ExecutionStrategy.SEQUENTIAL,
            max_concurrent_steps=5,
            step_timeout=1800,
            retry_policy={"max_retries": 5, "backoff_factor": 1.5, "max_delay": 600},
            enable_rollback=False,
            checkpoint_interval=5,
            resource_limits={"memory": "2Gi", "cpu": "1000m"},
            monitoring_enabled=False,
        )

        assert config.strategy == ExecutionStrategy.SEQUENTIAL
        assert config.max_concurrent_steps == 5
        assert config.step_timeout == 1800
        assert config.retry_policy["max_retries"] == 5
        assert config.enable_rollback is False
        assert config.checkpoint_interval == 5
        assert config.resource_limits == {"memory": "2Gi", "cpu": "1000m"}
        assert config.monitoring_enabled is False


class TestStepExecution:
    """Test StepExecution model."""

    def test_step_execution_creation(self):
        """Test step execution creation."""
        execution = StepExecution(
            step_id="step_123",
            status=ExecutionStatus.RUNNING,
            start_time=datetime.utcnow(),
            result={"output": "test"},
            metadata={"region": "us-east-1"},
        )

        assert execution.step_id == "step_123"
        assert execution.status == ExecutionStatus.RUNNING
        assert execution.start_time is not None
        assert execution.end_time is None
        assert execution.result == {"output": "test"}
        assert execution.error is None
        assert execution.retry_count == 0
        assert execution.rollback_info is None
        assert execution.metadata == {"region": "us-east-1"}

    def test_step_execution_defaults(self):
        """Test step execution with defaults."""
        execution = StepExecution(step_id="step_456")

        assert execution.step_id == "step_456"
        assert execution.status == ExecutionStatus.PENDING
        assert execution.start_time is None
        assert execution.end_time is None
        assert execution.result is None
        assert execution.error is None
        assert execution.retry_count == 0
        assert execution.rollback_info is None
        assert execution.metadata == {}


class TestExecutionContext:
    """Test ExecutionContext model."""

    def test_execution_context_creation(self):
        """Test execution context creation."""
        context = ExecutionContext(
            execution_id="exec_123",
            plan_id="plan_456",
            strategy=ExecutionStrategy.PARALLEL,
            start_time=datetime.utcnow(),
            steps={"step1": StepExecution(step_id="step1")},
            artifacts={"vpc_id": "vpc-123"},
            logs=["Step 1 started"],
            metrics={"duration": 300.0},
        )

        assert context.execution_id == "exec_123"
        assert context.plan_id == "plan_456"
        assert context.strategy == ExecutionStrategy.PARALLEL
        assert context.start_time is not None
        assert context.end_time is None
        assert context.status == ExecutionStatus.PENDING
        assert "step1" in context.steps
        assert context.artifacts == {"vpc_id": "vpc-123"}
        assert context.logs == ["Step 1 started"]
        assert context.metrics == {"duration": 300.0}
        assert context.rollback_plan is None
        assert context.cancellation_requested is False


class TestExecutor:
    """Test Executor base class."""

    @pytest.fixture
    def concrete_executor(self, executor_config):
        """Create a concrete executor for testing."""

        class ConcreteExecutor(Executor):
            def __init__(self, config, should_fail=False):
                super().__init__(config)
                self.should_fail = should_fail
                self.tools_initialized = False
                self.monitoring_setup = False
                self.executions_performed = []

            async def _execute_plan_with_strategy(self, plan, context):
                self.executions_performed.append((plan, context))

                if self.should_fail:
                    return ExecutionResult(
                        success=False,
                        execution_id=context.execution_id,
                        error="Mock execution failure",
                    )

                return ExecutionResult(
                    success=True,
                    execution_id=context.execution_id,
                    artifacts={"mock": "result"},
                    logs=["Execution completed"],
                    metrics={"duration": 600.0},
                )

            async def _initialize_tool_registry(self):
                self.tools_initialized = True

            async def _setup_monitoring(self):
                self.monitoring_setup = True

        return ConcreteExecutor(executor_config)

    def test_executor_initialization(self, concrete_executor):
        """Test executor initialization."""
        assert concrete_executor.config is not None
        assert len(concrete_executor._active_executions) == 0
        assert concrete_executor._thread_pool is not None
        assert concrete_executor._tool_registry == {}
        assert isinstance(concrete_executor._event_handlers, dict)

    @pytest.mark.asyncio
    async def test_executor_initialize(self, concrete_executor):
        """Test executor initialization process."""
        await concrete_executor.initialize()

        assert concrete_executor.tools_initialized is True
        assert concrete_executor.monitoring_setup is True

    @pytest.mark.asyncio
    async def test_executor_initialize_failure(self, executor_config):
        """Test executor initialization failure."""

        class FailingExecutor(Executor):
            async def _execute_plan_with_strategy(self, plan, context):
                pass

            async def _initialize_tool_registry(self):
                raise Exception("Tool registry initialization failed")

            async def _setup_monitoring(self):
                pass

        executor = FailingExecutor(executor_config)

        with pytest.raises(ExecutorError, match="Initialization failed"):
            await executor.initialize()

    @pytest.mark.asyncio
    async def test_execute_plan_success(self, concrete_executor, sample_plan):
        """Test successful plan execution."""
        await concrete_executor.initialize()

        result = await concrete_executor.execute_plan(sample_plan)

        assert result.success is True
        assert result.execution_id is not None
        assert result.artifacts == {"mock": "result"}
        assert result.logs == ["Execution completed"]
        assert result.metrics == {"duration": 600.0}
        assert len(concrete_executor.executions_performed) == 1
        assert len(concrete_executor._active_executions) == 0  # Should be cleaned up

    @pytest.mark.asyncio
    async def test_execute_plan_failure(
        self, concrete_executor, executor_config, sample_plan
    ):
        """Test failed plan execution."""
        failing_executor = concrete_executor.__class__(
            executor_config, should_fail=True
        )
        await failing_executor.initialize()

        result = await failing_executor.execute_plan(sample_plan)

        assert result.success is False
        assert "Mock execution failure" in result.error
        assert len(failing_executor._active_executions) == 0  # Should be cleaned up

    @pytest.mark.asyncio
    async def test_execute_plan_exception(self, executor_config, sample_plan):
        """Test plan execution with exception."""

        class ExceptionExecutor(Executor):
            async def _execute_plan_with_strategy(self, plan, context):
                raise Exception("Unexpected error")

            async def _initialize_tool_registry(self):
                pass

            async def _setup_monitoring(self):
                pass

        executor = ExceptionExecutor(executor_config)
        await executor.initialize()

        result = await executor.execute_plan(sample_plan)

        assert result.success is False
        assert "Unexpected error" in result.error
        assert len(executor._active_executions) == 0  # Should be cleaned up

    @pytest.mark.asyncio
    async def test_get_execution_status(self, concrete_executor, sample_plan):
        """Test getting execution status."""
        await concrete_executor.initialize()

        # Mock an active execution
        context = ExecutionContext(
            execution_id="test_exec",
            plan_id=sample_plan.id,
            strategy=ExecutionStrategy.SEQUENTIAL,
            start_time=datetime.utcnow(),
        )
        context.steps["step1"] = StepExecution(
            step_id="step1", status=ExecutionStatus.RUNNING
        )

        async with concrete_executor._execution_lock:
            concrete_executor._active_executions["test_exec"] = context

        status = await concrete_executor.get_execution_status("test_exec")

        assert status is not None
        assert status.execution_id == "test_exec"
        assert status.plan_id == sample_plan.id

    @pytest.mark.asyncio
    async def test_get_execution_status_not_found(self, concrete_executor):
        """Test getting status for non-existent execution."""
        await concrete_executor.initialize()

        status = await concrete_executor.get_execution_status("nonexistent")
        assert status is None

    @pytest.mark.asyncio
    async def test_cancel_execution(self, concrete_executor, sample_plan):
        """Test canceling an execution."""
        await concrete_executor.initialize()

        # Mock an active execution
        context = ExecutionContext(
            execution_id="test_exec",
            plan_id=sample_plan.id,
            strategy=ExecutionStrategy.SEQUENTIAL,
            start_time=datetime.utcnow(),
        )

        async with concrete_executor._execution_lock:
            concrete_executor._active_executions["test_exec"] = context

        result = await concrete_executor.cancel_execution("test_exec")

        assert result is True
        assert context.cancellation_requested is True

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_execution(self, concrete_executor):
        """Test canceling non-existent execution."""
        await concrete_executor.initialize()

        result = await concrete_executor.cancel_execution("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_rollback_execution(self, concrete_executor):
        """Test rollback execution."""
        await concrete_executor.initialize()

        result = await concrete_executor.rollback_execution("test_exec")

        assert result.success is True
        assert result.execution_id == "rollback_test_exec"
        assert result.duration is not None

    @pytest.mark.asyncio
    async def test_execute_step(self, concrete_executor, sample_plan):
        """Test executing a single step."""
        await concrete_executor.initialize()

        # Mock tool in registry
        mock_tool = Mock()
        mock_tool_result = Mock()
        mock_tool_result.success = True
        mock_tool_result.output = {"resource_id": "res-123"}
        mock_tool.execute = AsyncMock(return_value=mock_tool_result)
        concrete_executor._tool_registry["aws_tool"] = mock_tool

        # Create execution context
        context = ExecutionContext(
            execution_id="test_exec",
            plan_id=sample_plan.id,
            strategy=ExecutionStrategy.SEQUENTIAL,
            start_time=datetime.utcnow(),
        )
        context.steps["step1"] = StepExecution(step_id="step1")

        step = sample_plan.steps[0]  # First step
        step_execution = await concrete_executor._execute_step(step, context)

        assert step_execution.status == ExecutionStatus.COMPLETED
        assert step_execution.start_time is not None
        assert step_execution.end_time is not None
        assert step_execution.result == {"resource_id": "res-123"}
        assert step_execution.error is None
        mock_tool.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_step_tool_not_found(self, concrete_executor, sample_plan):
        """Test executing step with missing tool."""
        await concrete_executor.initialize()

        context = ExecutionContext(
            execution_id="test_exec",
            plan_id=sample_plan.id,
            strategy=ExecutionStrategy.SEQUENTIAL,
            start_time=datetime.utcnow(),
        )
        context.steps["step1"] = StepExecution(step_id="step1")

        step = sample_plan.steps[0]  # First step (uses aws_tool)
        step_execution = await concrete_executor._execute_step(step, context)

        assert step_execution.status == ExecutionStatus.FAILED
        assert "Tool not available" in step_execution.error

    @pytest.mark.asyncio
    async def test_execute_step_with_retry(self, concrete_executor):
        """Test step execution with retry logic."""
        await concrete_executor.initialize()

        # Mock tool that fails first time, succeeds second time
        mock_tool = Mock()
        call_count = 0

        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            mock_result = Mock()
            mock_result.success = True
            mock_result.output = {"success": True}
            return mock_result

        mock_tool.execute = mock_execute
        context = ExecutionContext(
            execution_id="test_exec",
            plan_id="plan_123",
            strategy=ExecutionStrategy.SEQUENTIAL,
            start_time=datetime.utcnow(),
        )

        result = await concrete_executor._execute_step_with_retry(
            mock_tool, {"action": "test_action", "parameters": {}}, context
        )

        assert result == {"success": True}
        assert call_count == 2  # Should have retried once

    @pytest.mark.asyncio
    async def test_execute_step_timeout(self, concrete_executor):
        """Test step execution timeout."""
        await concrete_executor.initialize()

        # Set very short timeout
        concrete_executor.config.step_timeout = 0.1

        # Mock tool that takes too long
        mock_tool = Mock()

        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(1)  # Longer than timeout
            mock_result = Mock()
            mock_result.success = True
            mock_result.output = {"success": True}
            return mock_result

        mock_tool.execute = slow_execute
        context = ExecutionContext(
            execution_id="test_exec",
            plan_id="plan_123",
            strategy=ExecutionStrategy.SEQUENTIAL,
            start_time=datetime.utcnow(),
        )

        with pytest.raises(ExecutorError, match="timed out"):
            await concrete_executor._execute_step_with_retry(
                mock_tool, {"action": "test_action", "parameters": {}}, context
            )

    @pytest.mark.asyncio
    async def test_execute_step_cancellation(self, concrete_executor):
        """Test step execution with cancellation."""
        await concrete_executor.initialize()

        mock_tool = Mock()
        context = ExecutionContext(
            execution_id="test_exec",
            plan_id="plan_123",
            strategy=ExecutionStrategy.SEQUENTIAL,
            start_time=datetime.utcnow(),
        )
        context.cancellation_requested = True

        with pytest.raises(ExecutorError, match="cancelled"):
            await concrete_executor._execute_step_with_retry(
                mock_tool, {"action": "test_action", "parameters": {}}, context
            )

    @pytest.mark.asyncio
    async def test_check_dependencies(self, concrete_executor):
        """Test dependency checking."""
        context = ExecutionContext(
            execution_id="test_exec",
            plan_id="plan_123",
            strategy=ExecutionStrategy.SEQUENTIAL,
            start_time=datetime.utcnow(),
        )

        # Add completed dependency
        context.steps["dep1"] = StepExecution(
            step_id="dep1", status=ExecutionStatus.COMPLETED
        )
        context.steps["dep2"] = StepExecution(
            step_id="dep2", status=ExecutionStatus.RUNNING
        )

        # Test step with satisfied dependencies
        step_satisfied = {"id": "step1", "dependencies": ["dep1"]}
        assert (
            await concrete_executor._check_dependencies(step_satisfied, context) is True
        )

        # Test step with unsatisfied dependencies
        step_unsatisfied = {"id": "step2", "dependencies": ["dep2"]}
        assert (
            await concrete_executor._check_dependencies(step_unsatisfied, context)
            is False
        )

        # Test step with missing dependency
        step_missing = {"id": "step3", "dependencies": ["missing"]}
        assert (
            await concrete_executor._check_dependencies(step_missing, context) is False
        )

    @pytest.mark.asyncio
    async def test_create_dependency_graph(self, concrete_executor, sample_plan):
        """Test dependency graph creation."""
        graph = await concrete_executor._create_dependency_graph(sample_plan)

        expected = {"step1": set(), "step2": {"step1"}, "step3": {"step1", "step2"}}

        assert graph == expected

    @pytest.mark.asyncio
    async def test_topological_sort(self, concrete_executor):
        """Test topological sort."""
        graph = {"step1": set(), "step2": {"step1"}, "step3": {"step1", "step2"}}

        sorted_steps = await concrete_executor._topological_sort(graph)

        # step1 should come first, step3 should come last
        print(sorted_steps)
        assert sorted_steps.index("step1") < sorted_steps.index("step2")
        assert sorted_steps.index("step1") < sorted_steps.index("step3")
        assert sorted_steps.index("step2") < sorted_steps.index("step3")

    @pytest.mark.asyncio
    async def test_topological_sort_circular(self, concrete_executor):
        """Test topological sort with circular dependency."""
        graph = {"step1": {"step2"}, "step2": {"step1"}}  # Circular dependency

        with pytest.raises(ExecutorError, match="Circular dependency detected"):
            await concrete_executor._topological_sort(graph)

    def test_event_subscription(self, concrete_executor):
        """Test event subscription."""
        mock_handler = Mock()

        concrete_executor.subscribe_to_event("test_event", mock_handler)

        assert "test_event" in concrete_executor._event_handlers
        assert mock_handler in concrete_executor._event_handlers["test_event"]

    @pytest.mark.asyncio
    async def test_event_emission(self, concrete_executor):
        """Test event emission."""
        mock_handler = AsyncMock()
        failing_handler = AsyncMock(side_effect=Exception("Handler failed"))

        concrete_executor.subscribe_to_event("test_event", mock_handler)
        concrete_executor.subscribe_to_event("test_event", failing_handler)

        await concrete_executor._emit_event("test_event", {"data": "test"})

        mock_handler.assert_called_once_with({"data": "test"})
        failing_handler.assert_called_once_with({"data": "test"})


class TestSequentialExecutor:
    """Test SequentialExecutor implementation."""

    @pytest.fixture
    def sequential_executor(self, executor_config):
        """Create sequential executor for testing."""
        return SequentialExecutor(executor_config)

    @pytest.mark.asyncio
    async def test_sequential_execution_success(self, sequential_executor, sample_plan):
        """Test successful sequential execution."""
        await sequential_executor.initialize()

        # Mock step execution
        async def mock_execute_step(step, context):
            step_id = step["id"]
            step_execution = context.steps[step_id]
            step_execution.status = ExecutionStatus.COMPLETED
            return step_execution

        with patch.object(
            sequential_executor, "_execute_step", side_effect=mock_execute_step
        ) as mock_execute:
            context = ExecutionContext(
                execution_id="test_exec",
                plan_id=sample_plan.id,
                strategy=ExecutionStrategy.SEQUENTIAL,
                start_time=datetime.utcnow(),
            )

            # Initialize step executions
            for step in sample_plan.steps:
                context.steps[step["id"]] = StepExecution(step_id=step["id"])

            result = await sequential_executor._execute_plan_with_strategy(
                sample_plan, context
            )

            assert result.success is True
            assert mock_execute.call_count == len(sample_plan.steps)

    @pytest.mark.asyncio
    async def test_sequential_execution_step_failure(
        self, sequential_executor, sample_plan
    ):
        """Test sequential execution with step failure."""
        await sequential_executor.initialize()

        call_count = 0

        async def mock_execute_step(step, context):
            nonlocal call_count
            call_count += 1

            step_id = step["id"]
            step_execution = context.steps[step_id]

            if call_count == 2:  # Second step fails
                step_execution.status = ExecutionStatus.FAILED
                step_execution.error = "Step failed"
            else:
                step_execution.status = ExecutionStatus.COMPLETED

            return step_execution

        with patch.object(
            sequential_executor, "_execute_step", side_effect=mock_execute_step
        ):
            context = ExecutionContext(
                execution_id="test_exec",
                plan_id=sample_plan.id,
                strategy=ExecutionStrategy.SEQUENTIAL,
                start_time=datetime.utcnow(),
            )

            # Initialize step executions
            for step in sample_plan.steps:
                context.steps[step["id"]] = StepExecution(step_id=step["id"])

            result = await sequential_executor._execute_plan_with_strategy(
                sample_plan, context
            )

            assert result.success is False
            # Should stop after second step fails
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_sequential_execution_cancellation(
        self, sequential_executor, sample_plan
    ):
        """Test sequential execution with cancellation."""
        await sequential_executor.initialize()

        call_count = 0

        async def mock_execute_step(step, context):
            nonlocal call_count
            call_count += 1

            if call_count == 2:  # Cancel after first step
                context.cancellation_requested = True

            return StepExecution(step_id=step["id"], status=ExecutionStatus.COMPLETED)

        with patch.object(
            sequential_executor, "_execute_step", side_effect=mock_execute_step
        ):
            context = ExecutionContext(
                execution_id="test_exec",
                plan_id=sample_plan.id,
                strategy=ExecutionStrategy.SEQUENTIAL,
                start_time=datetime.utcnow(),
            )

            # Initialize step executions
            for step in sample_plan.steps:
                context.steps[step["id"]] = StepExecution(step_id=step["id"])

            await sequential_executor._execute_plan_with_strategy(sample_plan, context)

            # Should stop after cancellation
            assert call_count == 2


class TestParallelExecutor:
    """Test ParallelExecutor implementation."""

    @pytest.fixture
    def parallel_executor(self, executor_config):
        """Create parallel executor for testing."""
        return ParallelExecutor(executor_config)

    @pytest.mark.asyncio
    async def test_parallel_execution_success(self, parallel_executor, sample_plan):
        """Test successful parallel execution."""
        await parallel_executor.initialize()

        execution_order = []

        async def mock_execute_step(step, context):
            execution_order.append(step["id"])
            step_id = step["id"]
            step_execution = context.steps[step_id]
            step_execution.status = ExecutionStatus.COMPLETED
            return step_execution

        with (
            patch.object(
                parallel_executor, "_execute_step", side_effect=mock_execute_step
            ),
            patch.object(parallel_executor, "_check_dependencies", return_value=True),
        ):
            context = ExecutionContext(
                execution_id="test_exec",
                plan_id=sample_plan.id,
                strategy=ExecutionStrategy.PARALLEL,
                start_time=datetime.utcnow(),
            )

            # Initialize step executions
            for step in sample_plan.steps:
                context.steps[step["id"]] = StepExecution(step_id=step["id"])

            result = await parallel_executor._execute_plan_with_strategy(
                sample_plan, context
            )

            assert result.success is True
            assert len(execution_order) == len(sample_plan.steps)

    @pytest.mark.asyncio
    async def test_parallel_execution_with_dependencies(
        self, parallel_executor, sample_plan
    ):
        """Test parallel execution respecting dependencies."""
        await parallel_executor.initialize()

        execution_order = []
        completed_steps = set()

        async def mock_execute_step(step, context):
            execution_order.append(step["id"])
            completed_steps.add(step["id"])
            step_id = step["id"]
            step_execution = context.steps[step_id]
            step_execution.status = ExecutionStatus.COMPLETED
            return step_execution

        async def mock_check_dependencies(step, context):
            dependencies = step.get("dependencies", [])
            return all(dep in completed_steps for dep in dependencies)

        with (
            patch.object(
                parallel_executor, "_execute_step", side_effect=mock_execute_step
            ),
            patch.object(
                parallel_executor,
                "_check_dependencies",
                side_effect=mock_check_dependencies,
            ),
        ):
            context = ExecutionContext(
                execution_id="test_exec",
                plan_id=sample_plan.id,
                strategy=ExecutionStrategy.PARALLEL,
                start_time=datetime.utcnow(),
            )

            # Initialize step executions
            for step in sample_plan.steps:
                context.steps[step["id"]] = StepExecution(step_id=step["id"])

            result = await parallel_executor._execute_plan_with_strategy(
                sample_plan, context
            )

            assert result.success is True
            # step1 should execute before step2 and step3
            assert execution_order.index("step1") < execution_order.index("step2")
            assert execution_order.index("step1") < execution_order.index("step3")


class TestExecutorExceptions:
    """Test executor-related exceptions."""

    def test_executor_error(self):
        """Test ExecutorError exception."""
        error = ExecutorError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_execution_timeout_error(self):
        """Test ExecutionTimeoutError exception."""
        error = ExecutionTimeoutError("Execution timed out")
        assert str(error) == "Execution timed out"
        assert isinstance(error, ExecutorError)

    def test_execution_cancellation_error(self):
        """Test ExecutionCancellationError exception."""
        error = ExecutionCancellationError("Execution cancelled")
        assert str(error) == "Execution cancelled"
        assert isinstance(error, ExecutorError)
