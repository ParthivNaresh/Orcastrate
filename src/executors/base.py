"""
Base executor interface and implementations.

This module defines the execution system that takes plans and executes them
using various execution strategies and engines.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from ..agent.base import ExecutionResult, Plan


class ExecutionStrategy(Enum):
    """Available execution strategies."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    ADAPTIVE = "adaptive"


class ExecutionStatus(Enum):
    """Execution status for individual steps."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


class StepExecution(BaseModel):
    """Execution state for a single step."""

    step_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    rollback_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionContext(BaseModel):
    """Context for tracking execution state."""

    execution_id: str
    plan_id: str
    strategy: ExecutionStrategy
    start_time: datetime
    end_time: Optional[datetime] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    steps: Dict[str, StepExecution] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    rollback_plan: Optional[List[str]] = None
    cancellation_requested: bool = False
    progress: Dict[str, Any] = Field(default_factory=dict)


class ExecutorConfig(BaseModel):
    """Configuration for an executor."""

    strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    max_concurrent_steps: int = 10
    step_timeout: int = 3600  # seconds
    retry_policy: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_retries": 3,
            "backoff_factor": 2.0,
            "max_delay": 300,
        }
    )
    enable_rollback: bool = True
    checkpoint_interval: int = 10  # steps
    resource_limits: Dict[str, Any] = Field(default_factory=dict)
    monitoring_enabled: bool = True
    continue_on_failure: bool = False

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with fallback (for compatibility)."""
        return getattr(self, key, default)


class Executor(ABC):
    """
    Base executor interface for plan execution.

    This abstract base class defines the contract that all executors must implement.
    It provides a consistent interface for plan execution, monitoring, and rollback.
    """

    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._active_executions: Dict[str, ExecutionContext] = {}
        self._execution_lock = asyncio.Lock()
        self._thread_pool = ThreadPoolExecutor(max_workers=config.max_concurrent_steps)
        self._tool_registry: Dict[str, Any] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}

    async def initialize(self) -> None:
        """Initialize the executor and its components."""
        self.logger.info("Initializing executor")

        try:
            await self._initialize_tool_registry()
            await self._setup_monitoring()

            self.logger.info("Executor initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize executor: {e}")
            raise ExecutorError(f"Initialization failed: {e}")

    async def execute_plan(self, plan: Plan) -> ExecutionResult:
        """
        Execute a plan and return the result.

        Args:
            plan: The plan to execute

        Returns:
            ExecutionResult: Result of the execution
        """
        execution_id = f"exec_{plan.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.utcnow()

        try:
            self.logger.info(f"Starting plan execution: {execution_id}")

            # Create execution context
            context = ExecutionContext(
                execution_id=execution_id,
                plan_id=plan.id,
                strategy=self.config.strategy,
                start_time=start_time,
                status=ExecutionStatus.RUNNING,
            )

            # Initialize step executions
            for step in plan.steps:
                context.steps[step["id"]] = StepExecution(step_id=step["id"])

            # Register active execution
            async with self._execution_lock:
                self._active_executions[execution_id] = context

            # Execute the plan
            result = await self._execute_plan_with_strategy(plan, context)

            # Clean up
            async with self._execution_lock:
                self._active_executions.pop(execution_id, None)

            return result

        except Exception as e:
            self.logger.error(f"Plan execution failed: {e}")

            # Clean up on failure
            async with self._execution_lock:
                self._active_executions.pop(execution_id, None)

            return ExecutionResult(
                success=False,
                execution_id=execution_id,
                error=str(e),
                duration=(datetime.utcnow() - start_time).total_seconds(),
            )

    async def get_execution_status(
        self, execution_id: str
    ) -> Optional[ExecutionContext]:
        """Get the status of an active execution."""
        async with self._execution_lock:
            return self._active_executions.get(execution_id)

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution."""
        async with self._execution_lock:
            context = self._active_executions.get(execution_id)
            if not context:
                return False

            context.cancellation_requested = True
            self.logger.info(f"Cancellation requested for execution: {execution_id}")
            return True

    async def rollback_execution(self, execution_id: str) -> ExecutionResult:
        """Rollback a completed or failed execution."""
        start_time = datetime.utcnow()

        try:
            self.logger.info(f"Starting rollback for execution: {execution_id}")

            # TODO: Implement rollback logic
            # This would involve reversing the operations performed

            return ExecutionResult(
                success=True,
                execution_id=f"rollback_{execution_id}",
                duration=(datetime.utcnow() - start_time).total_seconds(),
            )

        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return ExecutionResult(
                success=False,
                execution_id=f"rollback_{execution_id}",
                error=str(e),
                duration=(datetime.utcnow() - start_time).total_seconds(),
            )

    @abstractmethod
    async def _execute_plan_with_strategy(
        self, plan: Plan, context: ExecutionContext
    ) -> ExecutionResult:
        """Execute plan using the configured strategy."""

    async def _execute_step(
        self, step: Dict[str, Any], context: ExecutionContext
    ) -> StepExecution:
        """Execute a single step."""
        step_id = step["id"]
        step_execution = context.steps[step_id]

        try:
            self.logger.info(f"Executing step: {step_id}")

            step_execution.status = ExecutionStatus.RUNNING
            step_execution.start_time = datetime.utcnow()

            # Get the tool for this step
            tool_name = step.get("tool")
            if tool_name and tool_name not in self._tool_registry:
                raise ExecutorError(f"Tool not available: {tool_name}")

            if tool_name:
                tool = self._tool_registry[tool_name]
                # Execute the step with retry logic
                result = await self._execute_step_with_retry(tool, step, context)
            else:
                # No tool specified, return empty result
                result = {
                    "status": "completed",
                    "message": "No tool execution required",
                }

            step_execution.status = ExecutionStatus.COMPLETED
            step_execution.end_time = datetime.utcnow()
            step_execution.result = result

            self.logger.info(f"Step completed: {step_id}")

        except Exception as e:
            self.logger.error(f"Step failed: {step_id} - {e}")

            step_execution.status = ExecutionStatus.FAILED
            step_execution.end_time = datetime.utcnow()
            step_execution.error = str(e)

        return step_execution

    async def _execute_step_with_retry(
        self, tool: Any, step: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute a step with retry logic."""
        max_retries = self.config.retry_policy["max_retries"]
        backoff_factor = self.config.retry_policy["backoff_factor"]
        max_delay = self.config.retry_policy["max_delay"]

        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Check for cancellation
                if context.cancellation_requested:
                    raise ExecutorError("Execution cancelled")

                # Execute the tool action
                result = await asyncio.wait_for(
                    tool.execute(step["action"], step["parameters"]),
                    timeout=self.config.step_timeout,
                )

                if result.success:
                    return result.output  # type: ignore[no-any-return]
                else:
                    raise ExecutorError(f"Tool execution failed: {result.error}")

            except asyncio.TimeoutError:
                last_error = f"Step timed out after {self.config.step_timeout} seconds"
                if attempt < max_retries:
                    delay = min(backoff_factor**attempt, max_delay)
                    await asyncio.sleep(delay)
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    delay = min(backoff_factor**attempt, max_delay)
                    self.logger.warning(
                        f"Step attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        raise ExecutorError(
            f"Step failed after {max_retries + 1} attempts: {last_error}"
        )

    async def _check_dependencies(
        self, step: Dict[str, Any], context: ExecutionContext
    ) -> bool:
        """Check if step dependencies are satisfied."""
        dependencies = step.get("dependencies", [])

        for dep_id in dependencies:
            if dep_id not in context.steps:
                return False

            dep_execution = context.steps[dep_id]
            if dep_execution.status != ExecutionStatus.COMPLETED:
                return False

        return True

    async def _create_dependency_graph(self, plan: Plan) -> Dict[str, Set[str]]:
        """Create a dependency graph from the plan."""
        graph = {}

        for step in plan.steps:
            step_id = step["id"]
            dependencies = set(step.get("dependencies", []))
            graph[step_id] = dependencies

        return graph

    async def _topological_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        """Perform topological sort on dependency graph."""
        # Kahn's algorithm
        # In our graph, graph[node] contains dependencies OF that node
        in_degree = {node: len(graph[node]) for node in graph}

        # Add any missing nodes (dependencies that aren't in the main graph)
        for node in graph:
            for dep in graph[node]:
                if dep not in in_degree:
                    in_degree[dep] = 0

        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Find all nodes that depend on this node
            for dependent in graph:
                if node in graph[dependent]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if len(result) != len(in_degree):
            raise ExecutorError("Circular dependency detected")

        return result

    async def _initialize_tool_registry(self) -> None:
        """Initialize the tool registry."""
        # TODO: Initialize actual tools
        self._tool_registry = {}

    async def _setup_monitoring(self) -> None:
        """Set up execution monitoring."""
        if self.config.monitoring_enabled:
            # TODO: Set up monitoring
            pass

    def subscribe_to_event(self, event_type: str, handler: Callable) -> None:
        """Subscribe to execution events."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to all subscribers."""
        if event_type not in self._event_handlers:
            return

        for handler in self._event_handlers[event_type]:
            try:
                await handler(data)
            except Exception as e:
                self.logger.error(f"Event handler failed: {e}")


class SequentialExecutor(Executor):
    """Sequential execution strategy."""

    async def _execute_plan_with_strategy(
        self, plan: Plan, context: ExecutionContext
    ) -> ExecutionResult:
        """Execute steps sequentially."""
        start_time = datetime.utcnow()

        try:
            # Sort steps by dependencies
            dependency_graph = await self._create_dependency_graph(plan)
            execution_order = await self._topological_sort(dependency_graph)

            # Execute steps in order
            for step_id in execution_order:
                if context.cancellation_requested:
                    break

                step = next(s for s in plan.steps if s["id"] == step_id)
                await self._execute_step(step, context)

                # Check if step failed
                if context.steps[step_id].status == ExecutionStatus.FAILED:
                    break

            # Determine overall success
            success = all(
                step_exec.status == ExecutionStatus.COMPLETED
                for step_exec in context.steps.values()
            )

            return ExecutionResult(
                success=success,
                execution_id=context.execution_id,
                artifacts=context.artifacts,
                logs=context.logs,
                metrics=context.metrics,
                duration=(datetime.utcnow() - start_time).total_seconds(),
            )

        except Exception as e:
            self.logger.error(f"Sequential execution failed: {e}")
            return ExecutionResult(
                success=False,
                execution_id=context.execution_id,
                error=str(e),
                duration=(datetime.utcnow() - start_time).total_seconds(),
            )


class ParallelExecutor(Executor):
    """Parallel execution strategy."""

    async def _execute_plan_with_strategy(
        self, plan: Plan, context: ExecutionContext
    ) -> ExecutionResult:
        """Execute steps in parallel where possible."""
        start_time = datetime.utcnow()

        try:
            # Create dependency graph
            await self._create_dependency_graph(plan)

            # Execute steps in parallel batches
            executed_steps: Set[str] = set()

            while len(executed_steps) < len(plan.steps):
                if context.cancellation_requested:
                    break

                # Find steps that can be executed (dependencies satisfied)
                ready_steps = []
                for step in plan.steps:
                    step_id = step["id"]
                    if step_id in executed_steps:
                        continue

                    if await self._check_dependencies(step, context):
                        ready_steps.append(step)

                if not ready_steps:
                    break  # No more steps can be executed

                # Execute ready steps in parallel
                tasks = []
                for step in ready_steps:
                    task = asyncio.create_task(self._execute_step(step, context))
                    tasks.append((step["id"], task))

                # Wait for all tasks to complete
                for step_id, task in tasks:
                    await task
                    executed_steps.add(step_id)

            # Determine overall success
            success = all(
                step_exec.status == ExecutionStatus.COMPLETED
                for step_exec in context.steps.values()
            )

            return ExecutionResult(
                success=success,
                execution_id=context.execution_id,
                artifacts=context.artifacts,
                logs=context.logs,
                metrics=context.metrics,
                duration=(datetime.utcnow() - start_time).total_seconds(),
            )

        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            return ExecutionResult(
                success=False,
                execution_id=context.execution_id,
                error=str(e),
                duration=(datetime.utcnow() - start_time).total_seconds(),
            )


class ExecutorError(Exception):
    """Base exception for executor-related errors."""


class ExecutionTimeoutError(ExecutorError):
    """Exception raised when execution times out."""


class ExecutionCancellationError(ExecutorError):
    """Exception raised when execution is cancelled."""
