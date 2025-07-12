"""
Agent Coordinator - Orchestrates planning and execution across the system.

This module implements the orchestration layer that coordinates between
different agents, tools, and services to fulfill environment creation requests.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..executors.base import Executor
from ..planners.base import Planner
from ..security.manager import SecurityManager
from .base import AgentError, AgentStatus, ExecutionResult, Plan, Requirements


class AgentCoordinator:
    """
    Central orchestrator for agent operations.

    This class coordinates the entire workflow from requirements gathering
    through planning, execution, and monitoring. It implements the Mediator
    pattern to reduce coupling between components.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.status = AgentStatus.IDLE

        # Core components
        self.planner: Optional[Planner] = None
        self.executor: Optional[Executor] = None
        self.security_manager: Optional[SecurityManager] = None

        # Active executions tracking
        self._active_executions: Dict[str, "ExecutionContext"] = {}
        self._execution_lock = asyncio.Lock()

        # Event system for component communication
        self._event_handlers: Dict[str, List[Callable]] = {}

    async def initialize(self) -> None:
        """Initialize the coordinator and its components."""
        try:
            self.logger.info("Initializing Agent Coordinator")

            # Initialize components based on configuration
            await self._initialize_components()

            # Set up event handlers
            self._setup_event_handlers()

            self.status = AgentStatus.IDLE
            self.logger.info("Agent Coordinator initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Agent Coordinator: {e}")
            raise AgentError(f"Initialization failed: {e}")

    async def create_environment(self, requirements: Requirements) -> ExecutionResult:
        """
        Create a development environment based on requirements.

        Args:
            requirements: Environment requirements specification

        Returns:
            ExecutionResult: Result of the environment creation
        """
        execution_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        try:
            self.logger.info(f"Starting environment creation: {execution_id}")

            # Validate requirements
            if not await self._validate_requirements(requirements):
                raise AgentError("Requirements validation failed")

            # Create execution context
            context = ExecutionContext(
                execution_id=execution_id,
                requirements=requirements,
                start_time=start_time,
                status=AgentStatus.PLANNING,
            )

            async with self._execution_lock:
                self._active_executions[execution_id] = context

            # Execute the workflow
            result = await self._execute_workflow(context)

            # Clean up
            async with self._execution_lock:
                self._active_executions.pop(execution_id, None)

            return result

        except Exception as e:
            self.logger.error(f"Environment creation failed: {e}")

            # Clean up on failure
            async with self._execution_lock:
                self._active_executions.pop(execution_id, None)

            return ExecutionResult(
                success=False,
                execution_id=execution_id,
                error=str(e),
                duration=(datetime.utcnow() - start_time).total_seconds(),
            )

    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an active execution."""
        async with self._execution_lock:
            context = self._active_executions.get(execution_id)
            if not context:
                return None

            return {
                "execution_id": context.execution_id,
                "status": context.status.value,
                "progress": context.progress,
                "current_step": context.current_step,
                "start_time": context.start_time.isoformat(),
                "elapsed_time": (
                    datetime.utcnow() - context.start_time
                ).total_seconds(),
                "metrics": context.metrics,
            }

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution."""
        async with self._execution_lock:
            context = self._active_executions.get(execution_id)
            if not context:
                return False

            context.cancelled = True
            self.logger.info(f"Execution {execution_id} marked for cancellation")
            return True

    def subscribe_to_event(self, event_type: str, handler: Callable) -> None:
        """Subscribe to system events."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    async def _execute_workflow(self, context: "ExecutionContext") -> ExecutionResult:
        """Execute the complete workflow for environment creation."""
        try:
            # Phase 1: Planning
            await self._emit_event("planning_started", context)
            plan = await self._create_plan(context)
            context.plan = plan
            context.status = AgentStatus.EXECUTING

            # Phase 2: Security validation
            await self._emit_event("security_validation_started", context)
            if not await self._validate_security(context):
                raise AgentError("Security validation failed")

            # Phase 3: Execution
            await self._emit_event("execution_started", context)
            result = await self._execute_plan(context)

            # Phase 4: Monitoring setup
            await self._emit_event("monitoring_setup_started", context)
            await self._setup_monitoring(context)

            await self._emit_event("execution_completed", context)
            return result

        except Exception as e:
            await self._emit_event("execution_failed", context, error=str(e))
            raise

    async def _create_plan(self, context: "ExecutionContext") -> Plan:
        """Create an execution plan."""
        if not self.planner:
            raise AgentError("No planner configured")

        context.current_step = "planning"
        context.progress = 0.1

        plan = await self.planner.create_plan(context.requirements)
        self.logger.info(f"Created plan with {len(plan.steps)} steps")

        return plan

    async def _validate_security(self, context: "ExecutionContext") -> bool:
        """Validate security requirements."""
        if not self.security_manager:
            self.logger.warning("No security manager configured")
            return True

        context.current_step = "security_validation"
        context.progress = 0.2

        if context.plan is None:
            raise AgentError("No plan available for security validation")
        return await self.security_manager.validate_plan(context.plan)

    async def _execute_plan(self, context: "ExecutionContext") -> ExecutionResult:
        """Execute the plan."""
        if not self.executor:
            raise AgentError("No executor configured")

        context.current_step = "execution"
        context.progress = 0.3

        if context.plan is None:
            raise AgentError("No plan available for execution")
        result = await self.executor.execute_plan(context.plan)

        # Update progress based on execution
        context.progress = 0.9

        return result

    async def _setup_monitoring(self, context: "ExecutionContext") -> None:
        """Set up monitoring for the created environment."""
        context.current_step = "monitoring_setup"
        context.progress = 1.0

        # TODO: Implement monitoring setup
        self.logger.info("Monitoring setup completed")

    async def _validate_requirements(self, requirements: Requirements) -> bool:
        """Validate requirements before processing."""
        if not requirements.description:
            return False

        # Additional validation logic
        # TODO: Implement comprehensive validation

        return True

    async def _initialize_components(self) -> None:
        """Initialize system components."""
        # TODO: Initialize based on configuration
        # This would typically involve dependency injection

    def _setup_event_handlers(self) -> None:
        """Set up default event handlers."""
        # TODO: Set up default event handlers for logging, metrics, etc.

    async def _emit_event(
        self, event_type: str, context: "ExecutionContext", **kwargs
    ) -> None:
        """Emit an event to all subscribers."""
        if event_type not in self._event_handlers:
            return

        event_data = {
            "type": event_type,
            "execution_id": context.execution_id,
            "timestamp": datetime.utcnow().isoformat(),
            "context": context,
            **kwargs,
        }

        for handler in self._event_handlers[event_type]:
            try:
                await handler(event_data)
            except Exception as e:
                self.logger.error(f"Event handler failed: {e}")


class ExecutionContext:
    """Context for tracking execution state."""

    def __init__(
        self,
        execution_id: str,
        requirements: Requirements,
        start_time: datetime,
        status: AgentStatus,
    ):
        self.execution_id = execution_id
        self.requirements = requirements
        self.start_time = start_time
        self.status = status
        self.plan: Optional[Plan] = None
        self.progress: float = 0.0
        self.current_step: str = "initializing"
        self.metrics: Dict[str, Any] = {}
        self.cancelled: bool = False
        self.artifacts: Dict[str, Any] = {}
        self.logs: List[str] = []
