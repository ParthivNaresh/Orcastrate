"""
Base tool interface and implementations.

This module defines the tool abstraction layer that allows the system
to interact with various external services and resources in a consistent manner.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolStatus(Enum):
    """Tool operation status."""

    IDLE = "idle"
    VALIDATING = "validating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ToolResult(BaseModel):
    """Result of a tool operation."""

    success: bool
    tool_name: str
    action: str
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    rollback_info: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ValidationResult(BaseModel):
    """Result of parameter validation."""

    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    normalized_params: Dict[str, Any] = Field(default_factory=dict)


class ToolConfig(BaseModel):
    """Configuration for a tool."""

    name: str
    version: str
    enabled: bool = True
    timeout: int = 300  # seconds
    retry_count: int = 3
    retry_delay: int = 5  # seconds
    environment: Dict[str, Any] = Field(default_factory=dict)
    credentials: Dict[str, Any] = Field(default_factory=dict)
    resource_limits: Dict[str, Any] = Field(default_factory=dict)


class ToolSchema(BaseModel):
    """Schema describing tool capabilities."""

    name: str
    description: str
    version: str
    actions: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required_permissions: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    cost_model: Dict[str, Any] = Field(default_factory=dict)


class CostEstimate(BaseModel):
    """Cost estimation for a tool operation."""

    estimated_cost: float
    currency: str = "USD"
    cost_breakdown: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.8  # 0-1 scale
    factors: List[str] = Field(default_factory=list)


class Tool(ABC):
    """
    Base tool interface for all system tools.

    This abstract base class defines the contract that all tools must implement.
    It provides a consistent interface for validation, execution, and rollback
    operations across different tool types.
    """

    def __init__(self, config: ToolConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}:{config.name}")
        self.status = ToolStatus.IDLE
        self._client: Optional[Any] = None
        self._validator: Optional[Any] = None
        self._last_operation: Optional[Dict[str, Any]] = None

    async def initialize(self) -> None:
        """Initialize the tool (setup clients, validate config, etc.)."""
        try:
            self._validator = await self._create_validator()
            self._client = await self._create_client()
            await self._validate_configuration()

        except Exception as e:
            self.logger.error(f"Failed to initialize tool {self.config.name}: {e}")
            raise ToolError(f"Initialization failed: {e}")

    async def execute(self, action: str, params: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool action with the given parameters.

        Args:
            action: The action to execute
            params: Parameters for the action

        Returns:
            ToolResult: Result of the operation
        """
        start_time = datetime.utcnow()
        operation_id = f"{self.config.name}_{action}_{int(start_time.timestamp())}"

        try:
            self.status = ToolStatus.VALIDATING

            # Validate parameters
            validation = await self.validate(action, params)
            if not validation.valid:
                self.logger.error(
                    "Tool parameter validation failed",
                    extra={
                        "tool_name": self.config.name,
                        "operation_id": operation_id,
                        "action": action,
                        "operation": "tool_execution",
                        "phase": "validation_error",
                        "validation_errors": validation.errors,
                        "validation_warnings": validation.warnings,
                    },
                )
                raise ToolError(f"Validation failed: {validation.errors}")

            # Use normalized parameters
            normalized_params = validation.normalized_params or params

            # Execute with retry logic
            self.status = ToolStatus.EXECUTING
            result = await self._execute_with_retry(action, normalized_params)

            # Store operation for potential rollback
            self._last_operation = {
                "action": action,
                "params": normalized_params,
                "result": result,
                "timestamp": start_time,
            }

            self.status = ToolStatus.COMPLETED

            duration = (datetime.utcnow() - start_time).total_seconds()

            return ToolResult(
                success=True,
                tool_name=self.config.name,
                action=action,
                output=result,
                duration=duration,
                rollback_info=self._get_rollback_info(
                    action, normalized_params, result
                ),
            )

        except Exception as e:
            self.status = ToolStatus.FAILED
            duration = (datetime.utcnow() - start_time).total_seconds()

            self.logger.error(
                "Tool action failed",
                extra={
                    "tool_name": self.config.name,
                    "operation_id": operation_id,
                    "action": action,
                    "operation": "tool_execution",
                    "phase": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "duration_seconds": duration,
                    "tool_status": self.status.value,
                },
            )

            return ToolResult(
                success=False,
                tool_name=self.config.name,
                action=action,
                error=str(e),
                duration=duration,
            )

    async def validate(self, action: str, params: Dict[str, Any]) -> ValidationResult:
        """
        Validate action parameters.

        Args:
            action: The action to validate
            params: Parameters to validate

        Returns:
            ValidationResult: Validation result
        """
        if not self._validator:
            # Default validation - check if action exists
            if action not in await self._get_supported_actions():
                return ValidationResult(
                    valid=False, errors=[f"Unsupported action: {action}"]
                )

            return ValidationResult(valid=True, normalized_params=params)

        if hasattr(self._validator, "validate"):
            result = self._validator.validate(action, params)
            # Handle both sync and async validators
            if hasattr(result, "__await__"):
                result = await result
        else:
            result = ValidationResult(valid=True, normalized_params=params)
        return result  # type: ignore[no-any-return]

    async def rollback(self, execution_id: str) -> ToolResult:
        """
        Rollback a previous operation.

        Args:
            execution_id: ID of the execution to rollback

        Returns:
            ToolResult: Result of the rollback operation
        """
        start_time = datetime.utcnow()

        try:
            self.logger.info(f"Rolling back execution: {execution_id}")
            self.status = ToolStatus.EXECUTING

            result = await self._execute_rollback(execution_id)

            self.status = ToolStatus.ROLLED_BACK
            duration = (datetime.utcnow() - start_time).total_seconds()

            return ToolResult(
                success=True,
                tool_name=self.config.name,
                action="rollback",
                output=result,
                duration=duration,
            )

        except Exception as e:
            self.status = ToolStatus.FAILED
            duration = (datetime.utcnow() - start_time).total_seconds()

            self.logger.error(f"Rollback failed: {e}")

            return ToolResult(
                success=False,
                tool_name=self.config.name,
                action="rollback",
                error=str(e),
                duration=duration,
            )

    @abstractmethod
    async def get_schema(self) -> ToolSchema:
        """Return the tool's schema describing its capabilities."""

    @abstractmethod
    async def estimate_cost(self, action: str, params: Dict[str, Any]) -> CostEstimate:
        """Estimate the cost of executing an action."""

    @abstractmethod
    async def _create_client(self) -> Any:
        """Create and configure the underlying client."""

    @abstractmethod
    async def _create_validator(self) -> Any:
        """Create and configure the parameter validator."""

    @abstractmethod
    async def _execute_action(
        self, action: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the actual action (implemented by subclasses)."""

    @abstractmethod
    async def _execute_rollback(self, execution_id: str) -> Dict[str, Any]:
        """Execute rollback operation (implemented by subclasses)."""

    @abstractmethod
    async def _get_supported_actions(self) -> List[str]:
        """Get list of supported actions."""

    async def _validate_configuration(self) -> None:
        """Validate tool configuration."""
        if not self.config.name:
            raise ToolError("Tool name is required")

        # Additional validation can be added here

    async def _execute_with_retry(
        self, action: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute action with retry logic."""
        last_error = None
        operation_id = (
            f"{self.config.name}_{action}_{int(datetime.utcnow().timestamp())}"
        )

        for attempt in range(self.config.retry_count + 1):
            try:
                return await asyncio.wait_for(
                    self._execute_action(action, params), timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                last_error = f"Operation timed out after {self.config.timeout} seconds"
                if attempt < self.config.retry_count:
                    self.logger.warning(
                        "Tool action timed out, retrying",
                        extra={
                            "tool_name": self.config.name,
                            "operation_id": operation_id,
                            "action": action,
                            "operation": "tool_execution",
                            "phase": "timeout_retry",
                            "attempt": attempt + 1,
                            "max_attempts": self.config.retry_count + 1,
                            "timeout": self.config.timeout,
                            "retry_delay": self.config.retry_delay,
                        },
                    )
                    await asyncio.sleep(self.config.retry_delay)
            except Exception as e:
                last_error = str(e)
                if attempt < self.config.retry_count:
                    self.logger.warning(
                        "Tool action failed, retrying",
                        extra={
                            "tool_name": self.config.name,
                            "operation_id": operation_id,
                            "action": action,
                            "operation": "tool_execution",
                            "phase": "error_retry",
                            "attempt": attempt + 1,
                            "max_attempts": self.config.retry_count + 1,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "retry_delay": self.config.retry_delay,
                        },
                    )
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    raise

        self.logger.error(
            "Tool action failed after all retry attempts",
            extra={
                "tool_name": self.config.name,
                "operation_id": operation_id,
                "action": action,
                "operation": "tool_execution",
                "phase": "retry_exhausted",
                "total_attempts": self.config.retry_count + 1,
                "final_error": last_error,
            },
        )
        raise ToolError(
            f"Action failed after {self.config.retry_count + 1} attempts: {last_error}"
        )

    def _get_rollback_info(
        self, action: str, params: Dict[str, Any], result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get rollback information for an operation."""
        # Default implementation - can be overridden by subclasses
        return {
            "action": action,
            "params": params,
            "timestamp": datetime.utcnow().isoformat(),
            "reversible": True,
        }


class ToolError(Exception):
    """Base exception for tool-related errors."""


class ToolValidationError(ToolError):
    """Exception raised during parameter validation."""


class ToolExecutionError(ToolError):
    """Exception raised during tool execution."""


class ToolTimeoutError(ToolError):
    """Exception raised when tool operation times out."""
