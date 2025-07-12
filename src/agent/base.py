"""
Core agent abstractions and base classes.

This module defines the fundamental interfaces for the agent system,
following the hexagonal architecture pattern to ensure clean separation
of concerns and testability.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentStatus(Enum):
    """Agent execution status."""

    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    FAILED = "failed"
    COMPLETED = "completed"


class ExecutionResult(BaseModel):
    """Result of an agent execution."""

    success: bool
    execution_id: str
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    duration: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Requirements(BaseModel):
    """Environment requirements specification."""

    description: str
    framework: Optional[str] = None
    database: Optional[str] = None
    cloud_provider: Optional[str] = None
    scaling_requirements: Optional[Dict[str, Any]] = None
    security_requirements: Optional[Dict[str, Any]] = None
    budget_constraints: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """Execution plan for environment creation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    estimated_cost: float = 0.0
    estimated_duration: float = 0.0
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    requirements: Requirements


class Agent(ABC):
    """
    Base agent interface following the Command pattern.

    This abstract base class defines the core operations that all agents
    must implement. It provides a consistent interface for planning,
    execution, and monitoring of development environment operations.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.status = AgentStatus.IDLE
        self.logger = logging.getLogger(f"{self.__class__.__name__}:{agent_id}")
        self._execution_history: List[ExecutionResult] = []

    @abstractmethod
    async def plan(self, requirements: Requirements) -> Plan:
        """
        Create an execution plan based on requirements.

        Args:
            requirements: Environment requirements specification

        Returns:
            Plan: Detailed execution plan

        Raises:
            PlanningError: If planning fails
        """

    @abstractmethod
    async def execute(self, plan: Plan) -> ExecutionResult:
        """
        Execute a plan and return the result.

        Args:
            plan: Execution plan to run

        Returns:
            ExecutionResult: Result of the execution

        Raises:
            ExecutionError: If execution fails
        """

    @abstractmethod
    async def monitor(self, execution_id: str) -> Dict[str, Any]:
        """
        Monitor an ongoing execution.

        Args:
            execution_id: ID of the execution to monitor

        Returns:
            Dict[str, Any]: Current status and metrics

        Raises:
            MonitoringError: If monitoring fails
        """

    async def validate_requirements(self, requirements: Requirements) -> bool:
        """
        Validate requirements before planning.

        Args:
            requirements: Requirements to validate

        Returns:
            bool: True if requirements are valid
        """
        # Basic validation - can be extended by subclasses
        return bool(requirements.description)

    def get_execution_history(self) -> List[ExecutionResult]:
        """Get execution history for this agent."""
        return self._execution_history.copy()

    def add_execution_result(self, result: ExecutionResult) -> None:
        """Add an execution result to history."""
        self._execution_history.append(result)
        # Keep only last 100 results to prevent memory issues
        if len(self._execution_history) > 100:
            self._execution_history = self._execution_history[-100:]


class AgentRegistry:
    """
    Registry for managing agent instances.

    This singleton class provides centralized management of agent instances,
    supporting dynamic agent creation and lifecycle management.
    """

    _instance: Optional["AgentRegistry"] = None

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._agents: Dict[str, Agent] = {}
            self._agent_types: Dict[str, type] = {}
            self._initialized = True

    def __new__(cls) -> "AgentRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register_agent_type(self, agent_type: str, agent_class: type) -> None:
        """Register an agent type for dynamic creation."""
        self._agent_types[agent_type] = agent_class

    def create_agent(
        self, agent_type: str, agent_id: str, config: Dict[str, Any]
    ) -> Agent:
        """Create and register a new agent instance."""
        if agent_type not in self._agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent_class = self._agent_types[agent_type]
        agent = agent_class(agent_id, config)
        self._agents[agent_id] = agent
        return agent  # type: ignore[no-any-return]

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the registry."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    def list_agents(self) -> List[str]:
        """List all registered agent IDs."""
        return list(self._agents.keys())


class AgentError(Exception):
    """Base exception for agent-related errors."""


class PlanningError(AgentError):
    """Exception raised during planning phase."""


class ExecutionError(AgentError):
    """Exception raised during execution phase."""


class MonitoringError(AgentError):
    """Exception raised during monitoring phase."""
