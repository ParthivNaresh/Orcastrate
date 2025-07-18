"""
Base planner interface and implementations.

This module defines the planning system that creates execution plans
from requirements using various planning strategies and algorithms.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..agent.base import Plan, Requirements


class PlanningStrategy(Enum):
    """Available planning strategies."""

    HIERARCHICAL = "hierarchical"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    TEMPLATE_MATCHING = "template_matching"
    HYBRID = "hybrid"


class PlanStep(BaseModel):
    """Individual step in an execution plan."""

    id: str
    name: str
    description: str
    tool: str
    action: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    estimated_duration: float = 0.0
    estimated_cost: float = 0.0
    retry_count: int = 3
    timeout: int = 300
    rollback_action: Optional[str] = None
    validation_rules: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Result of plan validation."""

    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    optimizations: List[str] = Field(default_factory=list)
    risk_score: float = 0.0
    confidence: float = 0.0


class RiskAssessment(BaseModel):
    """Risk assessment for a plan."""

    overall_risk: float = 0.0  # 0-1 scale
    risk_factors: List[str] = Field(default_factory=list)
    mitigation_strategies: List[str] = Field(default_factory=list)
    failure_probability: float = 0.0
    recovery_time: float = 0.0
    impact_assessment: Dict[str, Any] = Field(default_factory=dict)


class PlannerConfig(BaseModel):
    """Configuration for a planner."""

    strategy: PlanningStrategy = PlanningStrategy.HYBRID
    max_plan_steps: int = 100
    max_planning_time: int = 300  # seconds
    cost_optimization: bool = True
    risk_threshold: float = 0.7
    parallel_execution: bool = True
    enable_caching: bool = True
    template_library_path: Optional[str] = None
    knowledge_base_path: Optional[str] = None


class Planner(ABC):
    """
    Base planner interface for creating execution plans.

    This abstract base class defines the contract that all planners must implement.
    It provides a consistent interface for plan creation, optimization, and validation.
    """

    def __init__(self, config: PlannerConfig, progress_tracker=None):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._knowledge_base: Optional[Dict[str, Any]] = None
        self._template_library: Optional[Dict[str, Any]] = None
        self._cost_optimizer: Optional[Any] = None
        self._risk_analyzer: Optional[Any] = None
        self._progress_tracker = progress_tracker

    async def initialize(self) -> None:
        """Initialize the planner and its components."""
        try:
            await self._load_knowledge_base()
            await self._load_template_library()
            await self._initialize_optimizers()

        except Exception as e:
            self.logger.error(f"Failed to initialize planner: {e}")
            raise PlannerError(f"Initialization failed: {e}")

    async def create_plan(self, requirements: Requirements) -> Plan:
        """
        Create an execution plan from requirements.

        Args:
            requirements: Environment requirements

        Returns:
            Plan: Complete execution plan
        """
        try:
            # Gather context and analyze requirements
            context = await self._gather_context(requirements)

            if self._progress_tracker:
                self._progress_tracker.update_step_progress()
                self._progress_tracker.add_step_message(
                    "🧩 Context gathered", 1, completed=True
                )

            initial_plan = await self._generate_initial_plan(context)

            if self._progress_tracker:
                self._progress_tracker.update_step_progress()
                self._progress_tracker.add_step_message(
                    "📐 Plan initialized", 1, completed=True
                )

            optimized_plan = await self._optimize_plan(initial_plan)

            if self._progress_tracker:
                self._progress_tracker.update_step_progress()
                self._progress_tracker.add_step_message(
                    "⚡ Plan optimized", 1, completed=True
                )

            validation = await self._validate_plan(optimized_plan)
            if not validation.valid:
                if self._progress_tracker:
                    self._progress_tracker.update_step_progress()
                    self._progress_tracker.add_step_message(
                        "🔍 Plan validation failed", 1, completed=False
                    )
                raise PlannerError(f"Plan validation failed: {validation.errors}")

            if self._progress_tracker:
                self._progress_tracker.update_step_progress()
                self._progress_tracker.add_step_message(
                    "🔍 Plan validated", 1, completed=True
                )

            # Perform risk assessment
            risk_assessment = await self._assess_risks(optimized_plan)

            if self._progress_tracker:
                self._progress_tracker.update_step_progress()
                self._progress_tracker.add_step_message(
                    "🛡️ Risks assessed", 1, completed=True
                )

            # Create final plan object
            plan = Plan(
                steps=[step.model_dump() for step in optimized_plan.steps],
                dependencies=self._extract_dependencies(optimized_plan.steps),
                estimated_cost=sum(
                    step.estimated_cost for step in optimized_plan.steps
                ),
                estimated_duration=self._calculate_critical_path_duration(
                    optimized_plan.steps
                ),
                risk_assessment=risk_assessment.model_dump(),
                requirements=requirements,
            )

            # duration = (datetime.utcnow() - start_time).total_seconds()

            return plan

        except Exception as e:
            self.logger.error(f"Plan creation failed: {e}")
            raise PlannerError(f"Plan creation failed: {e}")

    async def optimize_plan(self, plan: Plan) -> Plan:
        """
        Optimize an existing plan.

        Args:
            plan: Plan to optimize

        Returns:
            Plan: Optimized plan
        """
        # Convert plan steps back to PlanStep objects
        steps = [PlanStep(**step) for step in plan.steps]
        optimized_steps = await self._optimize_steps(steps)

        # Create optimized plan
        optimized_plan = Plan(
            id=plan.id,
            steps=[step.model_dump() for step in optimized_steps],
            dependencies=self._extract_dependencies(optimized_steps),
            estimated_cost=sum(step.estimated_cost for step in optimized_steps),
            estimated_duration=self._calculate_critical_path_duration(optimized_steps),
            risk_assessment=plan.risk_assessment,
            requirements=plan.requirements,
        )

        return optimized_plan

    async def validate_plan(self, plan: Plan) -> ValidationResult:
        """
        Validate a plan for correctness and feasibility.

        Args:
            plan: Plan to validate

        Returns:
            ValidationResult: Validation result
        """
        steps = [PlanStep(**step) for step in plan.steps]
        return await self._validate_steps(steps)

    @abstractmethod
    async def _generate_initial_plan(self, context: Dict[str, Any]) -> "PlanStructure":
        """Generate the initial plan structure."""

    @abstractmethod
    async def _gather_context(self, requirements: Requirements) -> Dict[str, Any]:
        """Gather context for planning."""

    async def _optimize_plan(self, plan: "PlanStructure") -> "PlanStructure":
        """Optimize the plan for cost and performance."""
        if not self.config.cost_optimization:
            return plan

        # Cost optimization
        if self._cost_optimizer:
            plan = await self._cost_optimizer.optimize(plan)

        # Performance optimization
        optimized_steps = await self._optimize_steps(plan.steps)

        return PlanStructure(steps=optimized_steps, metadata=plan.metadata)

    async def _validate_plan(self, plan: "PlanStructure") -> ValidationResult:
        """Validate the plan."""
        return await self._validate_steps(plan.steps)

    async def _validate_steps(self, steps: List[PlanStep]) -> ValidationResult:
        """Validate individual steps and their dependencies."""
        errors: List[str] = []
        warnings: List[str] = []

        # Check for circular dependencies
        if self._has_circular_dependencies(steps):
            errors.append("Circular dependencies detected")

        # Validate each step
        for step in steps:
            # Check if required tools are available
            if not await self._is_tool_available(step.tool):
                errors.append(f"Tool {step.tool} is not available")

            # Validate dependencies
            for dep in step.dependencies:
                if not any(s.id == dep for s in steps):
                    errors.append(f"Step {step.id} depends on non-existent step {dep}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=0.9 if len(errors) == 0 else 0.1,
        )

    async def _assess_risks(self, plan: "PlanStructure") -> RiskAssessment:
        """Assess risks in the plan."""
        if self._risk_analyzer:
            return await self._risk_analyzer.assess(plan)  # type: ignore[no-any-return]

        # Default risk assessment
        return RiskAssessment(
            overall_risk=0.3,
            risk_factors=["Default risk assessment"],
            mitigation_strategies=["Monitor execution closely"],
            failure_probability=0.1,
            recovery_time=300.0,
        )

    async def _optimize_steps(self, steps: List[PlanStep]) -> List[PlanStep]:
        """Optimize individual steps."""
        optimized_steps = []

        for step in steps:
            # Check if step can be parallelized
            if self.config.parallel_execution and await self._can_parallelize(step):
                step.metadata["parallelizable"] = True

            # Apply cost optimizations
            if self.config.cost_optimization:
                step = await self._optimize_step_cost(step)

            optimized_steps.append(step)

        return optimized_steps

    async def _load_knowledge_base(self) -> None:
        """Load the knowledge base."""
        if self.config.knowledge_base_path:
            # TODO: Implement knowledge base loading
            pass

        # Default knowledge base
        self._knowledge_base = {
            "patterns": {},
            "best_practices": {},
            "compatibility_matrix": {},
        }

    async def _load_template_library(self) -> None:
        """Load template library."""
        if self.config.template_library_path:
            # TODO: Implement template library loading
            pass

        # Default templates
        self._template_library = {
            "web_app": [],
            "microservice": [],
            "data_pipeline": [],
        }

    async def _initialize_optimizers(self) -> None:
        """Initialize cost and risk optimizers."""
        # TODO: Initialize actual optimizers

    def _extract_dependencies(self, steps: List[PlanStep]) -> Dict[str, List[str]]:
        """Extract dependency graph from steps."""
        dependencies = {}
        for step in steps:
            if step.dependencies:
                dependencies[step.id] = step.dependencies
        return dependencies

    def _calculate_critical_path_duration(self, steps: List[PlanStep]) -> float:
        """Calculate critical path duration."""
        # Simple implementation - can be improved with proper critical path algorithm
        return sum(step.estimated_duration for step in steps)

    def _has_circular_dependencies(self, steps: List[PlanStep]) -> bool:
        """Check for circular dependencies."""
        # Simple cycle detection - can be improved
        visited = set()
        rec_stack = set()

        def has_cycle(step_id: str) -> bool:
            if step_id in rec_stack:
                return True
            if step_id in visited:
                return False

            visited.add(step_id)
            rec_stack.add(step_id)

            # Find step dependencies
            for step in steps:
                if step.id == step_id:
                    for dep in step.dependencies:
                        if has_cycle(dep):
                            return True

            rec_stack.remove(step_id)
            return False

        for step in steps:
            if step.id not in visited:
                if has_cycle(step.id):
                    return True

        return False

    async def _is_tool_available(self, tool_name: str) -> bool:
        """Check if a tool is available."""
        # TODO: Implement tool availability check
        return True

    async def _can_parallelize(self, step: PlanStep) -> bool:
        """Check if a step can be parallelized."""
        # TODO: Implement parallelization logic
        return False

    async def _optimize_step_cost(self, step: PlanStep) -> PlanStep:
        """Optimize a single step for cost."""
        # TODO: Implement step-level cost optimization
        return step


class PlanStructure:
    """Internal structure for plan manipulation."""

    def __init__(
        self, steps: List[PlanStep], metadata: Optional[Dict[str, Any]] = None
    ):
        self.steps = steps
        self.metadata = metadata or {}


class PlannerError(Exception):
    """Base exception for planner-related errors."""


class PlanValidationError(PlannerError):
    """Exception raised during plan validation."""


class PlanOptimizationError(PlannerError):
    """Exception raised during plan optimization."""
