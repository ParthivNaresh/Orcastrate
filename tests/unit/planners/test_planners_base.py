"""
Tests for planner base classes and functionality.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.agent.base import Plan
from src.planners.base import (
    Planner,
    PlannerConfig,
    PlannerError,
    PlanningStrategy,
    PlanOptimizationError,
    PlanStep,
    PlanStructure,
    PlanValidationError,
    RiskAssessment,
    ValidationResult,
)


class TestPlannerConfig:
    """Test PlannerConfig model."""

    def test_planner_config_defaults(self):
        """Test planner configuration with defaults."""
        config = PlannerConfig()

        assert config.strategy == PlanningStrategy.HYBRID
        assert config.max_plan_steps == 100
        assert config.max_planning_time == 300
        assert config.cost_optimization is True
        assert config.risk_threshold == 0.7
        assert config.parallel_execution is True
        assert config.enable_caching is True
        assert config.template_library_path is None
        assert config.knowledge_base_path is None

    def test_planner_config_custom(self):
        """Test planner configuration with custom values."""
        config = PlannerConfig(
            strategy=PlanningStrategy.HIERARCHICAL,
            max_plan_steps=50,
            max_planning_time=600,
            cost_optimization=False,
            risk_threshold=0.5,
            parallel_execution=False,
            enable_caching=False,
            template_library_path="/path/to/templates",
            knowledge_base_path="/path/to/kb",
        )

        assert config.strategy == PlanningStrategy.HIERARCHICAL
        assert config.max_plan_steps == 50
        assert config.max_planning_time == 600
        assert config.cost_optimization is False
        assert config.risk_threshold == 0.5
        assert config.parallel_execution is False
        assert config.enable_caching is False
        assert config.template_library_path == "/path/to/templates"
        assert config.knowledge_base_path == "/path/to/kb"


class TestPlanStep:
    """Test PlanStep model."""

    def test_plan_step_creation(self):
        """Test basic plan step creation."""
        step = PlanStep(
            id="step1",
            name="Create VPC",
            description="Create Virtual Private Cloud",
            tool="aws_tool",
            action="create_vpc",
            parameters={"cidr": "10.0.0.0/16"},
            estimated_duration=300.0,
            estimated_cost=50.0,
        )

        assert step.id == "step1"
        assert step.name == "Create VPC"
        assert step.description == "Create Virtual Private Cloud"
        assert step.tool == "aws_tool"
        assert step.action == "create_vpc"
        assert step.parameters == {"cidr": "10.0.0.0/16"}
        assert step.dependencies == []
        assert step.estimated_duration == 300.0
        assert step.estimated_cost == 50.0
        assert step.retry_count == 3
        assert step.timeout == 300
        assert step.rollback_action is None
        assert step.validation_rules == []
        assert step.metadata == {}

    def test_plan_step_with_dependencies(self):
        """Test plan step with dependencies and optional fields."""
        step = PlanStep(
            id="step2",
            name="Create Database",
            description="Create PostgreSQL database",
            tool="aws_tool",
            action="create_rds",
            parameters={"engine": "postgresql"},
            dependencies=["step1"],
            estimated_duration=600.0,
            estimated_cost=100.0,
            retry_count=5,
            timeout=1800,
            rollback_action="delete_rds",
            validation_rules=["check_vpc_exists"],
            metadata={"environment": "production"},
        )

        assert step.dependencies == ["step1"]
        assert step.retry_count == 5
        assert step.timeout == 1800
        assert step.rollback_action == "delete_rds"
        assert step.validation_rules == ["check_vpc_exists"]
        assert step.metadata == {"environment": "production"}


class TestValidationResult:
    """Test ValidationResult model."""

    def test_valid_result(self):
        """Test valid validation result."""
        result = ValidationResult(
            valid=True,
            optimizations=["Use smaller instance"],
            risk_score=0.2,
            confidence=0.9,
        )

        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.optimizations == ["Use smaller instance"]
        assert result.risk_score == 0.2
        assert result.confidence == 0.9

    def test_invalid_result(self):
        """Test invalid validation result."""
        result = ValidationResult(
            valid=False,
            errors=["Circular dependency detected", "Tool not available"],
            warnings=["High cost estimate"],
            risk_score=0.8,
            confidence=0.6,
        )

        assert result.valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert result.risk_score == 0.8
        assert result.confidence == 0.6


class TestRiskAssessment:
    """Test RiskAssessment model."""

    def test_risk_assessment_creation(self):
        """Test risk assessment creation."""
        assessment = RiskAssessment(
            overall_risk=0.6,
            risk_factors=["Network complexity", "New technology"],
            mitigation_strategies=["Use proven patterns", "Add monitoring"],
            failure_probability=0.15,
            recovery_time=1800.0,
            impact_assessment={"cost": "medium", "timeline": "high"},
        )

        assert assessment.overall_risk == 0.6
        assert assessment.risk_factors == ["Network complexity", "New technology"]
        assert assessment.mitigation_strategies == [
            "Use proven patterns",
            "Add monitoring",
        ]
        assert assessment.failure_probability == 0.15
        assert assessment.recovery_time == 1800.0
        assert assessment.impact_assessment == {"cost": "medium", "timeline": "high"}


class TestPlanStructure:
    """Test PlanStructure class."""

    def test_plan_structure_creation(self, sample_plan_steps):
        """Test plan structure creation."""
        structure = PlanStructure(
            steps=sample_plan_steps, metadata={"created_by": "test_planner"}
        )

        assert len(structure.steps) == 3
        assert structure.metadata == {"created_by": "test_planner"}

    def test_plan_structure_default_metadata(self, sample_plan_steps):
        """Test plan structure with default metadata."""
        structure = PlanStructure(steps=sample_plan_steps)

        assert structure.metadata == {}


class TestPlanner:
    """Test Planner base class."""

    @pytest.fixture
    def concrete_planner(self, planner_config):
        """Create a concrete planner for testing."""

        class ConcretePlanner(Planner):
            def __init__(self, config, should_fail=False):
                super().__init__(config)
                self.should_fail = should_fail
                self.kb_loaded = False
                self.templates_loaded = False
                self.optimizers_initialized = False
                self.plans_created = []

            async def _generate_initial_plan(self, context):
                if self.should_fail:
                    raise Exception("Plan generation failed")

                steps = [
                    PlanStep(
                        id="test_step_1",
                        name="Test Step 1",
                        description="First test step",
                        tool="test_tool",
                        action="create_resource",
                        estimated_duration=300.0,
                        estimated_cost=50.0,
                    ),
                    PlanStep(
                        id="test_step_2",
                        name="Test Step 2",
                        description="Second test step",
                        tool="test_tool",
                        action="configure_resource",
                        dependencies=["test_step_1"],
                        estimated_duration=600.0,
                        estimated_cost=75.0,
                    ),
                ]

                return PlanStructure(steps=steps)

            async def _gather_context(self, requirements):
                return {
                    "requirements": requirements.model_dump(),
                    "available_tools": ["test_tool"],
                    "patterns": [],
                }

            async def _load_knowledge_base(self):
                self.kb_loaded = True

            async def _load_template_library(self):
                self.templates_loaded = True

            async def _initialize_optimizers(self):
                self.optimizers_initialized = True

        return ConcretePlanner(planner_config)

    def test_planner_initialization(self, concrete_planner):
        """Test planner initialization."""
        assert concrete_planner.config is not None
        assert concrete_planner._knowledge_base is None
        assert concrete_planner._template_library is None
        assert concrete_planner._cost_optimizer is None
        assert concrete_planner._risk_analyzer is None

    @pytest.mark.asyncio
    async def test_planner_initialize(self, concrete_planner):
        """Test planner initialization process."""
        await concrete_planner.initialize()

        assert concrete_planner.kb_loaded is True
        assert concrete_planner.templates_loaded is True
        assert concrete_planner.optimizers_initialized is True

    @pytest.mark.asyncio
    async def test_planner_initialize_failure(self, planner_config):
        """Test planner initialization failure."""

        class FailingPlanner(Planner):
            async def _generate_initial_plan(self, context):
                pass

            async def _gather_context(self, requirements):
                pass

            async def _load_knowledge_base(self):
                raise Exception("KB loading failed")

        planner = FailingPlanner(planner_config)

        with pytest.raises(PlannerError, match="Initialization failed"):
            await planner.initialize()

    @pytest.mark.asyncio
    async def test_create_plan_success(self, concrete_planner, sample_requirements):
        """Test successful plan creation."""
        await concrete_planner.initialize()

        with patch.object(concrete_planner, "_assess_risks") as mock_assess:
            mock_assess.return_value = RiskAssessment(overall_risk=0.3)

            plan = await concrete_planner.create_plan(sample_requirements)

            assert isinstance(plan, Plan)
            assert len(plan.steps) == 2
            assert plan.estimated_cost == 125.0  # 50 + 75
            assert plan.estimated_duration == 900.0  # 300 + 600
            assert plan.requirements == sample_requirements

            # Verify dependencies are extracted correctly
            assert "test_step_2" in plan.dependencies
            assert plan.dependencies["test_step_2"] == ["test_step_1"]

    @pytest.mark.asyncio
    async def test_create_plan_validation_failure(
        self, concrete_planner, sample_requirements
    ):
        """Test plan creation with validation failure."""
        await concrete_planner.initialize()

        # Mock validation to fail
        with patch.object(concrete_planner, "_validate_steps") as mock_validate:
            mock_validate.return_value = ValidationResult(
                valid=False, errors=["Invalid step configuration"]
            )

            with pytest.raises(PlannerError, match="Plan validation failed"):
                await concrete_planner.create_plan(sample_requirements)

    @pytest.mark.asyncio
    async def test_create_plan_generation_failure(
        self, planner_config, sample_requirements
    ):
        """Test plan creation with generation failure."""
        # Test implementation below

        class ConcretePlanner(Planner):
            def __init__(self, config, should_fail=False):
                super().__init__(config)
                self.should_fail = should_fail
                self.kb_loaded = False
                self.templates_loaded = False
                self.optimizers_initialized = False
                self.plans_created = []

            async def _generate_initial_plan(self, context):
                if self.should_fail:
                    raise Exception("Plan generation failed")
                return PlanStructure(steps=[])

            async def _gather_context(self, requirements):
                return {
                    "requirements": requirements.model_dump(),
                    "templates": [],
                    "constraints": {},
                }

        failing_planner = ConcretePlanner(planner_config, should_fail=True)
        await failing_planner.initialize()

        with pytest.raises(PlannerError, match="Plan creation failed"):
            await failing_planner.create_plan(sample_requirements)

    @pytest.mark.asyncio
    async def test_optimize_plan(self, concrete_planner, sample_plan):
        """Test plan optimization."""
        await concrete_planner.initialize()

        optimized_plan = await concrete_planner.optimize_plan(sample_plan)

        assert isinstance(optimized_plan, Plan)
        assert optimized_plan.id == sample_plan.id
        assert len(optimized_plan.steps) == len(sample_plan.steps)

    @pytest.mark.asyncio
    async def test_validate_plan(self, concrete_planner, sample_plan):
        """Test plan validation."""
        await concrete_planner.initialize()

        result = await concrete_planner.validate_plan(sample_plan)

        assert isinstance(result, ValidationResult)
        assert result.valid is True  # Should pass basic validation

    @pytest.mark.asyncio
    async def test_validate_steps_circular_dependency(self, concrete_planner):
        """Test validation detects circular dependencies."""
        await concrete_planner.initialize()

        # Create steps with circular dependency
        steps = [
            PlanStep(
                id="step1",
                name="Step 1",
                description="First step",
                tool="test_tool",
                action="action1",
                dependencies=["step2"],  # Depends on step2
            ),
            PlanStep(
                id="step2",
                name="Step 2",
                description="Second step",
                tool="test_tool",
                action="action2",
                dependencies=["step1"],  # Depends on step1 - circular!
            ),
        ]

        result = await concrete_planner._validate_steps(steps)

        assert result.valid is False
        assert any("Circular dependencies detected" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_validate_steps_missing_dependency(self, concrete_planner):
        """Test validation detects missing dependencies."""
        await concrete_planner.initialize()

        steps = [
            PlanStep(
                id="step1",
                name="Step 1",
                description="First step",
                tool="test_tool",
                action="action1",
                dependencies=["nonexistent_step"],  # Missing dependency
            )
        ]

        result = await concrete_planner._validate_steps(steps)

        assert result.valid is False
        assert any("depends on non-existent step" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_validate_steps_unavailable_tool(self, concrete_planner):
        """Test validation detects unavailable tools."""
        await concrete_planner.initialize()

        # Mock tool availability check to return False
        with patch.object(concrete_planner, "_is_tool_available", return_value=False):
            steps = [
                PlanStep(
                    id="step1",
                    name="Step 1",
                    description="First step",
                    tool="unavailable_tool",
                    action="action1",
                )
            ]

            result = await concrete_planner._validate_steps(steps)

            assert result.valid is False
            assert any(
                "Tool unavailable_tool is not available" in error
                for error in result.errors
            )

    @pytest.mark.asyncio
    async def test_assess_risks_with_analyzer(self, concrete_planner):
        """Test risk assessment with risk analyzer."""
        await concrete_planner.initialize()

        # Mock risk analyzer
        mock_analyzer = Mock()
        mock_analyzer.assess = AsyncMock(
            return_value=RiskAssessment(
                overall_risk=0.5,
                risk_factors=["Custom risk factor"],
                mitigation_strategies=["Custom mitigation"],
            )
        )
        concrete_planner._risk_analyzer = mock_analyzer

        plan_structure = PlanStructure(steps=[])
        assessment = await concrete_planner._assess_risks(plan_structure)

        assert assessment.overall_risk == 0.5
        assert "Custom risk factor" in assessment.risk_factors
        assert "Custom mitigation" in assessment.mitigation_strategies
        mock_analyzer.assess.assert_called_once_with(plan_structure)

    @pytest.mark.asyncio
    async def test_assess_risks_default(self, concrete_planner):
        """Test default risk assessment."""
        await concrete_planner.initialize()

        plan_structure = PlanStructure(steps=[])
        assessment = await concrete_planner._assess_risks(plan_structure)

        # Should return default assessment
        assert assessment.overall_risk == 0.3
        assert "Default risk assessment" in assessment.risk_factors
        assert "Monitor execution closely" in assessment.mitigation_strategies

    @pytest.mark.asyncio
    async def test_optimize_steps(self, concrete_planner, sample_plan_steps):
        """Test step optimization."""
        await concrete_planner.initialize()

        # Enable parallelization and cost optimization
        concrete_planner.config.parallel_execution = True
        concrete_planner.config.cost_optimization = True

        with (
            patch.object(
                concrete_planner, "_can_parallelize", return_value=True
            ) as mock_parallel,
            patch.object(
                concrete_planner, "_optimize_step_cost", side_effect=lambda x: x
            ) as mock_optimize,
        ):
            optimized_steps = await concrete_planner._optimize_steps(sample_plan_steps)

            assert len(optimized_steps) == len(sample_plan_steps)
            # Should have checked parallelization for each step
            assert mock_parallel.call_count == len(sample_plan_steps)
            # Should have optimized cost for each step
            assert mock_optimize.call_count == len(sample_plan_steps)

    def test_extract_dependencies(self, concrete_planner, sample_plan_steps):
        """Test dependency extraction from steps."""
        dependencies = concrete_planner._extract_dependencies(sample_plan_steps)

        expected = {"step2": ["step1"], "step3": ["step1", "step2"]}

        assert dependencies == expected

    def test_calculate_critical_path_duration(
        self, concrete_planner, sample_plan_steps
    ):
        """Test critical path duration calculation."""
        duration = concrete_planner._calculate_critical_path_duration(sample_plan_steps)

        # Simple sum implementation
        expected = sum(step.estimated_duration for step in sample_plan_steps)
        assert duration == expected

    def test_has_circular_dependencies_none(self, concrete_planner, sample_plan_steps):
        """Test circular dependency detection with no cycles."""
        has_cycle = concrete_planner._has_circular_dependencies(sample_plan_steps)
        assert has_cycle is False

    def test_has_circular_dependencies_cycle(self, concrete_planner):
        """Test circular dependency detection with cycle."""
        steps = [
            PlanStep(
                id="step1",
                name="Step 1",
                description="",
                tool="tool",
                action="action",
                dependencies=["step2"],
            ),
            PlanStep(
                id="step2",
                name="Step 2",
                description="",
                tool="tool",
                action="action",
                dependencies=["step1"],
            ),
        ]

        has_cycle = concrete_planner._has_circular_dependencies(steps)
        assert has_cycle is True


class TestPlannerExceptions:
    """Test planner-related exceptions."""

    def test_planner_error(self):
        """Test PlannerError exception."""
        error = PlannerError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_plan_validation_error(self):
        """Test PlanValidationError exception."""
        error = PlanValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, PlannerError)

    def test_plan_optimization_error(self):
        """Test PlanOptimizationError exception."""
        error = PlanOptimizationError("Optimization failed")
        assert str(error) == "Optimization failed"
        assert isinstance(error, PlannerError)
