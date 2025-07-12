"""
Test suite for IntelligentPlanner class.

This module tests the core intelligent planning functionality including
plan generation, optimization, validation, and fallback mechanisms.
"""

from unittest.mock import patch

import pytest

from src.planners.analysis.requirements_analyzer import AnalysisResult
from src.planners.base import PlannerConfig, PlanStep, PlanStructure
from src.planners.intelligent import IntelligentPlanner, IntelligentPlanningStrategy
from src.planners.llm.base import LLMConfig, LLMProvider


class TestIntelligentPlanner:
    """Test IntelligentPlanner functionality."""

    @pytest.fixture
    def planner_config(self):
        """Create basic planner configuration."""
        return PlannerConfig()

    @pytest.fixture
    def llm_config(self):
        """Create LLM configuration for testing."""
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key",
            temperature=0.7,
            max_tokens=4000,
        )

    @pytest.fixture
    def planner_without_llm(self, planner_config):
        """Create planner without LLM for rule-based testing."""
        return IntelligentPlanner(
            config=planner_config,
            llm_config=None,
            strategy=IntelligentPlanningStrategy.REQUIREMENTS_DRIVEN,
        )

    @pytest.fixture
    def mock_analysis_result(self):
        """Create mock analysis result."""
        return AnalysisResult(
            original_requirements="Test requirements",
            detected_technologies=[],
            extracted_requirements=[],
            identified_patterns=[],
            technology_stack={
                "backend": ["Python", "Django"],
                "database": ["PostgreSQL"],
                "infrastructure": ["AWS", "Docker"],
            },
            estimated_complexity=5.0,
            analysis_confidence=0.8,
            completeness_score=0.9,
            ambiguity_score=0.2,
        )

    @pytest.mark.asyncio
    async def test_planner_initialization_without_llm(self, planner_without_llm):
        """Test planner initialization without LLM."""
        await planner_without_llm.initialize()

        assert planner_without_llm.llm_client is None
        assert (
            planner_without_llm.intelligent_strategy
            == IntelligentPlanningStrategy.REQUIREMENTS_DRIVEN
        )
        assert planner_without_llm.fallback_to_template is True

    @pytest.mark.asyncio
    async def test_planner_initialization_with_llm(self, planner_config, llm_config):
        """Test planner initialization with LLM."""
        with patch("src.planners.llm.openai_client.OPENAI_AVAILABLE", True):
            with patch("src.planners.llm.openai_client.openai"):
                planner = IntelligentPlanner(
                    config=planner_config,
                    llm_config=llm_config,
                    strategy=IntelligentPlanningStrategy.LLM_POWERED,
                )

                assert planner.llm_client is not None
                assert (
                    planner.intelligent_strategy
                    == IntelligentPlanningStrategy.LLM_POWERED
                )

    @pytest.mark.asyncio
    async def test_tool_registration(self, planner_without_llm):
        """Test tool capabilities registration."""
        await planner_without_llm.initialize()

        # Register tools
        planner_without_llm.register_tool_capabilities(
            "aws", {"actions": ["create_vpc"]}
        )
        planner_without_llm.register_tool_capabilities(
            "docker", {"actions": ["build", "deploy"]}
        )

        assert "aws" in planner_without_llm._available_tools
        assert "docker" in planner_without_llm._available_tools
        assert planner_without_llm._tool_capabilities["aws"]["actions"] == [
            "create_vpc"
        ]

    @pytest.mark.asyncio
    async def test_create_plan_rule_based(
        self, planner_without_llm, mock_analysis_result
    ):
        """Test plan creation using rule-based approach."""
        await planner_without_llm.initialize()

        # Register required tools
        planner_without_llm.register_tool_capabilities(
            "aws", {"actions": ["create_vpc"]}
        )
        planner_without_llm.register_tool_capabilities(
            "postgresql", {"actions": ["create_database"]}
        )
        planner_without_llm.register_tool_capabilities(
            "docker", {"actions": ["deploy_application"]}
        )

        # Mock the requirements analyzer
        with patch.object(
            planner_without_llm,
            "_analyze_requirements",
            return_value=mock_analysis_result,
        ):
            plan_steps = await planner_without_llm.create_intelligent_plan(
                requirements="Build a Python Django app with PostgreSQL on AWS",
                available_tools=["aws", "postgresql", "docker"],
                constraints={"budget": {"max_monthly": 500}},
            )

        assert len(plan_steps) > 0
        assert isinstance(plan_steps[0], PlanStep)

        # Check that we have database and backend steps
        step_names = [step.name for step in plan_steps]
        assert any("PostgreSQL" in name for name in step_names)
        assert any("Backend" in name or "Application" in name for name in step_names)

    @pytest.mark.asyncio
    async def test_plan_validation_success(self, planner_without_llm):
        """Test successful plan validation."""
        await planner_without_llm.initialize()
        planner_without_llm.register_tool_capabilities(
            "aws", {"actions": ["create_vpc"]}
        )

        plan_steps = [
            PlanStep(
                id="step_1",
                name="Test Step",
                description="Test description",
                tool="aws",
                action="create_vpc",
                dependencies=[],
                estimated_duration=300.0,
                estimated_cost=50.0,
            )
        ]

        plan_structure = PlanStructure(steps=plan_steps)
        validation_result = await planner_without_llm._validate_plan(plan_structure)

        assert validation_result.valid is True
        assert len(validation_result.errors) == 0

    @pytest.mark.asyncio
    async def test_plan_validation_tool_not_available(self, planner_without_llm):
        """Test plan validation with unavailable tool."""
        await planner_without_llm.initialize()

        plan_steps = [
            PlanStep(
                id="step_1",
                name="Test Step",
                description="Test description",
                tool="unavailable_tool",
                action="some_action",
                dependencies=[],
                estimated_duration=300.0,
                estimated_cost=50.0,
            )
        ]

        plan_structure = PlanStructure(steps=plan_steps)
        validation_result = await planner_without_llm._validate_plan(plan_structure)

        assert validation_result.valid is False
        assert len(validation_result.errors) > 0
        assert "not available" in validation_result.errors[0]

    @pytest.mark.asyncio
    async def test_plan_validation_circular_dependencies(self, planner_without_llm):
        """Test plan validation with circular dependencies."""
        await planner_without_llm.initialize()
        planner_without_llm.register_tool_capabilities("aws", {"actions": ["action"]})

        plan_steps = [
            PlanStep(
                id="step_1",
                name="Step 1",
                description="First step",
                tool="aws",
                action="action",
                dependencies=["step_2"],
                estimated_duration=300.0,
                estimated_cost=50.0,
            ),
            PlanStep(
                id="step_2",
                name="Step 2",
                description="Second step",
                tool="aws",
                action="action",
                dependencies=["step_1"],  # Circular dependency
                estimated_duration=300.0,
                estimated_cost=50.0,
            ),
        ]

        plan_structure = PlanStructure(steps=plan_steps)
        validation_result = await planner_without_llm._validate_plan(plan_structure)

        assert validation_result.valid is False
        assert any("circular" in error.lower() for error in validation_result.errors)

    @pytest.mark.asyncio
    async def test_cost_optimization(self, planner_without_llm):
        """Test cost optimization functionality."""
        plan_steps = [
            PlanStep(
                id="step_1",
                name="Create Instance",
                description="Create AWS instance",
                tool="aws",
                action="create_instance",
                parameters={"instance_type": "t3.large"},
                dependencies=[],
                estimated_duration=300.0,
                estimated_cost=100.0,
            )
        ]

        optimized_steps = planner_without_llm._optimize_for_cost(plan_steps)

        # Should optimize instance type from large to medium
        assert optimized_steps[0].parameters["instance_type"] == "t3.medium"
        assert optimized_steps[0].estimated_cost == 70.0  # 100 * 0.7

    @pytest.mark.asyncio
    async def test_timeline_optimization(self, planner_without_llm):
        """Test timeline optimization functionality."""
        plan_steps = [
            PlanStep(
                id="step_1",
                name="Independent Step",
                description="Step with no dependencies",
                tool="aws",
                action="create_vpc",
                dependencies=[],
                estimated_duration=300.0,
                estimated_cost=50.0,
            )
        ]

        optimized_steps = planner_without_llm._optimize_for_timeline(plan_steps)

        # Should mark independent steps for parallel execution
        assert optimized_steps[0].parameters.get("parallel_execution") is True

    @pytest.mark.asyncio
    async def test_remove_redundant_steps(self, planner_without_llm):
        """Test removal of redundant steps."""
        plan_steps = [
            PlanStep(
                id="step_1",
                name="Create VPC",
                description="First VPC creation",
                tool="aws",
                action="create_vpc",
                dependencies=[],
                estimated_duration=300.0,
                estimated_cost=50.0,
            ),
            PlanStep(
                id="step_2",
                name="Create Another VPC",
                description="Duplicate VPC creation",
                tool="aws",
                action="create_vpc",  # Same tool:action combination
                dependencies=[],
                estimated_duration=300.0,
                estimated_cost=50.0,
            ),
        ]

        unique_steps = planner_without_llm._remove_redundant_steps(plan_steps)

        # Should remove duplicate aws:create_vpc
        assert len(unique_steps) == 1
        assert unique_steps[0].id == "step_1"

    @pytest.mark.asyncio
    async def test_dependency_optimization(self, planner_without_llm):
        """Test dependency optimization."""
        plan_steps = [
            PlanStep(
                id="step_1",
                name="Step 1",
                description="First step",
                tool="aws",
                action="action",
                dependencies=[
                    "step_1",
                    "non_existent_step",
                ],  # Self-dep and invalid dep
                estimated_duration=300.0,
                estimated_cost=50.0,
            )
        ]

        optimized_steps = planner_without_llm._optimize_dependencies(plan_steps)

        # Should remove self-dependency and non-existent dependency
        assert len(optimized_steps[0].dependencies) == 0

    @pytest.mark.asyncio
    async def test_explain_plan_without_llm(self, planner_without_llm):
        """Test plan explanation without LLM."""
        plan_steps = [
            PlanStep(
                id="step_1",
                name="Create Database",
                description="Setup PostgreSQL database",
                tool="postgresql",
                action="create_database",
                dependencies=[],
                estimated_duration=600.0,
                estimated_cost=25.0,
            ),
            PlanStep(
                id="step_2",
                name="Deploy Application",
                description="Deploy backend application",
                tool="docker",
                action="deploy_application",
                dependencies=["step_1"],
                estimated_duration=900.0,
                estimated_cost=40.0,
            ),
        ]

        explanation = await planner_without_llm.explain_plan(plan_steps)

        assert "2 steps" in explanation
        assert "Create Database" in explanation
        assert "Deploy Application" in explanation
        assert "$65.00" in explanation  # Total cost
        assert "0.4 hours" in explanation  # Total duration in hours

    @pytest.mark.asyncio
    async def test_get_planning_recommendations(
        self, planner_without_llm, mock_analysis_result
    ):
        """Test planning recommendations generation."""
        # Set up analysis result with specific scores
        mock_analysis_result.ambiguity_score = 0.6  # High ambiguity
        mock_analysis_result.completeness_score = 0.5  # Low completeness
        mock_analysis_result.detected_technologies = []  # No technologies

        with patch.object(
            planner_without_llm,
            "_analyze_requirements",
            return_value=mock_analysis_result,
        ):
            recommendations = await planner_without_llm.get_planning_recommendations(
                "Vague requirements"
            )

        assert "analysis_confidence" in recommendations
        assert "recommendations" in recommendations

        # Should have recommendations for high ambiguity, low completeness, and no technologies
        assert len(recommendations["recommendations"]) >= 3
        assert any(
            "specific details" in rec for rec in recommendations["recommendations"]
        )
        assert any(
            "performance, security" in rec for rec in recommendations["recommendations"]
        )
        assert any(
            "technologies or frameworks" in rec
            for rec in recommendations["recommendations"]
        )

    @pytest.mark.asyncio
    async def test_fallback_plan_creation(self, planner_without_llm):
        """Test template fallback plan creation."""
        available_tools = ["aws", "docker"]

        fallback_steps = await planner_without_llm._create_template_fallback_plan(
            "Some requirements", available_tools
        )

        assert len(fallback_steps) > 0
        assert fallback_steps[0].name == "Basic Infrastructure Setup"
        assert fallback_steps[0].tool == "aws"  # Should use AWS since it's available

    @pytest.mark.asyncio
    async def test_cost_estimates_update(self, planner_without_llm):
        """Test cost estimates update functionality."""
        cost_data = {
            "aws:create_vpc": 10.0,
            "docker:deploy": 20.0,
            "postgresql:create_database": 15.0,
        }

        planner_without_llm.update_cost_estimates(cost_data)

        assert planner_without_llm._cost_estimates["aws:create_vpc"] == 10.0
        assert planner_without_llm._cost_estimates["docker:deploy"] == 20.0
        assert planner_without_llm._cost_estimates["postgresql:create_database"] == 15.0

    def test_has_circular_dependencies_detection(self, planner_without_llm):
        """Test circular dependency detection algorithm."""
        # Create plan with circular dependencies
        circular_steps = [
            PlanStep(
                id="step_1",
                name="Step 1",
                description="",
                tool="aws",
                action="action",
                dependencies=["step_2"],
                estimated_duration=300.0,
                estimated_cost=50.0,
            ),
            PlanStep(
                id="step_2",
                name="Step 2",
                description="",
                tool="aws",
                action="action",
                dependencies=["step_3"],
                estimated_duration=300.0,
                estimated_cost=50.0,
            ),
            PlanStep(
                id="step_3",
                name="Step 3",
                description="",
                tool="aws",
                action="action",
                dependencies=["step_1"],
                estimated_duration=300.0,
                estimated_cost=50.0,
            ),
        ]

        assert planner_without_llm._has_circular_dependencies(circular_steps) is True

        # Create plan without circular dependencies
        linear_steps = [
            PlanStep(
                id="step_1",
                name="Step 1",
                description="",
                tool="aws",
                action="action",
                dependencies=[],
                estimated_duration=300.0,
                estimated_cost=50.0,
            ),
            PlanStep(
                id="step_2",
                name="Step 2",
                description="",
                tool="aws",
                action="action",
                dependencies=["step_1"],
                estimated_duration=300.0,
                estimated_cost=50.0,
            ),
        ]

        assert planner_without_llm._has_circular_dependencies(linear_steps) is False

    @pytest.mark.asyncio
    async def test_infrastructure_steps_creation(self, planner_without_llm):
        """Test infrastructure steps creation."""
        infrastructure_techs = ["AWS", "Docker"]
        available_tools = ["aws", "docker"]

        infra_steps = planner_without_llm._create_infrastructure_steps(
            infrastructure_techs, available_tools, 1
        )

        # Should create both AWS and Docker steps
        assert len(infra_steps) == 2

        # Check AWS step
        aws_step = next((s for s in infra_steps if s.tool == "aws"), None)
        assert aws_step is not None
        assert "AWS Infrastructure" in aws_step.name
        assert aws_step.action == "create_vpc"

        # Check Docker step
        docker_step = next((s for s in infra_steps if s.tool == "docker"), None)
        assert docker_step is not None
        assert "Container Environment" in docker_step.name
        assert docker_step.action == "setup_environment"

    @pytest.mark.asyncio
    async def test_database_steps_creation(self, planner_without_llm):
        """Test database steps creation."""
        database_techs = ["PostgreSQL", "MySQL", "MongoDB"]
        available_tools = ["postgresql", "mysql", "mongodb"]

        db_steps = planner_without_llm._create_database_steps(
            database_techs, available_tools, 1
        )

        # Should create steps for all three databases
        assert len(db_steps) == 3

        # Check PostgreSQL step
        pg_step = next((s for s in db_steps if s.tool == "postgresql"), None)
        assert pg_step is not None
        assert "PostgreSQL Database" in pg_step.name
        assert pg_step.parameters["name"] == "main_db"

        # Check MySQL step
        mysql_step = next((s for s in db_steps if s.tool == "mysql"), None)
        assert mysql_step is not None
        assert "MySQL Database" in mysql_step.name

        # Check MongoDB step
        mongo_step = next((s for s in db_steps if s.tool == "mongodb"), None)
        assert mongo_step is not None
        assert "MongoDB Database" in mongo_step.name

    @pytest.mark.asyncio
    async def test_security_steps_creation(
        self, planner_without_llm, mock_analysis_result
    ):
        """Test security steps creation based on analysis."""
        from src.planners.analysis.requirements_analyzer import (
            AnalysisPriority,
            Requirement,
            RequirementType,
        )

        # Add security requirements to analysis result
        security_req = Requirement(
            id="req_1",
            type=RequirementType.SECURITY,
            priority=AnalysisPriority.HIGH,
            description="SSL encryption required",
            keywords=["ssl", "encryption"],
            confidence=0.9,
        )
        mock_analysis_result.extracted_requirements = [security_req]

        security_steps = planner_without_llm._create_security_steps(
            mock_analysis_result, ["aws"], 1
        )

        # Should create security step when security requirements are present
        assert len(security_steps) == 1
        assert "Configure Security" in security_steps[0].name
        assert security_steps[0].tool == "aws"
        assert security_steps[0].parameters["enable_ssl"] is True
        assert security_steps[0].parameters["enable_waf"] is True
