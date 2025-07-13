"""
Tests for Template Planner implementation.
"""

import pytest

from src.agent.base import Requirements
from src.planners.base import PlannerConfig, PlanningStrategy
from src.planners.template_planner import TemplatePlanner


class TestTemplatePlanner:
    """Test Template Planner functionality."""

    @pytest.fixture
    def planner_config(self):
        """Create planner configuration."""
        return PlannerConfig(
            strategy=PlanningStrategy.TEMPLATE_MATCHING,
            max_plan_steps=20,
            max_planning_time=60,
            cost_optimization=True,
            risk_threshold=0.8,
        )

    @pytest.fixture
    async def planner(self, planner_config):
        """Create and initialize template planner."""
        planner = TemplatePlanner(planner_config)
        await planner.initialize()
        return planner

    @pytest.mark.asyncio
    async def test_planner_initialization(self, planner):
        """Test planner initialization."""
        assert planner._loaded is True
        assert len(planner._technology_patterns) > 0
        assert "frameworks" in planner._technology_patterns
        assert "databases" in planner._technology_patterns

    @pytest.mark.asyncio
    async def test_get_available_templates(self, planner):
        """Test getting available templates."""
        templates = await planner.get_available_templates()

        assert len(templates) > 0

        for template in templates:
            assert "id" in template
            assert "name" in template
            assert "description" in template
            assert "framework" in template
            assert "estimated_duration" in template
            assert "estimated_cost" in template

        # Check specific templates exist
        template_names = [t["name"] for t in templates]
        assert "Node.js Web Application" in template_names
        assert "FastAPI REST API" in template_names

    @pytest.mark.asyncio
    async def test_technology_detection_nodejs(self, planner):
        """Test technology detection for Node.js requirements."""
        description = "Create a Node.js web application"

        detected = planner.detect_technologies(description)

        assert detected.framework == "nodejs"
        # Test composition works
        steps = planner.compose_plan_from_technologies(detected)
        assert len(steps) > 0

    @pytest.mark.asyncio
    async def test_technology_detection_fastapi(self, planner):
        """Test technology detection for FastAPI requirements."""
        description = "Build a FastAPI REST API"

        detected = planner.detect_technologies(description)

        assert detected.framework == "fastapi"
        # Test composition works
        steps = planner.compose_plan_from_technologies(detected)
        assert len(steps) > 0

    @pytest.mark.asyncio
    async def test_technology_detection_by_description(self, planner):
        """Test technology detection based on description keywords."""
        # Test Node.js detection by description
        description = "Express web application with JavaScript"
        detected = planner.detect_technologies(description)
        assert detected.framework == "nodejs"

        # Test FastAPI detection by description
        description = "Python API service with FastAPI"
        detected = planner.detect_technologies(description)
        assert detected.framework == "fastapi"

    @pytest.mark.asyncio
    async def test_technology_detection_fallback(self, planner):
        """Test technology detection for generic descriptions."""
        description = "A web application"
        detected = planner.detect_technologies(description)

        # Generic description may not detect specific framework
        # But plan should still be generated with base components
        steps = planner.compose_plan_from_technologies(detected)
        assert len(steps) >= 2  # At least base steps (directory, git)

    @pytest.mark.asyncio
    async def test_technology_detection_no_match(self, planner):
        """Test technology detection when no known technology matches."""
        description = "Desktop application with C++"
        detected = planner.detect_technologies(description)

        # Should not detect any supported technologies
        assert detected.framework is None
        assert len(detected.databases) == 0
        assert len(detected.cache) == 0

        # But should still generate base steps
        steps = planner.compose_plan_from_technologies(detected)
        assert len(steps) >= 2  # Base project setup steps

    @pytest.mark.asyncio
    async def test_multi_technology_detection(self, planner):
        """Test detection of multiple technologies in one description."""
        description = (
            "Node.js web application with PostgreSQL database and Redis caching"
        )
        detected = planner.detect_technologies(description)

        assert detected.framework == "nodejs"
        assert "postgresql" in detected.databases
        assert "redis" in detected.cache

        # Should generate more steps for multi-technology setup
        steps = planner.compose_plan_from_technologies(detected)
        assert len(steps) > 5  # More than basic setup due to additional services

    @pytest.mark.asyncio
    async def test_project_name_generation(self, planner):
        """Test project name generation from descriptions."""
        test_cases = [
            ("Node.js web application", "node-js-web"),
            ("FastAPI REST API service", "fastapi-rest-api"),
            ("My awesome app", "my-awesome-app"),
            ("Simple web app with database", "simple-web-app"),
            ("A", "a"),  # Single word
            ("", "orcastrate-project"),  # Empty should fallback
        ]

        for description, expected_pattern in test_cases:
            name = planner._generate_project_name(description)
            assert name
            if expected_pattern == "orcastrate-project":
                assert name == "orcastrate-project"
            else:
                assert expected_pattern.split("-")[0] in name or name.startswith(
                    expected_pattern.split("-")[0]
                )

    @pytest.mark.asyncio
    async def test_variable_substitution(self, planner):
        """Test variable substitution in step data."""
        template_data = {
            "name": "{project_name}",
            "path": "{project_path}/src",
            "nested": {
                "description": "This is {project_name} with {framework}",
                "array": ["{project_name}-item1", "{project_name}-item2"],
            },
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
        assert result["nested"]["array"] == ["my-app-item1", "my-app-item2"]

    @pytest.mark.asyncio
    async def test_step_processing_and_substitution(self, planner):
        """Test processing composed steps with variable substitution."""
        from src.agent.base import Requirements

        requirements = Requirements(
            description="Test Node.js application", framework="nodejs"
        )

        # Get composed steps for nodejs
        detected = planner.detect_technologies(requirements.description)
        if requirements.framework:
            detected.framework = requirements.framework.lower()
        composed_steps = planner.compose_plan_from_technologies(detected)

        # Process them with variable substitution
        steps = await planner._process_composed_steps(composed_steps, requirements)

        assert len(steps) > 0

        # Verify step structure
        for step in steps:
            assert hasattr(step, "id")
            assert hasattr(step, "name")
            assert hasattr(step, "description")
            assert hasattr(step, "tool")
            assert hasattr(step, "action")
            assert hasattr(step, "parameters")
            assert hasattr(step, "estimated_duration")
            assert hasattr(step, "estimated_cost")

        # Check that variables were replaced
        step_data = [step.model_dump() for step in steps]
        step_content = str(step_data)
        assert "{project_name}" not in step_content  # Variables should be replaced
        assert "test-node-js" in step_content.lower() or "test" in step_content.lower()

    @pytest.mark.asyncio
    async def test_create_plan_nodejs(self, planner):
        """Test creating a complete plan for Node.js application."""
        requirements = Requirements(
            description="Create a Node.js web application with Express",
            framework="nodejs",
        )

        plan = await planner.create_plan(requirements)

        assert plan is not None
        assert plan.id
        assert len(plan.steps) > 0
        assert plan.estimated_duration > 0
        assert plan.estimated_cost >= 0
        assert plan.requirements == requirements

        # Verify plan structure
        step_ids = [step["id"] for step in plan.steps]
        assert "setup_directory" in step_ids
        assert "init_git" in step_ids
        assert "create_package_json" in step_ids
        # Enhanced Template Planner only adds Docker for multi-service apps
        # Basic Node.js app gets simpler structure

        # Verify dependencies
        assert plan.dependencies
        # Git init should depend on directory setup
        if "init_git" in plan.dependencies:
            assert "setup_directory" in plan.dependencies["init_git"]

    @pytest.mark.asyncio
    async def test_create_plan_fastapi(self, planner):
        """Test creating a complete plan for FastAPI application."""
        requirements = Requirements(
            description="Build a FastAPI REST API service", framework="fastapi"
        )

        plan = await planner.create_plan(requirements)

        assert plan is not None
        assert len(plan.steps) > 0

        # Verify FastAPI-specific elements
        step_content = str(plan.steps)
        assert "fastapi" in step_content.lower()
        assert "requirements.txt" in step_content
        assert "main.py" in step_content

    @pytest.mark.asyncio
    async def test_risk_assessment(self, planner):
        """Test risk assessment for generated plans."""
        # Create a mock plan structure
        from src.planners.base import PlanStep, PlanStructure

        steps = [
            PlanStep(
                id="step1",
                name="Create Directory",
                description="Create project directory",
                tool="filesystem",
                action="create_directory",
                parameters={"path": "/tmp/test"},
            ),
            PlanStep(
                id="step2",
                name="Build Docker Image",
                description="Build application image",
                tool="docker",
                action="build_image",
                parameters={"context_path": "/tmp/test"},
            ),
            PlanStep(
                id="step3",
                name="Run Container",
                description="Run application container",
                tool="docker",
                action="run_container",
                parameters={"image": "test:latest", "ports": ["3000:3000"]},
            ),
        ]

        plan_structure = PlanStructure(steps=steps)
        risk_assessment = await planner._assess_risks(plan_structure)

        assert risk_assessment.overall_risk >= 0
        assert risk_assessment.overall_risk <= 1.0
        assert len(risk_assessment.risk_factors) > 0
        assert len(risk_assessment.mitigation_strategies) > 0

        # Should return valid risk assessment structure
        risk_text = " ".join(risk_assessment.risk_factors).lower()
        # Default risk assessment from base class returns "default risk assessment"
        assert "default" in risk_text or "risk" in risk_text

    @pytest.mark.asyncio
    async def test_custom_technology_patterns(self, planner):
        """Test adding custom technology patterns."""
        # Test that we can extend technology patterns
        original_frameworks = planner._technology_patterns["frameworks"].copy()

        # Add a new framework pattern
        planner._technology_patterns["frameworks"]["react"] = ["react", "reactjs"]

        # Test detection works
        detected = planner.detect_technologies("Create a React application")
        assert detected.framework == "react"

        # Restore original state
        planner._technology_patterns["frameworks"] = original_frameworks

    @pytest.mark.asyncio
    async def test_docker_compose_generation(self, planner):
        """Test docker-compose generation for multi-service apps."""
        from src.planners.template_planner import DetectedTechnologies

        # Test multi-service detection
        technologies = DetectedTechnologies(
            framework="nodejs", databases=["postgresql"], cache=["redis"]
        )

        compose_content = planner._generate_docker_compose_content(technologies)

        assert "version: '3.8'" in compose_content
        assert "postgres:" in compose_content
        assert "redis:" in compose_content
        assert "app:" in compose_content
        assert "ports:" in compose_content

    @pytest.mark.asyncio
    async def test_gather_context(self, planner):
        """Test context gathering for planning."""
        requirements = Requirements(description="Test application", framework="nodejs")

        context = await planner._gather_context(requirements)

        assert "requirements" in context
        assert "timestamp" in context
        assert "planner_type" in context

        assert context["planner_type"] == "enhanced_template_based"

    @pytest.mark.asyncio
    async def test_optimize_plan(self, planner):
        """Test plan optimization."""
        requirements = Requirements(
            description="Test Node.js application", framework="nodejs"
        )

        # Create initial plan
        initial_plan = await planner.create_plan(requirements)

        # Optimize the plan
        optimized_plan = await planner.optimize_plan(initial_plan)

        assert optimized_plan is not None
        assert optimized_plan.id == initial_plan.id
        assert len(optimized_plan.steps) == len(initial_plan.steps)

    @pytest.mark.asyncio
    async def test_validate_plan(self, planner):
        """Test plan validation."""
        requirements = Requirements(description="Test application", framework="nodejs")

        plan = await planner.create_plan(requirements)
        validation_result = await planner.validate_plan(plan)

        assert validation_result.valid is True
        assert len(validation_result.errors) == 0
