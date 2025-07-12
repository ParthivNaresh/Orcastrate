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
        assert len(planner._templates) > 0
        assert "nodejs_web_app" in planner._templates
        assert "python_fastapi" in planner._templates

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
        assert "Python FastAPI Application" in template_names

    @pytest.mark.asyncio
    async def test_template_selection_nodejs(self, planner):
        """Test template selection for Node.js requirements."""
        requirements = Requirements(
            description="Create a Node.js web application", framework="nodejs"
        )

        template = await planner._select_template(requirements)

        assert template is not None
        assert template["framework"] == "nodejs"
        assert "nodejs_web_app" in planner._templates

    @pytest.mark.asyncio
    async def test_template_selection_fastapi(self, planner):
        """Test template selection for FastAPI requirements."""
        requirements = Requirements(
            description="Build a FastAPI REST API", framework="fastapi"
        )

        template = await planner._select_template(requirements)

        assert template is not None
        assert template["framework"] == "fastapi"

    @pytest.mark.asyncio
    async def test_template_selection_by_description(self, planner):
        """Test template selection based on description keywords."""
        # Test Node.js selection by description
        requirements = Requirements(
            description="Express web application with JavaScript", framework=None
        )

        template = await planner._select_template(requirements)
        assert template is not None
        assert template["framework"] == "nodejs"

        # Test FastAPI selection by description
        requirements = Requirements(
            description="Python API service with FastAPI", framework=None
        )

        template = await planner._select_template(requirements)
        assert template is not None
        assert template["framework"] == "fastapi"

    @pytest.mark.asyncio
    async def test_template_selection_fallback(self, planner):
        """Test template fallback for generic web applications."""
        requirements = Requirements(description="A web application", framework=None)

        template = await planner._select_template(requirements)

        assert template is not None
        # Should fallback to default (Node.js web app)

    @pytest.mark.asyncio
    async def test_template_selection_no_match(self, planner):
        """Test template selection when no template matches."""
        requirements = Requirements(
            description="Desktop application with C++", framework="cpp"
        )

        template = await planner._select_template(requirements)
        assert template is None

    @pytest.mark.asyncio
    async def test_variable_extraction(self, planner):
        """Test variable extraction from requirements."""
        requirements = Requirements(
            description="My awesome Node.js web application", framework="nodejs"
        )

        variables = planner._extract_variables(requirements)

        assert "project_name" in variables
        assert "project_path" in variables
        assert "description" in variables
        assert "framework" in variables

        assert variables["framework"] == "nodejs"
        assert variables["description"] == requirements.description
        assert variables["project_name"]  # Should be non-empty
        assert "/tmp/orcastrate" in variables["project_path"]

    @pytest.mark.asyncio
    async def test_project_name_generation(self, planner):
        """Test project name generation from descriptions."""
        test_cases = [
            ("Node.js web application", "node-js-web"),
            ("FastAPI REST API service", "fastapi-rest-api"),
            ("My awesome app", "my-awesome-app"),
            ("Simple web app with database", "simple-web-app"),
            ("A", "a"),  # Single word
            ("", "app"),  # Empty should fallback
        ]

        for description, expected_pattern in test_cases:
            name = planner._generate_project_name(description)
            assert name
            if expected_pattern == "app":
                assert name == "app"
            else:
                assert expected_pattern.split("-")[0] in name or name.startswith(
                    expected_pattern.split("-")[0]
                )

    @pytest.mark.asyncio
    async def test_variable_replacement(self, planner):
        """Test variable replacement in templates."""
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

        result = planner._replace_variables(template_data, variables)

        assert result["name"] == "my-app"
        assert result["path"] == "/tmp/my-app/src"
        assert result["nested"]["description"] == "This is my-app with nodejs"
        assert result["nested"]["array"] == ["my-app-item1", "my-app-item2"]

    @pytest.mark.asyncio
    async def test_generate_steps_from_template(self, planner):
        """Test generating plan steps from template."""
        requirements = Requirements(
            description="Test Node.js application", framework="nodejs"
        )

        template = planner._templates["nodejs_web_app"]
        steps = await planner._generate_steps_from_template(template, requirements)

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
        assert "build_docker_image" in step_ids

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

        # Should identify Docker and port risks
        risk_text = " ".join(risk_assessment.risk_factors).lower()
        assert "docker" in risk_text or "port" in risk_text

    @pytest.mark.asyncio
    async def test_add_custom_template(self, planner):
        """Test adding custom templates."""
        custom_template = {
            "name": "Custom React App",
            "description": "A custom React application template",
            "framework": "react",
            "steps": [
                {
                    "id": "setup",
                    "name": "Setup React App",
                    "description": "Initialize React application",
                    "tool": "filesystem",
                    "action": "create_directory",
                    "parameters": {"path": "{project_path}"},
                }
            ],
        }

        planner.add_custom_template("custom_react", custom_template)

        assert "custom_react" in planner._templates
        assert planner._templates["custom_react"]["framework"] == "react"

        # Should be available in template list
        templates = await planner.get_available_templates()
        template_names = [t["name"] for t in templates]
        assert "Custom React App" in template_names

    @pytest.mark.asyncio
    async def test_add_custom_template_validation(self, planner):
        """Test validation when adding custom templates."""
        # Missing required fields
        invalid_template = {
            "name": "Invalid Template",
            # Missing description and steps
        }

        with pytest.raises(Exception):  # Should raise PlannerError
            planner.add_custom_template("invalid", invalid_template)

    @pytest.mark.asyncio
    async def test_gather_context(self, planner):
        """Test context gathering for planning."""
        requirements = Requirements(description="Test application", framework="nodejs")

        context = await planner._gather_context(requirements)

        assert "requirements" in context
        assert "timestamp" in context
        assert "available_templates" in context
        assert "planner_type" in context

        assert context["planner_type"] == "template_based"
        assert len(context["available_templates"]) > 0

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
