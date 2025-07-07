"""
Test suite for Prompt Template Manager.

This module tests the prompt template system that provides consistent
LLM interactions with 10 specialized prompt types and configurations.
"""

import pytest

from src.planners.llm.prompt_templates import (
    PlanningPrompt,
    PromptTemplateManager,
    PromptType,
    TemplateConfig,
)


class TestPromptTemplateManager:
    """Test PromptTemplateManager functionality."""

    @pytest.fixture
    def manager(self):
        """Create prompt template manager instance."""
        return PromptTemplateManager()

    @pytest.fixture
    def sample_context(self):
        """Sample context data for prompt generation."""
        return {
            "requirements": "Build a web application with user authentication",
            "technologies": ["React", "Node.js", "PostgreSQL"],
            "budget": {"max_monthly": 500},
            "timeline": "3 months",
            "team_size": 3,
        }

    def test_manager_initialization(self, manager):
        """Test prompt template manager initialization."""
        # Should initialize with all prompt types
        assert len(manager._templates) > 0

        # Should have all required prompt types
        required_types = [
            PromptType.REQUIREMENTS_ANALYSIS,
            PromptType.PLAN_GENERATION,
            PromptType.PLAN_OPTIMIZATION,
            PromptType.PLAN_VALIDATION,
            PromptType.PLAN_EXPLANATION,
            PromptType.TECHNOLOGY_RECOMMENDATION,
            PromptType.COST_ESTIMATION,
            PromptType.RISK_ASSESSMENT,
            PromptType.ARCHITECTURE_REVIEW,
            PromptType.GENERAL_PLANNING,
        ]

        for prompt_type in required_types:
            assert prompt_type in manager._templates

    def test_get_requirements_analysis_prompt(self, manager, sample_context):
        """Test requirements analysis prompt generation."""
        prompt = manager.get_prompt(PromptType.REQUIREMENTS_ANALYSIS, sample_context)

        assert isinstance(prompt, PlanningPrompt)
        assert prompt.system_prompt is not None
        assert prompt.user_prompt is not None

        # Should include context in prompts
        assert "web application" in prompt.user_prompt
        assert "user authentication" in prompt.user_prompt

    def test_get_plan_generation_prompt(self, manager, sample_context):
        """Test plan generation prompt generation."""
        prompt = manager.get_prompt(PromptType.PLAN_GENERATION, sample_context)

        assert isinstance(prompt, PlanningPrompt)
        assert "plan" in prompt.system_prompt.lower()
        assert "React" in prompt.user_prompt
        assert "Node.js" in prompt.user_prompt

    def test_get_plan_optimization_prompt(self, manager, sample_context):
        """Test plan optimization prompt generation."""
        optimization_context = {
            **sample_context,
            "original_plan": {
                "steps": [
                    {"name": "Setup Database", "cost": 100},
                    {"name": "Deploy App", "cost": 200},
                ]
            },
            "optimization_goals": ["cost", "performance"],
        }

        prompt = manager.get_prompt(PromptType.PLAN_OPTIMIZATION, optimization_context)

        assert isinstance(prompt, PlanningPrompt)
        assert "optimization" in prompt.system_prompt.lower()
        assert "cost" in prompt.user_prompt

    def test_get_plan_validation_prompt(self, manager, sample_context):
        """Test plan validation prompt generation."""
        validation_context = {
            **sample_context,
            "plan": {
                "steps": [
                    {"id": "step1", "name": "Setup", "dependencies": []},
                    {"id": "step2", "name": "Deploy", "dependencies": ["step1"]},
                ]
            },
        }

        prompt = manager.get_prompt(PromptType.PLAN_VALIDATION, validation_context)

        assert isinstance(prompt, PlanningPrompt)
        assert "validation" in prompt.system_prompt.lower()
        assert "step1" in prompt.user_prompt

    def test_get_plan_explanation_prompt(self, manager, sample_context):
        """Test plan explanation prompt generation."""
        explanation_context = {
            **sample_context,
            "plan": {
                "steps": [{"name": "Create Database"}, {"name": "Deploy Application"}],
                "total_cost": 300,
                "total_duration": 1800,
            },
        }

        prompt = manager.get_prompt(PromptType.PLAN_EXPLANATION, explanation_context)

        assert isinstance(prompt, PlanningPrompt)
        assert "explain" in prompt.system_prompt.lower()
        assert "300" in prompt.user_prompt  # Cost should be included

    def test_get_technology_recommendation_prompt(self, manager, sample_context):
        """Test technology recommendation prompt generation."""
        prompt = manager.get_prompt(
            PromptType.TECHNOLOGY_RECOMMENDATION, sample_context
        )

        assert isinstance(prompt, PlanningPrompt)
        assert "technology" in prompt.system_prompt.lower()
        assert "React" in prompt.user_prompt

    def test_get_cost_estimation_prompt(self, manager, sample_context):
        """Test cost estimation prompt generation."""
        cost_context = {
            **sample_context,
            "plan_steps": [
                {"name": "AWS Setup", "tool": "aws"},
                {"name": "Database", "tool": "postgresql"},
            ],
        }

        prompt = manager.get_prompt(PromptType.COST_ESTIMATION, cost_context)

        assert isinstance(prompt, PlanningPrompt)
        assert "cost" in prompt.system_prompt.lower()
        assert "AWS" in prompt.user_prompt

    def test_get_risk_assessment_prompt(self, manager, sample_context):
        """Test risk assessment prompt generation."""
        risk_context = {
            **sample_context,
            "complexity": 7.5,
            "identified_patterns": ["microservices", "distributed_system"],
        }

        prompt = manager.get_prompt(PromptType.RISK_ASSESSMENT, risk_context)

        assert isinstance(prompt, PlanningPrompt)
        assert "risk" in prompt.system_prompt.lower()
        assert "7.5" in prompt.user_prompt

    def test_get_architecture_review_prompt(self, manager, sample_context):
        """Test architecture review prompt generation."""
        arch_context = {
            **sample_context,
            "architecture_pattern": "three_tier",
            "technology_stack": {
                "frontend": ["React"],
                "backend": ["Node.js"],
                "database": ["PostgreSQL"],
            },
        }

        prompt = manager.get_prompt(PromptType.ARCHITECTURE_REVIEW, arch_context)

        assert isinstance(prompt, PlanningPrompt)
        assert "architecture" in prompt.system_prompt.lower()
        assert "three_tier" in prompt.user_prompt

    def test_get_general_planning_prompt(self, manager, sample_context):
        """Test general planning prompt generation."""
        prompt = manager.get_prompt(PromptType.GENERAL_PLANNING, sample_context)

        assert isinstance(prompt, PlanningPrompt)
        assert "planning" in prompt.system_prompt.lower()
        assert "web application" in prompt.user_prompt

    def test_template_config_default(self, manager, sample_context):
        """Test prompt generation with default template configuration."""
        prompt = manager.get_prompt(PromptType.REQUIREMENTS_ANALYSIS, sample_context)

        # Default config should work
        assert prompt.system_prompt is not None
        assert prompt.user_prompt is not None

    def test_template_config_with_examples(self, manager, sample_context):
        """Test prompt generation with examples enabled."""
        config = TemplateConfig(
            include_examples=True, include_constraints=False, temperature=0.5
        )

        prompt = manager.get_prompt(
            PromptType.REQUIREMENTS_ANALYSIS, sample_context, config
        )

        # Should include examples when enabled
        assert (
            "example" in prompt.user_prompt.lower()
            or "example" in prompt.system_prompt.lower()
        )

    def test_template_config_with_constraints(self, manager, sample_context):
        """Test prompt generation with constraints enabled."""
        config = TemplateConfig(
            include_examples=False, include_constraints=True, temperature=0.3
        )

        prompt = manager.get_prompt(PromptType.PLAN_GENERATION, sample_context, config)

        # Should include constraint information
        assert (
            "constraint" in prompt.user_prompt.lower()
            or "constraint" in prompt.system_prompt.lower()
        )

    def test_template_config_comprehensive(self, manager, sample_context):
        """Test prompt generation with comprehensive configuration."""
        config = TemplateConfig(
            include_examples=True,
            include_constraints=True,
            include_context=True,
            output_format="json",
            temperature=0.7,
            max_tokens=3000,
        )

        prompt = manager.get_prompt(
            PromptType.PLAN_OPTIMIZATION, sample_context, config
        )

        # Should include JSON format instruction
        assert "json" in prompt.user_prompt.lower()

    def test_context_variable_substitution(self, manager):
        """Test proper context variable substitution in templates."""
        context = {
            "requirements": "Build mobile app",
            "budget": {"max_monthly": 1000},
            "technologies": ["React Native", "Firebase"],
            "custom_field": "test_value",
        }

        prompt = manager.get_prompt(PromptType.REQUIREMENTS_ANALYSIS, context)

        # Should substitute context variables properly
        assert "Build mobile app" in prompt.user_prompt
        assert "React Native" in prompt.user_prompt

    def test_missing_context_variables(self, manager):
        """Test handling of missing context variables."""
        minimal_context = {"requirements": "Simple app"}

        # Should not raise error with minimal context
        prompt = manager.get_prompt(PromptType.PLAN_GENERATION, minimal_context)

        assert isinstance(prompt, PlanningPrompt)
        assert "Simple app" in prompt.user_prompt

    def test_empty_context(self, manager):
        """Test handling of empty context."""
        empty_context = {}

        # Should handle empty context gracefully
        prompt = manager.get_prompt(PromptType.GENERAL_PLANNING, empty_context)

        assert isinstance(prompt, PlanningPrompt)
        assert prompt.system_prompt is not None
        assert prompt.user_prompt is not None

    def test_special_characters_in_context(self, manager):
        """Test handling of special characters in context."""
        special_context = {
            "requirements": "App with $pecial ch@rs & symbols",
            "technologies": ["Node.js", "C++", "C#"],
            "budget": {"max_monthly": 1000},
        }

        prompt = manager.get_prompt(PromptType.REQUIREMENTS_ANALYSIS, special_context)

        # Should handle special characters without errors
        assert "$pecial ch@rs" in prompt.user_prompt
        assert "C++" in prompt.user_prompt

    def test_large_context_data(self, manager):
        """Test handling of large context data."""
        large_context = {
            "requirements": "Build application with many features " * 100,
            "technologies": [f"Tech{i}" for i in range(50)],
            "detailed_specs": {f"spec_{i}": f"value_{i}" for i in range(100)},
        }

        prompt = manager.get_prompt(PromptType.PLAN_GENERATION, large_context)

        # Should handle large context without errors
        assert isinstance(prompt, PlanningPrompt)
        assert prompt.user_prompt is not None

    def test_nested_context_objects(self, manager):
        """Test handling of nested context objects."""
        nested_context = {
            "requirements": "Complex system",
            "project": {
                "details": {
                    "budget": {"monthly": 500, "yearly": 6000},
                    "team": {"size": 5, "skills": ["Python", "React"]},
                    "timeline": {"phases": ["design", "development", "testing"]},
                }
            },
        }

        prompt = manager.get_prompt(PromptType.ARCHITECTURE_REVIEW, nested_context)

        # Should handle nested objects
        assert isinstance(prompt, PlanningPrompt)

    def test_prompt_consistency(self, manager, sample_context):
        """Test consistency of generated prompts."""
        # Generate same prompt multiple times
        prompt1 = manager.get_prompt(PromptType.REQUIREMENTS_ANALYSIS, sample_context)
        prompt2 = manager.get_prompt(PromptType.REQUIREMENTS_ANALYSIS, sample_context)

        # Should generate consistent prompts
        assert prompt1.system_prompt == prompt2.system_prompt
        assert prompt1.user_prompt == prompt2.user_prompt

    def test_different_prompt_types_unique(self, manager, sample_context):
        """Test that different prompt types generate unique content."""
        analysis_prompt = manager.get_prompt(
            PromptType.REQUIREMENTS_ANALYSIS, sample_context
        )
        generation_prompt = manager.get_prompt(
            PromptType.PLAN_GENERATION, sample_context
        )

        # Different prompt types should have different system prompts
        assert analysis_prompt.system_prompt != generation_prompt.system_prompt

    def test_prompt_length_reasonable(self, manager, sample_context):
        """Test that generated prompts have reasonable length."""
        prompt = manager.get_prompt(PromptType.REQUIREMENTS_ANALYSIS, sample_context)

        # Prompts should not be too short or too long
        assert len(prompt.system_prompt) > 50  # Not too short
        assert len(prompt.system_prompt) < 5000  # Not too long
        assert len(prompt.user_prompt) > 10
        assert len(prompt.user_prompt) < 10000

    def test_prompt_structure_valid(self, manager, sample_context):
        """Test that generated prompts have valid structure."""
        prompt = manager.get_prompt(PromptType.PLAN_GENERATION, sample_context)

        # Should have proper prompt structure
        assert isinstance(prompt.system_prompt, str)
        assert isinstance(prompt.user_prompt, str)
        assert len(prompt.system_prompt.strip()) > 0
        assert len(prompt.user_prompt.strip()) > 0

    def test_error_handling_invalid_prompt_type(self, manager, sample_context):
        """Test error handling for invalid prompt type."""
        # Should handle invalid prompt type gracefully
        try:
            invalid_type = "INVALID_PROMPT_TYPE"
            prompt = manager.get_prompt(invalid_type, sample_context)
            # If it doesn't raise an error, should return something reasonable
            assert prompt is not None
        except (KeyError, ValueError, AttributeError):
            # Expected to raise an error for invalid type
            pass

    def test_template_customization(self, manager):
        """Test template customization capabilities."""
        # Test that manager supports template customization
        assert hasattr(manager, "_templates")
        assert isinstance(manager._templates, dict)

        # Should have proper template structure
        for prompt_type, template in manager._templates.items():
            assert isinstance(template, dict)
            assert "system" in template
            assert "user" in template
