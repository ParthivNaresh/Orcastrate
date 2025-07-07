"""
Test suite for OpenAI LLM client implementation.

This module tests the OpenAI integration for intelligent planning
including authentication, request handling, and response parsing.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.planners.llm.base import (
    LLMConfig,
    LLMProvider,
    LLMResponse,
    PlanGenerationRequest,
    PlanningPrompt,
    PlanOptimizationRequest,
    RequirementsAnalysis,
)
from src.planners.llm.openai_client import OpenAIClient


class TestOpenAIClient:
    """Test OpenAI client functionality."""

    @pytest.fixture
    def openai_config(self):
        """Create OpenAI configuration for testing."""
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-api-key",
            temperature=0.7,
            max_tokens=4000,
        )

    @pytest.fixture
    def mock_openai_response(self):
        """Create mock OpenAI API response."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Test response content"))
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=100, completion_tokens=200, total_tokens=300
        )
        return mock_response

    @pytest.fixture
    def openai_client(self, openai_config):
        """Create OpenAI client for testing."""
        with patch("src.planners.llm.openai_client.OPENAI_AVAILABLE", True):
            with patch("src.planners.llm.openai_client.openai"):
                return OpenAIClient(openai_config)

    @pytest.mark.asyncio
    async def test_client_initialization(self, openai_client, openai_config):
        """Test client initialization with valid config."""
        assert openai_client.config == openai_config
        assert openai_client.model == "gpt-4"
        assert openai_client.temperature == 0.7
        assert openai_client.max_tokens == 4000

    @pytest.mark.asyncio
    async def test_client_initialization_without_openai(self, openai_config):
        """Test client initialization when OpenAI is not available."""
        with patch("src.planners.llm.openai_client.OPENAI_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="OpenAI library not available"):
                OpenAIClient(openai_config)

    @pytest.mark.asyncio
    async def test_initialize_client(self, openai_client):
        """Test client initialization process."""
        with patch.object(openai_client, "_validate_api_key", new_callable=AsyncMock):
            await openai_client.initialize()
            assert openai_client._initialized is True

    @pytest.mark.asyncio
    async def test_validate_api_key_success(self, openai_client, mock_openai_response):
        """Test successful API key validation."""
        with patch.object(
            openai_client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response,
        ):
            result = await openai_client._validate_api_key()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_api_key_failure(self, openai_client):
        """Test API key validation failure."""
        with patch.object(
            openai_client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=Exception("Invalid API key"),
        ):
            result = await openai_client._validate_api_key()
            assert result is False

    @pytest.mark.asyncio
    async def test_generate_text_success(self, openai_client, mock_openai_response):
        """Test successful text generation."""
        prompt = PlanningPrompt(
            system_prompt="You are a helpful assistant",
            user_prompt="Create a plan",
            context={"project_type": "web_app"},
        )

        with patch.object(
            openai_client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response,
        ):
            response = await openai_client.generate_text(prompt)

            assert isinstance(response, LLMResponse)
            assert response.content == "Test response content"
            assert response.model == "gpt-4"
            assert response.tokens_used == 300

    @pytest.mark.asyncio
    async def test_generate_text_with_retry(self, openai_client, mock_openai_response):
        """Test text generation with retry logic."""
        prompt = PlanningPrompt(
            system_prompt="You are a helpful assistant",
            user_prompt="Create a plan",
        )

        # First call fails, second succeeds
        with patch.object(
            openai_client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[Exception("Rate limit"), mock_openai_response],
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                response = await openai_client.generate_text(prompt)
                assert response.content == "Test response content"

    @pytest.mark.asyncio
    async def test_analyze_requirements(self, openai_client):
        """Test requirements analysis functionality."""
        requirements = "Build a web application with user authentication"

        mock_analysis = {
            "technologies": ["React", "Node.js", "PostgreSQL"],
            "architecture_pattern": "MVC",
            "complexity_score": 6.5,
            "estimated_duration": 120.0,
            "confidence_score": 0.85,
        }

        with patch.object(
            openai_client,
            "_make_requirements_analysis_request",
            new_callable=AsyncMock,
            return_value=mock_analysis,
        ):
            analysis = await openai_client.analyze_requirements(requirements)

            assert isinstance(analysis, RequirementsAnalysis)
            assert analysis.technologies == ["React", "Node.js", "PostgreSQL"]
            assert analysis.architecture_pattern == "MVC"
            assert analysis.complexity_score == 6.5
            assert analysis.estimated_duration == 120.0
            assert analysis.confidence_score == 0.85

    @pytest.mark.asyncio
    async def test_generate_plan(self, openai_client):
        """Test plan generation functionality."""
        request = PlanGenerationRequest(
            requirements_description="Build an e-commerce site",
            available_tools=["aws", "docker", "postgresql"],
            budget_constraints={"max_monthly": 500},
        )

        mock_plan = {
            "steps": [
                {
                    "id": "step_1",
                    "name": "Setup Database",
                    "description": "Create PostgreSQL database",
                    "tool": "postgresql",
                    "action": "create_database",
                    "estimated_duration": 600.0,
                    "estimated_cost": 25.0,
                }
            ]
        }

        with patch.object(
            openai_client,
            "_make_plan_generation_request",
            new_callable=AsyncMock,
            return_value=mock_plan,
        ):
            plan = await openai_client.generate_plan(request)

            assert "steps" in plan
            assert len(plan["steps"]) == 1
            assert plan["steps"][0]["name"] == "Setup Database"

    @pytest.mark.asyncio
    async def test_optimize_plan(self, openai_client):
        """Test plan optimization functionality."""
        original_plan = {
            "steps": [
                {
                    "id": "step_1",
                    "name": "Create Infrastructure",
                    "tool": "aws",
                    "estimated_cost": 100.0,
                }
            ]
        }

        request = PlanOptimizationRequest(
            original_plan=original_plan,
            optimization_goals=["cost", "performance"],
            constraints=["budget"],
            available_tools=["aws", "docker"],
        )

        mock_optimized_plan = {
            "steps": [
                {
                    "id": "step_1",
                    "name": "Create Infrastructure",
                    "tool": "aws",
                    "estimated_cost": 70.0,  # Optimized cost
                }
            ]
        }

        with patch.object(
            openai_client,
            "_make_plan_optimization_request",
            new_callable=AsyncMock,
            return_value=mock_optimized_plan,
        ):
            optimized_plan = await openai_client.optimize_plan(request)

            assert "steps" in optimized_plan
            assert optimized_plan["steps"][0]["estimated_cost"] == 70.0

    @pytest.mark.asyncio
    async def test_validate_plan(self, openai_client):
        """Test plan validation functionality."""
        plan = {
            "steps": [
                {
                    "id": "step_1",
                    "name": "Deploy Application",
                    "dependencies": ["step_2"],  # Invalid dependency
                }
            ]
        }

        requirements = "Deploy a simple web application"

        mock_validation = {"is_valid": False, "issues": ["Invalid dependency step_2"]}

        with patch.object(
            openai_client,
            "_make_plan_validation_request",
            new_callable=AsyncMock,
            return_value=mock_validation,
        ):
            validation_result = await openai_client.validate_plan(plan, requirements)

            assert validation_result["is_valid"] is False
            assert "Invalid dependency" in validation_result["issues"][0]

    @pytest.mark.asyncio
    async def test_explain_plan(self, openai_client):
        """Test plan explanation functionality."""
        plan = {
            "steps": [
                {"name": "Setup Database", "description": "Create PostgreSQL instance"},
                {"name": "Deploy App", "description": "Deploy web application"},
            ],
            "total_cost": 65.0,
            "total_duration": 1200.0,
        }

        mock_explanation = (
            "This plan creates a PostgreSQL database and deploys a web application. "
            "The total cost is $65 and will take 20 minutes to complete."
        )

        with patch.object(
            openai_client,
            "_make_plan_explanation_request",
            new_callable=AsyncMock,
            return_value=mock_explanation,
        ):
            explanation = await openai_client.explain_plan(plan)
            assert "PostgreSQL database" in explanation
            assert "$65" in explanation

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, openai_client):
        """Test rate limit handling and backoff."""
        # Test that rate limiting is checked before requests
        await openai_client._rate_limit_check()

        # Simulate rate limit hit
        openai_client._last_request_time = openai_client._get_current_time()
        openai_client._request_count = 60  # Hit rate limit

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await openai_client._rate_limit_check()
            mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_json_response_success(self, openai_client):
        """Test successful JSON response parsing."""
        json_text = '{"technologies": ["React", "Node.js"], "complexity": 5}'
        result = openai_client._parse_json_response(json_text)

        assert result["technologies"] == ["React", "Node.js"]
        assert result["complexity"] == 5

    def test_parse_json_response_invalid(self, openai_client):
        """Test handling of invalid JSON responses."""
        invalid_json = "This is not valid JSON"
        result = openai_client._parse_json_response(invalid_json)

        assert result == {}

    def test_parse_json_response_partial(self, openai_client):
        """Test parsing of partial JSON with extra text."""
        partial_json = (
            'Here is the analysis: {"technologies": ["Python"]} and more text'
        )
        result = openai_client._parse_json_response(partial_json)

        assert result["technologies"] == ["Python"]

    @pytest.mark.asyncio
    async def test_error_handling_max_retries(self, openai_client):
        """Test error handling when max retries are exceeded."""
        prompt = PlanningPrompt(system_prompt="Test", user_prompt="Test")

        with patch.object(
            openai_client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=Exception("Persistent error"),
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(Exception, match="Persistent error"):
                    await openai_client.generate_text(prompt)

    @pytest.mark.asyncio
    async def test_context_length_handling(self, openai_client):
        """Test handling of context length limits."""
        # Create a very long prompt that exceeds token limits
        long_prompt = PlanningPrompt(
            system_prompt="Test system prompt",
            user_prompt="A" * 10000,  # Very long prompt
        )

        with patch.object(
            openai_client, "_estimate_tokens", return_value=5000
        ):  # Exceeds limit
            truncated_prompt = openai_client._truncate_prompt_if_needed(long_prompt)
            assert len(truncated_prompt.user_prompt) < len(long_prompt.user_prompt)

    def test_estimate_tokens(self, openai_client):
        """Test token estimation functionality."""
        text = "This is a test prompt for token estimation"
        estimated_tokens = openai_client._estimate_tokens(text)

        assert isinstance(estimated_tokens, int)
        assert estimated_tokens > 0
        # Rough estimation: should be less than word count but more than word count / 2
        word_count = len(text.split())
        assert word_count // 2 <= estimated_tokens <= word_count * 2

    @pytest.mark.asyncio
    async def test_cleanup_and_shutdown(self, openai_client):
        """Test client cleanup and shutdown procedures."""
        await openai_client.initialize()

        # Test that cleanup doesn't raise errors
        try:
            # OpenAI client doesn't require explicit cleanup, but test the method exists
            if hasattr(openai_client, "cleanup"):
                await openai_client.cleanup()
        except Exception as e:
            pytest.fail(f"Cleanup should not raise exceptions: {e}")
