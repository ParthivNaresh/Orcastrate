"""
Test suite for Anthropic LLM client implementation.

This module tests the Anthropic Claude integration for intelligent planning
including authentication, request handling, and response parsing.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.planners.llm.anthropic_client import AnthropicClient
from src.planners.llm.base import (
    LLMConfig,
    LLMProvider,
    LLMResponse,
    PlanGenerationRequest,
    PlanningPrompt,
    PlanOptimizationRequest,
    RequirementsAnalysis,
)


class TestAnthropicClient:
    """Test Anthropic client functionality."""

    @pytest.fixture
    def anthropic_config(self):
        """Create Anthropic configuration for testing."""
        return LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-sonnet-20240229",
            api_key="test-api-key",
            temperature=0.7,
            max_tokens=4000,
        )

    @pytest.fixture
    def mock_anthropic_response(self):
        """Create mock Anthropic API response."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response content")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=200)
        mock_response.model = "claude-3-sonnet-20240229"
        return mock_response

    @pytest.fixture
    def anthropic_client(self, anthropic_config):
        """Create Anthropic client for testing."""
        with patch("src.planners.llm.anthropic_client.ANTHROPIC_AVAILABLE", True):
            with patch("src.planners.llm.anthropic_client.anthropic"):
                return AnthropicClient(anthropic_config)

    @pytest.mark.asyncio
    async def test_client_initialization(self, anthropic_client, anthropic_config):
        """Test client initialization with valid config."""
        assert anthropic_client.config == anthropic_config
        assert anthropic_client.model == "claude-3-sonnet-20240229"
        assert anthropic_client.temperature == 0.7
        assert anthropic_client.max_tokens == 4000

    @pytest.mark.asyncio
    async def test_client_initialization_without_anthropic(self, anthropic_config):
        """Test client initialization when Anthropic is not available."""
        with patch("src.planners.llm.anthropic_client.ANTHROPIC_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="Anthropic library not available"):
                AnthropicClient(anthropic_config)

    @pytest.mark.asyncio
    async def test_initialize_client(self, anthropic_client):
        """Test client initialization process."""
        with patch.object(
            anthropic_client, "_validate_api_key", new_callable=AsyncMock
        ):
            await anthropic_client.initialize()
            assert anthropic_client._initialized is True

    @pytest.mark.asyncio
    async def test_validate_api_key_success(
        self, anthropic_client, mock_anthropic_response
    ):
        """Test successful API key validation."""
        with patch.object(
            anthropic_client._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_anthropic_response,
        ):
            result = await anthropic_client._validate_api_key()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_api_key_failure(self, anthropic_client):
        """Test API key validation failure."""
        with patch.object(
            anthropic_client._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=Exception("Invalid API key"),
        ):
            result = await anthropic_client._validate_api_key()
            assert result is False

    @pytest.mark.asyncio
    async def test_generate_text_success(
        self, anthropic_client, mock_anthropic_response
    ):
        """Test successful text generation."""
        prompt = PlanningPrompt(
            system_prompt="You are a helpful assistant",
            user_prompt="Create a plan",
            context={"project_type": "web_app"},
        )

        with patch.object(
            anthropic_client._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_anthropic_response,
        ):
            response = await anthropic_client.generate_text(prompt)

            assert isinstance(response, LLMResponse)
            assert response.content == "Test response content"
            assert response.model == "claude-3-sonnet-20240229"
            assert response.tokens_used == 300  # input + output tokens

    @pytest.mark.asyncio
    async def test_generate_text_with_retry(
        self, anthropic_client, mock_anthropic_response
    ):
        """Test text generation with retry logic."""
        prompt = PlanningPrompt(
            system_prompt="You are a helpful assistant",
            user_prompt="Create a plan",
        )

        # First call fails, second succeeds
        with patch.object(
            anthropic_client._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=[Exception("Rate limit"), mock_anthropic_response],
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                response = await anthropic_client.generate_text(prompt)
                assert response.content == "Test response content"

    @pytest.mark.asyncio
    async def test_analyze_requirements(self, anthropic_client):
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
            anthropic_client,
            "_make_requirements_analysis_request",
            new_callable=AsyncMock,
            return_value=mock_analysis,
        ):
            analysis = await anthropic_client.analyze_requirements(requirements)

            assert isinstance(analysis, RequirementsAnalysis)
            assert analysis.technologies == ["React", "Node.js", "PostgreSQL"]
            assert analysis.architecture_pattern == "MVC"
            assert analysis.complexity_score == 6.5
            assert analysis.estimated_duration == 120.0
            assert analysis.confidence_score == 0.85

    @pytest.mark.asyncio
    async def test_generate_plan(self, anthropic_client):
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
            anthropic_client,
            "_make_plan_generation_request",
            new_callable=AsyncMock,
            return_value=mock_plan,
        ):
            plan = await anthropic_client.generate_plan(request)

            assert "steps" in plan
            assert len(plan["steps"]) == 1
            assert plan["steps"][0]["name"] == "Setup Database"

    @pytest.mark.asyncio
    async def test_optimize_plan(self, anthropic_client):
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
            anthropic_client,
            "_make_plan_optimization_request",
            new_callable=AsyncMock,
            return_value=mock_optimized_plan,
        ):
            optimized_plan = await anthropic_client.optimize_plan(request)

            assert "steps" in optimized_plan
            assert optimized_plan["steps"][0]["estimated_cost"] == 70.0

    @pytest.mark.asyncio
    async def test_validate_plan(self, anthropic_client):
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
            anthropic_client,
            "_make_plan_validation_request",
            new_callable=AsyncMock,
            return_value=mock_validation,
        ):
            validation_result = await anthropic_client.validate_plan(plan, requirements)

            assert validation_result["is_valid"] is False
            assert "Invalid dependency" in validation_result["issues"][0]

    @pytest.mark.asyncio
    async def test_explain_plan(self, anthropic_client):
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
            anthropic_client,
            "_make_plan_explanation_request",
            new_callable=AsyncMock,
            return_value=mock_explanation,
        ):
            explanation = await anthropic_client.explain_plan(plan)
            assert "PostgreSQL database" in explanation
            assert "$65" in explanation

    @pytest.mark.asyncio
    async def test_anthropic_message_format(self, anthropic_client):
        """Test Anthropic-specific message formatting."""
        system_prompt = "You are a planning assistant"
        messages = [{"role": "user", "content": "Create a plan"}]

        # Test that the message format is correct for Anthropic
        formatted_request = anthropic_client._format_request(system_prompt, messages)

        assert "system" in formatted_request
        assert "messages" in formatted_request
        assert formatted_request["system"] == system_prompt
        assert formatted_request["messages"] == messages

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, anthropic_client):
        """Test rate limit handling and backoff."""
        # Test that rate limiting is checked before requests
        await anthropic_client._rate_limit_check()

        # Simulate rate limit hit
        anthropic_client._last_request_time = anthropic_client._get_current_time()
        anthropic_client._request_count = 50  # Hit rate limit

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await anthropic_client._rate_limit_check()
            mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_json_response_success(self, anthropic_client):
        """Test successful JSON response parsing."""
        json_text = '{"technologies": ["React", "Node.js"], "complexity": 5}'
        result = anthropic_client._parse_json_response(json_text)

        assert result["technologies"] == ["React", "Node.js"]
        assert result["complexity"] == 5

    def test_parse_json_response_invalid(self, anthropic_client):
        """Test handling of invalid JSON responses."""
        invalid_json = "This is not valid JSON"
        result = anthropic_client._parse_json_response(invalid_json)

        assert result == {}

    def test_parse_json_response_with_claude_tags(self, anthropic_client):
        """Test parsing JSON with Claude-specific XML tags."""
        claude_response = """
        <analysis>
        {"technologies": ["Python", "Flask"], "complexity": 4}
        </analysis>
        """
        result = anthropic_client._parse_json_response(claude_response)

        assert result["technologies"] == ["Python", "Flask"]
        assert result["complexity"] == 4

    @pytest.mark.asyncio
    async def test_error_handling_max_retries(self, anthropic_client):
        """Test error handling when max retries are exceeded."""
        prompt = PlanningPrompt(system_prompt="Test", user_prompt="Test")

        with patch.object(
            anthropic_client._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=Exception("Persistent error"),
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(Exception, match="Persistent error"):
                    await anthropic_client.generate_text(prompt)

    @pytest.mark.asyncio
    async def test_context_length_handling(self, anthropic_client):
        """Test handling of context length limits."""
        # Create a very long prompt that exceeds token limits
        long_prompt = PlanningPrompt(
            system_prompt="Test system prompt",
            user_prompt="A" * 10000,  # Very long prompt
        )

        with patch.object(
            anthropic_client, "_estimate_tokens", return_value=200000
        ):  # Exceeds Claude's limit
            truncated_prompt = anthropic_client._truncate_prompt_if_needed(long_prompt)
            assert len(truncated_prompt.user_prompt) < len(long_prompt.user_prompt)

    def test_estimate_tokens(self, anthropic_client):
        """Test token estimation functionality."""
        text = "This is a test prompt for token estimation"
        estimated_tokens = anthropic_client._estimate_tokens(text)

        assert isinstance(estimated_tokens, int)
        assert estimated_tokens > 0
        # Rough estimation: should be less than word count but more than word count / 2
        word_count = len(text.split())
        assert word_count // 2 <= estimated_tokens <= word_count * 2

    def test_extract_json_from_xml_tags(self, anthropic_client):
        """Test extraction of JSON from Claude's XML-style tags."""
        xml_response = """
        Here is my analysis:
        <json>
        {"technologies": ["Vue.js", "Django"], "architecture": "SPA"}
        </json>
        This completes the analysis.
        """

        json_content = anthropic_client._extract_json_from_xml(xml_response)
        assert (
            '{"technologies": ["Vue.js", "Django"], "architecture": "SPA"}'
            in json_content
        )

    def test_model_version_handling(self, anthropic_client):
        """Test handling of different Claude model versions."""
        # Test that the client can handle different model versions
        assert anthropic_client.model == "claude-3-sonnet-20240229"

        # Test model capabilities detection
        capabilities = anthropic_client._get_model_capabilities()
        assert "max_tokens" in capabilities
        assert "context_window" in capabilities

    @pytest.mark.asyncio
    async def test_streaming_response_handling(self, anthropic_client):
        """Test handling of streaming responses if supported."""
        # Note: This test is placeholder for future streaming support
        prompt = PlanningPrompt(system_prompt="Test", user_prompt="Generate a plan")

        # For now, just ensure non-streaming works
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Streaming response")]
        mock_response.usage = MagicMock(input_tokens=50, output_tokens=100)
        mock_response.model = "claude-3-sonnet-20240229"

        with patch.object(
            anthropic_client._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = await anthropic_client.generate_text(prompt)
            assert response.content == "Streaming response"

    @pytest.mark.asyncio
    async def test_anthropic_specific_error_handling(self, anthropic_client):
        """Test handling of Anthropic-specific errors."""
        prompt = PlanningPrompt(system_prompt="Test", user_prompt="Test")

        # Test handling of Anthropic rate limit error
        anthropic_error = Exception("rate_limit_error")
        anthropic_error.error = MagicMock()
        anthropic_error.error.type = "rate_limit_error"

        with patch.object(
            anthropic_client._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=anthropic_error,
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                # Should retry on rate limit
                with pytest.raises(Exception):
                    await anthropic_client.generate_text(prompt)

    @pytest.mark.asyncio
    async def test_cleanup_and_shutdown(self, anthropic_client):
        """Test client cleanup and shutdown procedures."""
        await anthropic_client.initialize()

        # Test that cleanup doesn't raise errors
        try:
            # Anthropic client doesn't require explicit cleanup, but test the method exists
            if hasattr(anthropic_client, "cleanup"):
                await anthropic_client.cleanup()
        except Exception as e:
            pytest.fail(f"Cleanup should not raise exceptions: {e}")
