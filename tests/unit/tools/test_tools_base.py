"""
Tests for tool base classes and functionality.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from src.tools.base import (
    CostEstimate,
    Tool,
    ToolConfig,
    ToolError,
    ToolExecutionError,
    ToolResult,
    ToolSchema,
    ToolStatus,
    ToolTimeoutError,
    ToolValidationError,
    ValidationResult,
)


class TestToolConfig:
    """Test ToolConfig model."""

    def test_tool_config_creation(self):
        """Test basic tool configuration creation."""
        config = ToolConfig(name="test_tool", version="1.0.0")

        assert config.name == "test_tool"
        assert config.version == "1.0.0"
        assert config.enabled is True
        assert config.timeout == 300
        assert config.retry_count == 3
        assert config.retry_delay == 5
        assert config.environment == {}
        assert config.credentials == {}

    def test_tool_config_with_custom_values(self):
        """Test tool configuration with custom values."""
        config = ToolConfig(
            name="custom_tool",
            version="2.1.0",
            enabled=False,
            timeout=600,
            retry_count=5,
            retry_delay=10,
            environment={"ENV_VAR": "value"},
            credentials={"api_key": "secret"},
            resource_limits={"memory": "1Gi"},
        )

        assert config.enabled is False
        assert config.timeout == 600
        assert config.retry_count == 5
        assert config.retry_delay == 10
        assert config.environment == {"ENV_VAR": "value"}
        assert config.credentials == {"api_key": "secret"}
        assert config.resource_limits == {"memory": "1Gi"}


class TestToolSchema:
    """Test ToolSchema model."""

    def test_tool_schema_creation(self):
        """Test tool schema creation."""
        schema = ToolSchema(
            name="test_tool",
            description="A test tool",
            version="1.0.0",
            actions={
                "create": {"description": "Create resource"},
                "delete": {"description": "Delete resource"},
            },
            required_permissions=["read", "write"],
            dependencies=["boto3"],
            cost_model={"per_hour": 0.10},
        )

        assert schema.name == "test_tool"
        assert schema.description == "A test tool"
        assert len(schema.actions) == 2
        assert "create" in schema.actions
        assert schema.required_permissions == ["read", "write"]
        assert schema.dependencies == ["boto3"]
        assert schema.cost_model == {"per_hour": 0.10}


class TestToolResult:
    """Test ToolResult model."""

    def test_successful_tool_result(self):
        """Test successful tool result."""
        result = ToolResult(
            success=True,
            tool_name="test_tool",
            action="create_vpc",
            output={"vpc_id": "vpc-123"},
            duration=45.5,
            metadata={"region": "us-east-1"},
        )

        assert result.success is True
        assert result.tool_name == "test_tool"
        assert result.action == "create_vpc"
        assert result.output == {"vpc_id": "vpc-123"}
        assert result.duration == 45.5
        assert result.error is None
        assert result.metadata == {"region": "us-east-1"}
        assert isinstance(result.created_at, datetime)

    def test_failed_tool_result(self):
        """Test failed tool result."""
        result = ToolResult(
            success=False,
            tool_name="test_tool",
            action="create_vpc",
            error="Insufficient permissions",
            duration=5.0,
        )

        assert result.success is False
        assert result.error == "Insufficient permissions"
        assert result.output == {}


class TestValidationResult:
    """Test ValidationResult model."""

    def test_valid_validation_result(self):
        """Test valid validation result."""
        result = ValidationResult(valid=True, normalized_params={"cidr": "10.0.0.0/16"})

        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.normalized_params == {"cidr": "10.0.0.0/16"}

    def test_invalid_validation_result(self):
        """Test invalid validation result."""
        result = ValidationResult(
            valid=False,
            errors=["Invalid CIDR block", "Missing required parameter"],
            warnings=["Using default value"],
            normalized_params={},
        )

        assert result.valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert "Invalid CIDR block" in result.errors


class TestCostEstimate:
    """Test CostEstimate model."""

    def test_cost_estimate_creation(self):
        """Test cost estimate creation."""
        estimate = CostEstimate(
            estimated_cost=125.50,
            currency="USD",
            cost_breakdown={"compute": 100.0, "storage": 25.50},
            confidence=0.95,
            factors=["Instance type", "Region"],
        )

        assert estimate.estimated_cost == 125.50
        assert estimate.currency == "USD"
        assert estimate.cost_breakdown == {"compute": 100.0, "storage": 25.50}
        assert estimate.confidence == 0.95
        assert estimate.factors == ["Instance type", "Region"]


class TestTool:
    """Test Tool base class."""

    @pytest.fixture
    def concrete_tool(self, tool_config):
        """Create a concrete tool for testing."""

        class ConcreteTool(Tool):
            def __init__(self, config, should_fail=False):
                super().__init__(config)
                self.should_fail = should_fail
                self.client_created = False
                self.validator_created = False
                self.actions_executed = []

            async def get_schema(self):
                return ToolSchema(
                    name=self.config.name,
                    description="Test tool",
                    version=self.config.version,
                    actions={
                        "create_vpc": {"description": "Create VPC"},
                        "delete_vpc": {"description": "Delete VPC"},
                    },
                )

            async def estimate_cost(self, action, params):
                return CostEstimate(estimated_cost=50.0, cost_breakdown={"base": 50.0})

            async def _create_client(self):
                self.client_created = True
                return Mock()

            async def _create_validator(self):
                self.validator_created = True
                validator = Mock()
                validator.validate = Mock(
                    return_value=ValidationResult(valid=True, normalized_params={})
                )
                return validator

            async def _execute_action(self, action, params):
                self.actions_executed.append((action, params))
                if self.should_fail:
                    raise Exception(f"Tool execution failed for {action}")
                return {
                    "action": action,
                    "params": params,
                    "resource_id": f"{action}_123",
                }

            async def _execute_rollback(self, execution_id):
                return {"rollback": "success", "execution_id": execution_id}

            async def _get_supported_actions(self):
                return ["create_vpc", "delete_vpc"]

        return ConcreteTool(tool_config)

    def test_tool_initialization(self, concrete_tool):
        """Test tool initialization."""
        assert concrete_tool.config.name == "test_tool"
        assert concrete_tool.status == ToolStatus.IDLE
        assert concrete_tool._client is None
        assert concrete_tool._validator is None
        assert concrete_tool._last_operation is None

    @pytest.mark.asyncio
    async def test_tool_initialize(self, concrete_tool):
        """Test tool initialization process."""
        await concrete_tool.initialize()

        assert concrete_tool.client_created is True
        assert concrete_tool.validator_created is True
        assert concrete_tool._client is not None
        assert concrete_tool._validator is not None

    @pytest.mark.asyncio
    async def test_tool_initialize_failure(self, tool_config):
        """Test tool initialization failure."""

        class FailingTool(Tool):
            async def get_schema(self):
                pass

            async def estimate_cost(self, action, params):
                pass

            async def _create_client(self):
                raise Exception("Client creation failed")

            async def _create_validator(self):
                pass

            async def _execute_action(self, action, params):
                pass

            async def _execute_rollback(self, execution_id):
                pass

            async def _get_supported_actions(self):
                return []

        tool = FailingTool(tool_config)

        with pytest.raises(ToolError, match="Initialization failed"):
            await tool.initialize()

    @pytest.mark.asyncio
    async def test_successful_execute(self, concrete_tool):
        """Test successful tool execution."""
        await concrete_tool.initialize()

        result = await concrete_tool.execute("create_vpc", {"cidr": "10.0.0.0/16"})

        assert result.success is True
        assert result.tool_name == "test_tool"
        assert result.action == "create_vpc"
        assert result.output["action"] == "create_vpc"
        assert result.output["resource_id"] == "create_vpc_123"
        assert result.duration is not None
        assert result.rollback_info is not None
        assert concrete_tool.status == ToolStatus.COMPLETED

        # Verify action was recorded
        assert len(concrete_tool.actions_executed) == 1
        assert concrete_tool.actions_executed[0] == (
            "create_vpc",
            {"cidr": "10.0.0.0/16"},
        )

    @pytest.mark.asyncio
    async def test_failed_execute(self, tool_config):
        """Test failed tool execution."""

        class FailingTool(Tool):
            async def get_schema(self):
                pass

            async def estimate_cost(self, action, params):
                pass

            async def _create_client(self):
                return Mock()

            async def _create_validator(self):
                return Mock()

            async def _execute_action(self, action, params):
                raise Exception("Execution failed")

            async def _execute_rollback(self, execution_id):
                pass

            async def _get_supported_actions(self):
                return ["test_action"]

        tool = FailingTool(tool_config)
        await tool.initialize()

        result = await tool.execute("test_action", {})

        assert result.success is False
        assert result.error == "Execution failed"
        assert tool.status == ToolStatus.FAILED

    @pytest.mark.asyncio
    async def test_validation_before_execute(self, concrete_tool):
        """Test validation is performed before execution."""
        await concrete_tool.initialize()

        # Mock validator to return invalid result
        mock_validator = Mock()
        mock_validator.validate = AsyncMock(
            return_value=ValidationResult(valid=False, errors=["Invalid parameter"])
        )
        concrete_tool._validator = mock_validator

        result = await concrete_tool.execute("create_vpc", {"invalid": "param"})

        assert result.success is False
        assert "Validation failed" in result.error
        assert len(concrete_tool.actions_executed) == 0  # Action should not be executed

    @pytest.mark.asyncio
    async def test_execute_with_retry(self, tool_config):
        """Test execution with retry logic."""

        class RetryTool(Tool):
            def __init__(self, config):
                super().__init__(config)
                self.attempt_count = 0

            async def get_schema(self):
                pass

            async def estimate_cost(self, action, params):
                pass

            async def _create_client(self):
                return Mock()

            async def _create_validator(self):
                return Mock()

            async def _execute_action(self, action, params):
                self.attempt_count += 1
                if self.attempt_count < 3:  # Fail first 2 attempts
                    raise Exception("Temporary failure")
                return {"success": True}

            async def _execute_rollback(self, execution_id):
                pass

            async def _get_supported_actions(self):
                return ["test_action"]

        tool = RetryTool(tool_config)
        await tool.initialize()

        result = await tool.execute("test_action", {})

        assert result.success is True
        assert tool.attempt_count == 3  # Should have retried

    @pytest.mark.asyncio
    async def test_execute_timeout(self, tool_config):
        """Test execution timeout."""

        class SlowTool(Tool):
            async def get_schema(self):
                pass

            async def estimate_cost(self, action, params):
                pass

            async def _create_client(self):
                return Mock()

            async def _create_validator(self):
                return Mock()

            async def _execute_action(self, action, params):
                await asyncio.sleep(1)  # Longer than timeout
                return {"success": True}

            async def _execute_rollback(self, execution_id):
                pass

            async def _get_supported_actions(self):
                return ["test_action"]

        # Set very short timeout
        tool_config.timeout = 0.1
        tool = SlowTool(tool_config)
        await tool.initialize()

        result = await tool.execute("test_action", {})

        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_rollback(self, concrete_tool):
        """Test rollback functionality."""
        await concrete_tool.initialize()

        result = await concrete_tool.rollback("exec_123")

        assert result.success is True
        assert result.action == "rollback"
        assert result.output["rollback"] == "success"
        assert result.output["execution_id"] == "exec_123"

    @pytest.mark.asyncio
    async def test_rollback_failure(self, tool_config):
        """Test rollback failure."""

        class FailingRollbackTool(Tool):
            async def get_schema(self):
                pass

            async def estimate_cost(self, action, params):
                pass

            async def _create_client(self):
                return Mock()

            async def _create_validator(self):
                return Mock()

            async def _execute_action(self, action, params):
                pass

            async def _execute_rollback(self, execution_id):
                raise Exception("Rollback failed")

            async def _get_supported_actions(self):
                return []

        tool = FailingRollbackTool(tool_config)
        await tool.initialize()

        result = await tool.rollback("exec_123")

        assert result.success is False
        assert "Rollback failed" in result.error
        assert tool.status == ToolStatus.FAILED

    @pytest.mark.asyncio
    async def test_validate_with_default_validator(self, concrete_tool):
        """Test validation with default validator (no custom validator)."""
        await concrete_tool.initialize()
        concrete_tool._validator = None  # Remove validator to test default behavior

        # Test with supported action
        result = await concrete_tool.validate("create_vpc", {})
        assert result.valid is True
        assert result.normalized_params == {}

        # Test with unsupported action
        result = await concrete_tool.validate("unsupported_action", {})
        assert result.valid is False
        assert "Unsupported action" in result.errors[0]

    @pytest.mark.asyncio
    async def test_validate_with_custom_validator(self, concrete_tool):
        """Test validation with custom validator."""
        await concrete_tool.initialize()

        # Mock custom validator
        mock_validator = Mock()
        mock_validator.validate = AsyncMock(
            return_value=ValidationResult(
                valid=True, normalized_params={"normalized": "value"}
            )
        )
        concrete_tool._validator = mock_validator

        result = await concrete_tool.validate("create_vpc", {"param": "value"})

        assert result.valid is True
        assert result.normalized_params == {"normalized": "value"}
        mock_validator.validate.assert_called_once_with(
            "create_vpc", {"param": "value"}
        )

    def test_get_rollback_info(self, concrete_tool):
        """Test rollback info generation."""
        action = "create_vpc"
        params = {"cidr": "10.0.0.0/16"}
        result = {"vpc_id": "vpc-123"}

        rollback_info = concrete_tool._get_rollback_info(action, params, result)

        assert rollback_info["action"] == action
        assert rollback_info["params"] == params
        assert rollback_info["reversible"] is True
        assert "timestamp" in rollback_info


class TestToolExceptions:
    """Test tool-related exceptions."""

    def test_tool_error(self):
        """Test ToolError exception."""
        error = ToolError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_tool_validation_error(self):
        """Test ToolValidationError exception."""
        error = ToolValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, ToolError)

    def test_tool_execution_error(self):
        """Test ToolExecutionError exception."""
        error = ToolExecutionError("Execution failed")
        assert str(error) == "Execution failed"
        assert isinstance(error, ToolError)

    def test_tool_timeout_error(self):
        """Test ToolTimeoutError exception."""
        error = ToolTimeoutError("Operation timed out")
        assert str(error) == "Operation timed out"
        assert isinstance(error, ToolError)
