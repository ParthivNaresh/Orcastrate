"""
Integration tests for configuration system with actual tools and components.

Tests cover:
- Configuration integration with AWS tool
- Configuration integration with LLM clients
- Environment switching behavior
- Configuration validation in real scenarios
- End-to-end configuration workflows
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from src.config.settings import get_settings, reload_settings
from src.tools.aws import AWSCloudTool
from src.tools.base import ToolConfig


class TestAWSToolIntegration:
    """Test AWS tool integration with configuration system."""

    def test_aws_tool_uses_config(self):
        """Test that AWS tool properly uses configuration settings."""
        with patch.dict(
            os.environ,
            {
                "AWS_DEFAULT_REGION": "eu-west-1",
                "AWS_ACCESS_KEY_ID": "test-key-id",
                "AWS_SECRET_ACCESS_KEY": "test-secret-key",
            },
        ):
            reload_settings()

            # Create AWS tool - should use configuration
            aws_tool = AWSCloudTool()

            # Verify it uses the configured region
            assert aws_tool._region == "eu-west-1"

    def test_aws_tool_with_default_config(self):
        """Test AWS tool with default configuration."""
        # Clear AWS environment variables
        aws_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
        with patch.dict(os.environ, {var: "" for var in aws_vars}, clear=True):
            reload_settings()

            aws_tool = AWSCloudTool()

            # Should use default region from config
            assert aws_tool._region == "us-west-2"  # Default from settings

    def test_aws_tool_custom_config(self):
        """Test AWS tool with custom configuration."""
        custom_config = ToolConfig(
            name="custom-aws", version="1.0.0", timeout=600, retry_count=5
        )

        aws_tool = AWSCloudTool(config=custom_config)

        assert aws_tool.config.name == "custom-aws"
        assert aws_tool.config.timeout == 600
        assert aws_tool.config.retry_count == 5


class TestLLMClientIntegration:
    """Test LLM client integration with configuration system."""

    @pytest.mark.skipif(True, reason="LLM clients not yet updated to use config")
    def test_openai_client_uses_config(self):
        """Test that OpenAI client uses configuration settings."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test-key",
                "OPENAI_DEFAULT_MODEL": "gpt-4-turbo",
                "OPENAI_TEMPERATURE": "0.5",
            },
        ):
            reload_settings()

            # This would test the OpenAI client once it's updated
            # from src.planners.llm.openai_client import OpenAIClient
            # client = OpenAIClient()
            # assert client.model == 'gpt-4-turbo'
            # assert client.temperature == 0.5

    @pytest.mark.skipif(True, reason="LLM clients not yet updated to use config")
    def test_anthropic_client_uses_config(self):
        """Test that Anthropic client uses configuration settings."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "ant-test-key",
                "ANTHROPIC_DEFAULT_MODEL": "claude-3-opus",
                "ANTHROPIC_TEMPERATURE": "0.3",
            },
        ):
            reload_settings()

            # This would test the Anthropic client once it's updated
            # from src.planners.llm.anthropic_client import AnthropicClient
            # client = AnthropicClient()
            # assert client.model == 'claude-3-opus'
            # assert client.temperature == 0.3


class TestEnvironmentSwitching:
    """Test environment switching behavior."""

    def test_development_to_production_switch(self):
        """Test switching from development to production environment."""
        # Start with development
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "development",
                "DEBUG": "true",
                "SECRET_KEY": "dev-secret",
            },
        ):
            reload_settings()
            settings = get_settings()

            assert settings.is_development
            assert not settings.is_production
            assert settings.security.secret_key == "dev-secret"

        # Switch to production
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "production",
                "DEBUG": "false",
                "SECRET_KEY": "prod-secret-key-123",
            },
        ):
            reload_settings()
            settings = get_settings()

            assert not settings.is_development
            assert settings.is_production
            assert settings.security.secret_key == "prod-secret-key-123"

    def test_testing_environment_behavior(self):
        """Test testing environment specific behavior."""
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "testing",
                "USE_LOCALSTACK": "true",
                "ENABLE_LIVE_TESTING": "true",
                "TEST_POSTGRES_HOST": "test-db",
            },
        ):
            reload_settings()
            settings = get_settings()

            assert settings.is_testing
            assert settings.testing.use_localstack
            assert settings.testing.enable_live_testing
            assert settings.testing.test_postgres_host == "test-db"

    def test_ci_environment_behavior(self):
        """Test CI environment specific behavior."""
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "ci",
                "ENABLE_METRICS": "false",
                "LOG_LEVEL": "WARNING",
            },
        ):
            reload_settings()
            settings = get_settings()

            assert settings.is_testing  # CI is considered testing
            assert not settings.monitoring.enable_metrics
            assert settings.monitoring.log_level == "WARNING"


class TestConfigurationValidation:
    """Test configuration validation in real scenarios."""

    def test_production_configuration_warnings(self):
        """Test that production configuration triggers appropriate warnings."""
        # Simulate production with insecure defaults
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "production",
                "SECRET_KEY": "dev-secret-key",  # Insecure default
                "OPENAI_API_KEY": "your-api-key-here",  # Placeholder
            },
        ):
            reload_settings()
            settings = get_settings()

            # These would be caught by validation script
            warnings = []

            if settings.environment == "production":
                if settings.security.secret_key == "dev-secret-key":
                    warnings.append("SECRET_KEY using default development value")
                if settings.llm.openai_api_key and "your-" in str(
                    settings.llm.openai_api_key
                ):
                    warnings.append("OPENAI_API_KEY appears to be placeholder")

            assert len(warnings) == 2
            assert "SECRET_KEY using default development value" in warnings
            assert "OPENAI_API_KEY appears to be placeholder" in warnings

    def test_secure_production_configuration(self):
        """Test that secure production configuration passes validation."""
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "production",
                "SECRET_KEY": "secure-production-key-with-entropy",
                "OPENAI_API_KEY": "sk-real-openai-key",
                "AWS_SECRET_ACCESS_KEY": "real-aws-secret-key",
            },
        ):
            reload_settings()
            settings = get_settings()

            # These would pass validation script
            warnings = []

            if settings.environment == "production":
                if settings.security.secret_key == "dev-secret-key":
                    warnings.append("SECRET_KEY using default development value")
                if settings.llm.openai_api_key and "your-" in str(
                    settings.llm.openai_api_key
                ):
                    warnings.append("OPENAI_API_KEY appears to be placeholder")
                if settings.cloud.aws_secret_access_key and "your-" in str(
                    settings.cloud.aws_secret_access_key
                ):
                    warnings.append("AWS credentials appear to be placeholders")

            assert len(warnings) == 0

    def test_invalid_configuration_values(self):
        """Test that invalid configuration values are rejected."""
        # Test invalid environment
        with patch.dict(os.environ, {"ENVIRONMENT": "invalid-env"}):
            with pytest.raises(ValueError, match="Environment must be one of"):
                reload_settings()

        # Test invalid temperature
        with patch.dict(os.environ, {"OPENAI_TEMPERATURE": "3.0"}):
            with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
                reload_settings()

        # Test invalid log level
        with patch.dict(os.environ, {"LOG_LEVEL": "INVALID"}):
            with pytest.raises(ValueError, match="Log level must be one of"):
                reload_settings()


class TestEndToEndWorkflows:
    """Test end-to-end configuration workflows."""

    def test_full_development_workflow(self):
        """Test complete development environment workflow."""
        # Simulate loading local development environment
        dev_env = {
            "ENVIRONMENT": "development",
            "DEBUG": "true",
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5432",
            "REDIS_URL": "redis://localhost:6379",
            "USE_LOCALSTACK": "true",
            "AWS_DEFAULT_REGION": "us-west-2",
            "OPENAI_DEFAULT_MODEL": "gpt-4",
            "ENABLE_LLM_FEATURES": "true",
            "LOG_LEVEL": "DEBUG",
        }

        with patch.dict(os.environ, dev_env):
            reload_settings()
            settings = get_settings()

            # Verify development configuration
            assert settings.is_development
            assert settings.debug
            assert settings.database.postgres_host == "localhost"
            assert settings.database.postgres_port == 5432
            assert settings.testing.use_localstack
            assert settings.cloud.aws_region == "us-west-2"
            assert settings.llm.openai_default_model == "gpt-4"
            assert settings.enable_llm_features
            assert settings.monitoring.log_level == "DEBUG"

            # Test safe dictionary doesn't expose secrets
            safe_dict = settings.get_safe_dict()
            assert "MASKED" not in str(safe_dict["database"]["postgres_host"])
            assert safe_dict["monitoring"]["log_level"] == "DEBUG"

    def test_full_production_workflow(self):
        """Test complete production environment workflow."""
        # Simulate loading production environment
        prod_env = {
            "ENVIRONMENT": "production",
            "DEBUG": "false",
            "SECRET_KEY": "production-secret-key-with-128-bits-entropy",
            "POSTGRES_HOST": "prod-db.company.com",
            "POSTGRES_PORT": "5432",
            "POSTGRES_PASSWORD": "secure-db-password",
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_ACCESS_KEY_ID": "AKIA1234567890ABCDEF",
            "AWS_SECRET_ACCESS_KEY": "secret-key-for-production",
            "OPENAI_API_KEY": "sk-production-openai-key",
            "ANTHROPIC_API_KEY": "ant-production-anthropic-key",
            "CORS_ORIGINS": "https://app.company.com,https://api.company.com",
            "LOG_LEVEL": "WARNING",
            "ENABLE_METRICS": "true",
            "ENABLE_TRACING": "true",
        }

        with patch.dict(os.environ, prod_env):
            reload_settings()
            settings = get_settings()

            # Verify production configuration
            assert settings.is_production
            assert not settings.debug
            assert (
                settings.security.secret_key
                == "production-secret-key-with-128-bits-entropy"
            )
            assert settings.database.postgres_host == "prod-db.company.com"
            assert settings.cloud.aws_region == "us-east-1"
            assert settings.cloud.aws_access_key_id == "AKIA1234567890ABCDEF"
            assert settings.llm.openai_api_key == "sk-production-openai-key"
            assert settings.security.cors_origins == [
                "https://app.company.com",
                "https://api.company.com",
            ]
            assert settings.monitoring.log_level == "WARNING"
            assert settings.monitoring.enable_metrics
            assert settings.monitoring.enable_tracing

            # Test safe dictionary masks sensitive values
            safe_dict = settings.get_safe_dict()
            assert safe_dict["security"]["secret_key"] == "***MASKED***"
            assert safe_dict["database"]["postgres_password"] == "***MASKED***"
            assert safe_dict["cloud"]["aws_secret_access_key"] == "***MASKED***"
            assert safe_dict["llm"]["openai_api_key"] == "***MASKED***"
            assert safe_dict["llm"]["anthropic_api_key"] == "***MASKED***"

            # Non-sensitive values should not be masked
            assert safe_dict["database"]["postgres_host"] == "prod-db.company.com"
            assert safe_dict["cloud"]["aws_region"] == "us-east-1"

    def test_configuration_with_tools_integration(self):
        """Test configuration system working with actual tools."""
        test_env = {
            "ENVIRONMENT": "testing",
            "AWS_DEFAULT_REGION": "us-west-1",
            "USE_LOCALSTACK": "true",
            "LOCALSTACK_ENDPOINT": "http://localhost:4566",
        }

        with patch.dict(os.environ, test_env):
            reload_settings()
            settings = get_settings()

            # Create AWS tool with configuration
            aws_tool = AWSCloudTool()

            # Verify tool uses configuration
            assert aws_tool._region == "us-west-1"
            assert settings.testing.use_localstack
            assert settings.testing.localstack_endpoint == "http://localhost:4566"

            # Tool should be properly configured for testing
            assert settings.is_testing
            assert aws_tool.config.name == "aws"


class TestJustfileIntegration:
    """Test integration with justfile commands."""

    def test_environment_loading_simulation(self):
        """Test simulation of justfile environment loading."""
        # This simulates what happens when "just load-env local" is run

        # Simulate local.env file contents
        local_env_vars = {
            "ENVIRONMENT": "development",
            "DEBUG": "true",
            "POSTGRES_HOST": "localhost",
            "USE_LOCALSTACK": "true",
            "LOG_LEVEL": "DEBUG",
        }

        # Simulate loading environment
        with patch.dict(os.environ, local_env_vars):
            reload_settings()
            settings = get_settings()

            # Verify loaded configuration
            assert settings.environment == "development"
            assert settings.debug is True
            assert settings.database.postgres_host == "localhost"
            assert settings.testing.use_localstack is True
            assert settings.monitoring.log_level == "DEBUG"

    def test_configuration_validation_simulation(self):
        """Test simulation of justfile configuration validation."""
        # This simulates what happens when "just validate-env production" is run

        # Simulate production.env with some issues
        prod_env_vars = {
            "ENVIRONMENT": "production",
            "SECRET_KEY": "dev-secret-key",  # Issue: using dev key in prod
            "POSTGRES_HOST": "prod-db.company.com",
            "OPENAI_API_KEY": "sk-real-key",
        }

        with patch.dict(os.environ, prod_env_vars):
            reload_settings()
            settings = get_settings()

            # Simulate validation logic from validate_config.py
            validation_warnings = []

            if settings.environment == "production":
                if settings.security.secret_key == "dev-secret-key":
                    validation_warnings.append(
                        "SECRET_KEY using default development value"
                    )

            # Should catch the validation issue
            assert len(validation_warnings) == 1
            assert "SECRET_KEY using default development value" in validation_warnings

    def test_configuration_display_simulation(self):
        """Test simulation of justfile configuration display."""
        # This simulates what happens when "just show-config staging" is run

        staging_env_vars = {
            "ENVIRONMENT": "staging",
            "SECRET_KEY": "staging-secret-key",
            "POSTGRES_PASSWORD": "staging-db-password",
            "AWS_SECRET_ACCESS_KEY": "staging-aws-secret",
        }

        with patch.dict(os.environ, staging_env_vars):
            reload_settings()
            settings = get_settings()

            # Simulate safe configuration display
            safe_config = settings.get_safe_dict()

            # Sensitive values should be masked
            assert safe_config["security"]["secret_key"] == "***MASKED***"
            assert safe_config["database"]["postgres_password"] == "***MASKED***"
            assert safe_config["cloud"]["aws_secret_access_key"] == "***MASKED***"

            # Non-sensitive values should be visible
            assert safe_config["environment"] == "staging"
            assert safe_config["app_name"] == "Orcastrate"


@pytest.fixture
def temp_env_file():
    """Create a temporary environment file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write(
            """
# Test environment file
ENVIRONMENT=testing
DEBUG=true
POSTGRES_HOST=test-db
POSTGRES_PORT=5433
USE_LOCALSTACK=true
AWS_DEFAULT_REGION=us-west-2
OPENAI_DEFAULT_MODEL=gpt-4
LOG_LEVEL=DEBUG
"""
        )
        f.flush()
        yield f.name

    # Cleanup
    os.unlink(f.name)


class TestRealEnvironmentFiles:
    """Test with actual environment files."""

    def test_env_file_loading(self, temp_env_file):
        """Test loading configuration from actual environment file."""
        # Read the temporary env file and set environment variables
        env_vars = {}
        with open(temp_env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key] = value

        with patch.dict(os.environ, env_vars):
            reload_settings()
            settings = get_settings()

            # Verify configuration from file
            assert settings.environment == "testing"
            assert settings.debug is True
            assert settings.database.postgres_host == "test-db"
            assert settings.database.postgres_port == 5433
            assert settings.testing.use_localstack is True
            assert settings.cloud.aws_region == "us-west-2"
            assert settings.llm.openai_default_model == "gpt-4"
            assert settings.monitoring.log_level == "DEBUG"
