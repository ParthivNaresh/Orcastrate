"""
Integration tests for justfile environment management commands.

Tests cover:
- Environment loading commands
- Configuration validation commands
- Configuration display commands
- Environment listing commands
- Error handling and edge cases
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


class TestJustfileEnvironmentCommands:
    """Test justfile environment management commands."""

    def test_list_envs_command(self):
        """Test the 'just list-envs' command."""
        result = subprocess.run(
            ["just", "list-envs"], capture_output=True, text=True, cwd=Path.cwd()
        )

        assert result.returncode == 0
        output = result.stdout

        # Should show available environments
        assert "🌍 Available Environment Configurations:" in output
        assert "📁 Local environments:" in output
        assert "🔧 local" in output
        assert "🔧 testing" in output
        assert "🔧 staging" in output
        assert "🔧 ci" in output

        # Should show usage examples
        assert "📋 Usage examples:" in output
        assert "just load-env local" in output
        assert "just validate-env testing" in output
        assert "just show-config staging" in output

    def test_validate_env_local_command(self):
        """Test the 'just validate-env local' command."""
        result = subprocess.run(
            ["just", "validate-env", "local"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        assert result.returncode == 0
        output = result.stdout

        # Should show validation progress
        assert "🔍 Validating environment: local" in output
        assert "🔧 Checking configuration validity..." in output
        assert "✅ Configuration loaded successfully" in output
        assert "✅ Environment validation completed" in output

    def test_validate_env_testing_command(self):
        """Test the 'just validate-env testing' command."""
        result = subprocess.run(
            ["just", "validate-env", "testing"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        assert result.returncode == 0
        output = result.stdout

        # Should show validation for testing environment
        assert "🔍 Validating environment: testing" in output
        assert "✅ Configuration loaded successfully" in output

    def test_show_config_local_command(self):
        """Test the 'just show-config local' command."""
        result = subprocess.run(
            ["just", "show-config", "local"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        assert result.returncode == 0
        output = result.stdout

        # Should show configuration header
        assert "📋 Configuration for environment: local" in output

        # Should contain JSON configuration
        assert '"app_name": "Orcastrate"' in output
        assert '"environment": "development"' in output
        assert '"database":' in output
        assert '"cloud":' in output
        assert '"llm":' in output

        # Sensitive values should be masked
        assert "***MASKED***" in output

    def test_validate_env_nonexistent_environment(self):
        """Test validation with non-existent environment."""
        result = subprocess.run(
            ["just", "validate-env", "nonexistent"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        # Should fail with appropriate error
        assert result.returncode != 0
        output = result.stderr + result.stdout

        # Should show error about failed recipe execution
        assert "Recipe `load-env` failed with exit code 1" in output
        assert "Recipe `validate-env` failed" in output
        assert "nonexistent" in output

    def test_show_config_nonexistent_environment(self):
        """Test show config with non-existent environment."""
        result = subprocess.run(
            ["just", "show-config", "nonexistent"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        # Should fail with appropriate error
        assert result.returncode != 0

    def test_load_env_command_success(self):
        """Test successful environment loading."""
        result = subprocess.run(
            ["just", "load-env", "local"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        assert result.returncode == 0
        output = result.stdout

        # Should show success message
        assert "✅ Loaded environment: local" in output

    def test_load_env_command_failure(self):
        """Test environment loading failure."""
        result = subprocess.run(
            ["just", "load-env", "nonexistent"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        assert result.returncode != 0
        output = result.stdout

        # Should show error and available environments
        assert "❌ Environment file not found" in output
        assert "📋 Available environments:" in output


class TestJustfileScriptIntegration:
    """Test integration with Python validation and display scripts."""

    def test_validate_config_script_directly(self):
        """Test the validate_config.py script directly."""
        # Test with default environment
        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_config.py",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        assert result.returncode == 0
        output = result.stdout

        assert "✅ Configuration loaded successfully" in output
        assert "Environment: development" in output
        assert "App Name: Orcastrate" in output

    def test_show_config_script_directly(self):
        """Test the show_config.py script directly."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/show_config.py",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        assert result.returncode == 0
        output = result.stdout

        # Should output valid JSON
        import json

        try:
            config = json.loads(output)
            assert config["app_name"] == "Orcastrate"
            assert "database" in config
            assert "cloud" in config
            assert "llm" in config
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON output: {output}")

    def test_validate_config_with_warnings(self):
        """Test validation script with configuration that should trigger warnings."""
        # Set environment variables that should trigger warnings in production
        env = os.environ.copy()
        env.update(
            {
                "ENVIRONMENT": "production",
                "SECRET_KEY": "dev-secret-key",  # Should trigger warning
                "OPENAI_API_KEY": "your-api-key-here",  # Should trigger warning
            }
        )

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_config.py",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env=env,
        )

        assert result.returncode == 0
        output = result.stdout

        assert "✅ Configuration loaded successfully" in output
        assert "⚠️  Configuration warnings:" in output
        assert "SECRET_KEY using default development value" in output
        assert "OPENAI_API_KEY appears to be placeholder" in output

    def test_validate_config_secure_production(self):
        """Test validation script with secure production configuration."""
        env = os.environ.copy()
        env.update(
            {
                "ENVIRONMENT": "production",
                "SECRET_KEY": "secure-production-key-with-entropy",
                "OPENAI_API_KEY": "sk-real-openai-key",
                "AWS_SECRET_ACCESS_KEY": "real-aws-secret",
            }
        )

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_config.py",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env=env,
        )

        assert result.returncode == 0
        output = result.stdout

        assert "✅ Configuration loaded successfully" in output
        assert "✅ Security configuration looks good" in output
        assert "⚠️  Configuration warnings:" not in output


class TestEnvironmentFileHandling:
    """Test handling of environment files."""

    def test_all_environment_files_exist(self):
        """Test that all expected environment files exist."""
        env_dir = Path("scripts/env")
        assert env_dir.exists(), "Environment directory should exist"

        expected_files = ["local.env", "testing.env", "staging.env", "ci.env"]
        for filename in expected_files:
            env_file = env_dir / filename
            assert env_file.exists(), f"Environment file {filename} should exist"

    def test_environment_file_format(self):
        """Test that environment files have correct format."""
        env_dir = Path("scripts/env")
        env_files = list(env_dir.glob("*.env"))

        assert len(env_files) > 0, "Should have at least one environment file"

        for env_file in env_files:
            with open(env_file, "r") as f:
                content = f.read()

                # Should contain ENVIRONMENT variable
                assert (
                    "ENVIRONMENT=" in content
                ), f"{env_file.name} should specify ENVIRONMENT"

                # Should not contain obvious placeholder values
                lines = content.split("\n")
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        # Check for obvious placeholders
                        if (
                            "password" in key.lower()
                            or "secret" in key.lower()
                            or "key" in key.lower()
                        ):
                            assert (
                                "your-" not in value.lower()
                            ), f"Found placeholder in {env_file.name}: {line}"
                            assert (
                                "change-me" not in value.lower()
                            ), f"Found placeholder in {env_file.name}: {line}"

    def test_local_env_development_settings(self):
        """Test that local.env has development-appropriate settings."""
        local_env_file = Path("scripts/env/local.env")
        assert local_env_file.exists()

        with open(local_env_file, "r") as f:
            content = f.read()

            # Should be development environment
            assert "ENVIRONMENT=development" in content

            # Should use localhost for databases
            assert "POSTGRES_HOST=localhost" in content
            assert "localhost" in content  # Should contain localhost references

            # Should use LocalStack
            assert "USE_LOCALSTACK=true" in content

    def test_testing_env_testing_settings(self):
        """Test that testing.env has testing-appropriate settings."""
        testing_env_file = Path("scripts/env/testing.env")
        assert testing_env_file.exists()

        with open(testing_env_file, "r") as f:
            content = f.read()

            # Should be testing environment
            assert "ENVIRONMENT=testing" in content

            # Should have live testing setting (disabled by default for safety)
            assert "ENABLE_LIVE_TESTING=false" in content

            # Should use LocalStack
            assert "USE_LOCALSTACK=true" in content

    def test_ci_env_ci_settings(self):
        """Test that ci.env has CI-appropriate settings."""
        ci_env_file = Path("scripts/env/ci.env")
        assert ci_env_file.exists()

        with open(ci_env_file, "r") as f:
            content = f.read()

            # Should be ci environment
            assert "ENVIRONMENT=ci" in content

            # Should have appropriate log level for CI
            assert "LOG_LEVEL=" in content


class TestErrorHandling:
    """Test error handling in environment management."""

    def test_invalid_environment_variable_format(self):
        """Test handling of invalid environment variable formats."""
        # Test with invalid temperature
        env = os.environ.copy()
        env["OPENAI_TEMPERATURE"] = "invalid"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_config.py",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env=env,
        )

        # Should fail with validation error
        assert result.returncode != 0
        output = result.stderr + result.stdout
        assert "❌ Configuration error:" in output

    def test_missing_required_environment_in_production(self):
        """Test validation of production environment requirements."""
        env = os.environ.copy()
        # Clear important environment variables to simulate missing config
        for key in ["SECRET_KEY", "POSTGRES_PASSWORD"]:
            if key in env:
                del env[key]
        env["ENVIRONMENT"] = "production"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_config.py",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env=env,
        )

        # Should complete but show warnings
        assert result.returncode == 0
        output = result.stdout

        # Should show configuration warnings for production
        if "dev-secret-key" in output:
            assert "⚠️  Configuration warnings:" in output


class TestEnvironmentIsolation:
    """Test that different environments are properly isolated."""

    def test_development_vs_production_isolation(self):
        """Test that development and production environments are isolated."""
        # Test development environment
        dev_env = os.environ.copy()
        dev_env.update(
            {
                "ENVIRONMENT": "development",
                "DEBUG": "true",
                "USE_LOCALSTACK": "true",
            }
        )

        result_dev = subprocess.run(
            [
                sys.executable,
                "scripts/show_config.py",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env=dev_env,
        )

        assert result_dev.returncode == 0
        dev_config = result_dev.stdout

        # Test production environment
        prod_env = os.environ.copy()
        prod_env.update(
            {
                "ENVIRONMENT": "production",
                "DEBUG": "false",
                "USE_LOCALSTACK": "false",
            }
        )

        result_prod = subprocess.run(
            [
                sys.executable,
                "scripts/show_config.py",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env=prod_env,
        )

        assert result_prod.returncode == 0
        prod_config = result_prod.stdout

        # Verify environments are different
        assert '"environment": "development"' in dev_config
        assert '"environment": "production"' in prod_config
        assert dev_config != prod_config

    def test_testing_vs_staging_isolation(self):
        """Test that testing and staging environments are isolated."""
        # Test testing environment
        test_env = os.environ.copy()
        test_env.update(
            {
                "ENVIRONMENT": "testing",
                "ENABLE_LIVE_TESTING": "true",
            }
        )

        result_test = subprocess.run(
            [
                sys.executable,
                "scripts/show_config.py",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env=test_env,
        )

        assert result_test.returncode == 0

        # Test staging environment
        staging_env = os.environ.copy()
        staging_env.update(
            {
                "ENVIRONMENT": "staging",
                "ENABLE_LIVE_TESTING": "false",
            }
        )

        result_staging = subprocess.run(
            [
                sys.executable,
                "scripts/show_config.py",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env=staging_env,
        )

        assert result_staging.returncode == 0

        # Verify environments are different
        test_config = result_test.stdout
        staging_config = result_staging.stdout

        assert test_config != staging_config


@pytest.mark.slow
class TestPerformanceAndReliability:
    """Test performance and reliability of environment management."""

    def test_configuration_loading_performance(self):
        """Test that configuration loading is fast enough."""
        import time

        start_time = time.time()

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_config.py",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        end_time = time.time()
        duration = end_time - start_time

        assert result.returncode == 0
        assert (
            duration < 5.0
        ), f"Configuration loading took too long: {duration:.2f} seconds"

    def test_repeated_environment_switching(self):
        """Test that repeated environment switching works reliably."""
        environments = ["local", "testing", "staging", "ci"]

        for env in environments * 3:  # Test each environment 3 times
            result = subprocess.run(
                ["just", "validate-env", env],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )

            assert result.returncode == 0, f"Failed to validate environment {env}"
            assert f"Validating environment: {env}" in result.stdout
            assert "✅ Configuration loaded successfully" in result.stdout

    def test_concurrent_environment_access(self):
        """Test that concurrent access to environments works."""
        import concurrent.futures

        def validate_environment(env_name):
            result = subprocess.run(
                ["just", "validate-env", env_name],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )
            return result.returncode == 0

        environments = ["local", "testing", "staging", "ci"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(validate_environment, env) for env in environments
            ]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # All validations should succeed
        assert all(results), "Some concurrent validations failed"
