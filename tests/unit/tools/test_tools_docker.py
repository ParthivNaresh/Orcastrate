"""
Tests for Docker tool implementation.
"""

import json
from unittest.mock import patch

import pytest

from src.tools.base import ToolConfig, ToolError
from src.tools.docker import DockerTool


class TestDockerTool:
    """Test Docker tool functionality."""

    @pytest.fixture
    def docker_config(self):
        """Create Docker tool configuration."""
        return ToolConfig(
            name="docker",
            version="1.0.0",
            timeout=300,
            retry_count=3,
        )

    @pytest.fixture
    def docker_tool(self, docker_config):
        """Create Docker tool instance."""
        return DockerTool(docker_config)

    @pytest.mark.asyncio
    async def test_tool_initialization(self, docker_tool):
        """Test Docker tool initialization."""
        with patch.object(docker_tool, "_run_command") as mock_run:
            mock_run.side_effect = [
                {"returncode": 0, "stdout": "Docker version 20.10.0", "stderr": ""},
                {
                    "returncode": 0,
                    "stdout": "docker-compose version 1.29.0",
                    "stderr": "",
                },
            ]

            await docker_tool.initialize()

            assert docker_tool._docker_available is True
            assert docker_tool._compose_available is True
            assert "Docker version" in docker_tool._docker_version

    @pytest.mark.asyncio
    async def test_initialization_docker_not_available(self, docker_tool):
        """Test initialization when Docker is not available."""
        with patch.object(docker_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 1,
                "stdout": "",
                "stderr": "Command not found",
            }

            with pytest.raises(ToolError, match="Docker is not available"):
                await docker_tool.initialize()

    @pytest.mark.asyncio
    async def test_get_schema(self, docker_tool):
        """Test getting tool schema."""
        schema = await docker_tool.get_schema()

        assert schema.name == "docker"
        assert schema.description == "Docker container management tool"
        assert "build_image" in schema.actions
        assert "run_container" in schema.actions
        assert "compose_up" in schema.actions
        assert "docker" in schema.required_permissions
        assert "docker" in schema.dependencies

    @pytest.mark.asyncio
    async def test_estimate_cost_build_image(self, docker_tool):
        """Test cost estimation for building images."""
        cost = await docker_tool.estimate_cost("build_image", {})

        assert cost.estimated_cost > 0
        assert "image_build" in cost.cost_breakdown
        assert cost.confidence == 0.7

    @pytest.mark.asyncio
    async def test_estimate_cost_run_container(self, docker_tool):
        """Test cost estimation for running containers."""
        cost = await docker_tool.estimate_cost("run_container", {})

        assert cost.estimated_cost > 0
        assert "container_runtime" in cost.cost_breakdown

    @pytest.mark.asyncio
    async def test_validate_build_image_params(self, docker_tool):
        """Test parameter validation for build_image action."""
        # Create validator directly without full initialization
        validator = await docker_tool._create_validator()

        # Valid parameters
        result = validator.validate(
            "build_image",
            {
                "context_path": "/path/to/context",
                "image_name": "my-app",
                "tag": "v1.0",
            },
        )
        assert result.valid is True

        # Missing required parameters
        result = validator.validate("build_image", {})
        assert result.valid is False
        assert "context_path is required for build_image" in result.errors
        assert "image_name is required for build_image" in result.errors

    @pytest.mark.asyncio
    async def test_validate_run_container_params(self, docker_tool):
        """Test parameter validation for run_container action."""
        # Create validator directly without full initialization
        validator = await docker_tool._create_validator()

        # Valid parameters
        result = validator.validate(
            "run_container",
            {
                "image": "nginx:latest",
                "ports": ["8080:80"],
            },
        )
        assert result.valid is True

        # Missing required parameters
        result = validator.validate("run_container", {})
        assert result.valid is False
        assert "image is required for run_container" in result.errors

        # Invalid port format
        result = validator.validate(
            "run_container",
            {
                "image": "nginx:latest",
                "ports": ["8080"],  # Missing container port
            },
        )
        assert result.valid is True  # Only warning, not error
        assert any(
            "Port '8080' should be in format" in warning for warning in result.warnings
        )

    @pytest.mark.asyncio
    async def test_build_image_success(self, docker_tool):
        """Test successful image building."""
        with patch.object(docker_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "Successfully built image",
                "stderr": "",
            }

            docker_tool._docker_available = True

            result = await docker_tool._build_image(
                {
                    "context_path": "/app",
                    "image_name": "my-app",
                    "tag": "v1.0",
                    "dockerfile": "Dockerfile",
                    "build_args": {"NODE_ENV": "production"},
                    "no_cache": True,
                }
            )

            assert result["success"] is True
            assert result["image_name"] == "my-app:v1.0"  # Now returns full tag
            assert result["tag"] == "my-app:v1.0"
            assert "Successfully built" in result["build_output"]

            # Verify command was constructed correctly
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "docker" in call_args
            assert "build" in call_args
            assert "--no-cache" in call_args
            assert "--build-arg" in call_args
            assert "NODE_ENV=production" in call_args

    @pytest.mark.asyncio
    async def test_run_container_success(self, docker_tool):
        """Test successful container running."""
        with patch.object(docker_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "container_id_123",
                "stderr": "",
            }

            docker_tool._docker_available = True

            result = await docker_tool._run_container(
                {
                    "image": "nginx:latest",
                    "name": "my-nginx",
                    "ports": ["8080:80"],
                    "volumes": ["/host/data:/container/data"],
                    "environment": {"ENV": "production"},
                    "detached": True,
                }
            )

            assert result["success"] is True
            assert result["container_id"] == "container_id_123"
            assert result["image"] == "nginx:latest"

            # Verify command was constructed correctly
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "docker" in call_args
            assert "run" in call_args
            assert "-d" in call_args  # detached
            assert "--name" in call_args
            assert "my-nginx" in call_args

    @pytest.mark.asyncio
    async def test_list_containers_success(self, docker_tool):
        """Test successful container listing."""
        mock_containers = [
            {"ID": "abc123", "Image": "nginx:latest", "Status": "Up 5 minutes"},
            {"ID": "def456", "Image": "redis:6", "Status": "Up 1 hour"},
        ]

        with patch.object(docker_tool, "_run_command") as mock_run:
            # Simulate JSON output from docker ps
            json_output = "\n".join(
                json.dumps(container) for container in mock_containers
            )
            mock_run.return_value = {
                "returncode": 0,
                "stdout": json_output,
                "stderr": "",
            }

            docker_tool._docker_available = True

            result = await docker_tool._list_containers({"all": True})

            assert result["success"] is True
            assert len(result["containers"]) == 2
            assert (
                result["containers"][0]["container_id"] == "abc123"
            )  # Now uses normalized field names
            assert (
                result["containers"][1]["image"] == "redis:6"
            )  # Now uses normalized field names

    @pytest.mark.asyncio
    async def test_compose_up_success(self, docker_tool):
        """Test successful Docker Compose up."""
        with patch.object(docker_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "Starting services...\nServices started successfully",
                "stderr": "",
            }

            docker_tool._docker_available = True
            docker_tool._compose_available = True
            docker_tool._compose_cmd = ["docker", "compose"]  # Set the compose command

            result = await docker_tool._compose_up(
                {
                    "compose_file": "docker-compose.yml",
                    "project_name": "myproject",
                    "services": ["web", "db"],
                    "detached": True,
                    "build": True,
                }
            )

            assert result["success"] is True
            assert result["compose_file"] == "docker-compose.yml"
            assert result["services"] == ["web", "db"]

            # Verify command was constructed correctly
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert (
                "docker" in call_args and "compose" in call_args
            )  # Modern docker compose syntax
            assert "-p" in call_args
            assert "myproject" in call_args
            assert "up" in call_args
            assert "-d" in call_args
            assert "--build" in call_args

    @pytest.mark.asyncio
    async def test_compose_up_not_available(self, docker_tool):
        """Test Docker Compose operations when compose is not available."""
        docker_tool._docker_available = True
        docker_tool._compose_available = False

        with pytest.raises(ToolError, match="Docker Compose is not available"):
            await docker_tool._compose_up({})

    @pytest.mark.asyncio
    async def test_execute_action_unknown(self, docker_tool):
        """Test executing unknown action."""
        docker_tool._docker_available = True

        with pytest.raises(ToolError, match="Unknown action"):
            await docker_tool._execute_action("unknown_action", {})

    @pytest.mark.asyncio
    async def test_execute_action_docker_not_available(self, docker_tool):
        """Test executing action when Docker is not available."""
        docker_tool._docker_available = False

        with pytest.raises(ToolError, match="Docker is not available"):
            await docker_tool._execute_action("build_image", {})

    @pytest.mark.asyncio
    async def test_get_logs_success(self, docker_tool):
        """Test getting container logs."""
        with patch.object(docker_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "2023-01-01T00:00:00Z Starting application\n2023-01-01T00:00:01Z Ready",
                "stderr": "",
            }

            docker_tool._docker_available = True

            result = await docker_tool._get_logs(
                {
                    "container": "my-container",
                    "tail": 50,
                    "timestamps": True,
                }
            )

            assert result["success"] is True
            assert result["container"] == "my-container"
            assert "Starting application" in result["logs"]

    @pytest.mark.asyncio
    async def test_stop_container_success(self, docker_tool):
        """Test stopping container."""
        with patch.object(docker_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "my-container",
                "stderr": "",
            }

            docker_tool._docker_available = True

            result = await docker_tool._stop_container(
                {
                    "container": "my-container",
                    "timeout": 30,
                }
            )

            assert result["success"] is True
            assert result["container"] == "my-container"

    @pytest.mark.asyncio
    async def test_pull_image_success(self, docker_tool):
        """Test pulling Docker image."""
        with patch.object(docker_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "latest: Pulling from library/nginx\nPull complete",
                "stderr": "",
            }

            docker_tool._docker_available = True

            result = await docker_tool._pull_image(
                {
                    "image": "nginx",
                    "tag": "latest",
                }
            )

            assert result["success"] is True
            assert result["image"] == "nginx:latest"
            assert "Pull complete" in result["output"]

    @pytest.mark.asyncio
    async def test_run_command_error_handling(self, docker_tool):
        """Test command execution error handling."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_subprocess.side_effect = Exception("Process creation failed")

            result = await docker_tool._run_command(["docker", "--version"])

            assert result["returncode"] == -1
            assert "Process creation failed" in result["stderr"]

    @pytest.mark.asyncio
    async def test_get_supported_actions(self, docker_tool):
        """Test getting supported actions."""
        actions = await docker_tool._get_supported_actions()

        expected_actions = [
            "build_image",
            "run_container",
            "stop_container",
            "remove_container",
            "list_containers",
            "get_logs",
            "compose_up",
            "compose_down",
            "pull_image",
            "list_images",
        ]

        assert all(action in actions for action in expected_actions)

    @pytest.mark.asyncio
    async def test_full_workflow_integration(self, docker_tool):
        """Test a complete Docker workflow."""
        with patch.object(docker_tool, "_run_command") as mock_run:
            # Mock successful responses for each step
            mock_run.side_effect = [
                # Docker version check
                {"returncode": 0, "stdout": "Docker version 20.10.0", "stderr": ""},
                # Docker Compose version check
                {
                    "returncode": 0,
                    "stdout": "docker-compose version 1.29.0",
                    "stderr": "",
                },
                # Pull image
                {"returncode": 0, "stdout": "Pull complete", "stderr": ""},
                # Run container
                {"returncode": 0, "stdout": "container_123", "stderr": ""},
                # Get logs
                {"returncode": 0, "stdout": "Application started", "stderr": ""},
                # Stop container
                {"returncode": 0, "stdout": "container_123", "stderr": ""},
            ]

            # Initialize the tool
            await docker_tool.initialize()
            assert docker_tool._docker_available is True

            # Pull an image
            pull_result = await docker_tool.execute("pull_image", {"image": "nginx"})
            assert pull_result.success is True

            # Run a container
            run_result = await docker_tool.execute(
                "run_container",
                {
                    "image": "nginx:latest",
                    "name": "test-nginx",
                    "ports": ["8080:80"],
                },
            )
            assert run_result.success is True

            # Get logs
            logs_result = await docker_tool.execute(
                "get_logs",
                {
                    "container": "test-nginx",
                    "tail": 10,
                },
            )
            assert logs_result.success is True

            # Stop container
            stop_result = await docker_tool.execute(
                "stop_container",
                {
                    "container": "test-nginx",
                },
            )
            assert stop_result.success is True
