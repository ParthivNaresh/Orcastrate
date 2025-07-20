"""
Tool-specific fixtures for unit testing.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from src.tools.base import ToolConfig


@pytest.fixture
def docker_config() -> ToolConfig:
    """Create Docker tool configuration for testing."""
    return ToolConfig(
        name="docker",
        version="1.0.0",
        timeout=300,
        retry_count=3,
        environment={
            "docker_host": "unix:///var/run/docker.sock",
            "registry_url": "registry.example.com",
            "build_context": "/tmp/build",
        },
    )


@pytest.fixture
def git_config() -> ToolConfig:
    """Create Git tool configuration for testing."""
    return ToolConfig(
        name="git",
        version="1.0.0",
        timeout=300,
        retry_count=3,
        environment={
            "git_user": "test_user",
            "git_email": "test@example.com",
            "default_branch": "main",
        },
    )


@pytest.fixture
def filesystem_config() -> ToolConfig:
    """Create Filesystem tool configuration for testing."""
    return ToolConfig(
        name="filesystem",
        version="1.0.0",
        timeout=300,
        retry_count=3,
        environment={
            "base_path": "/tmp/test",
            "max_file_size": 10485760,  # 10MB
            "allowed_extensions": [".txt", ".json", ".py"],
        },
    )


@pytest.fixture
def terraform_config() -> ToolConfig:
    """Create Terraform tool configuration for testing."""
    return ToolConfig(
        name="terraform",
        version="1.0.0",
        timeout=300,
        retry_count=3,
        environment={
            "working_dir": None,  # Will be set to temp directory
            "terraform_version": "latest",
            "backend_config": {},
            "var_files": [],
            "variables": {},
            "parallelism": 10,
            "auto_approve": False,
        },
    )


@pytest.fixture
def aws_config() -> ToolConfig:
    """Create AWS tool configuration for testing."""
    return ToolConfig(
        name="aws",
        version="1.0.0",
        timeout=300,
        retry_count=3,
        environment={
            "region": "us-east-1",
            "profile": "default",
            "endpoint_url": None,
            "cost_tracking_enabled": True,
            "default_tags": {"Environment": "test", "Project": "orcastrate"},
        },
    )


@pytest.fixture
def postgresql_config(database_config) -> ToolConfig:
    """Create PostgreSQL tool configuration for testing."""
    return ToolConfig(
        name="postgresql",
        version="1.0.0",
        timeout=300,
        retry_count=3,
        environment=database_config,
    )


@pytest.fixture
def mysql_config(mysql_config) -> ToolConfig:
    """Create MySQL tool configuration for testing."""
    return ToolConfig(
        name="mysql",
        version="1.0.0",
        timeout=300,
        retry_count=3,
        environment=mysql_config,
    )


@pytest.fixture
def mongodb_config(mongodb_config) -> ToolConfig:
    """Create MongoDB tool configuration for testing."""
    return ToolConfig(
        name="mongodb",
        version="1.0.0",
        timeout=300,
        retry_count=3,
        environment=mongodb_config,
    )


@pytest.fixture
def redis_config(redis_config) -> ToolConfig:
    """Create Redis tool configuration for testing."""
    return ToolConfig(
        name="redis",
        version="1.0.0",
        timeout=300,
        retry_count=3,
        environment=redis_config,
    )


@pytest.fixture
def tool_temp_dir():
    """Create a temporary directory for tool testing."""
    with tempfile.TemporaryDirectory(prefix="orcastrate_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_docker_client():
    """Create a mock Docker client for testing."""
    client = AsyncMock()
    client.images = AsyncMock()
    client.containers = AsyncMock()
    client.networks = AsyncMock()
    client.volumes = AsyncMock()
    client.version = AsyncMock(return_value={"Version": "20.10.0"})
    client.ping = AsyncMock(return_value=True)
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_git_repo():
    """Create a mock Git repository for testing."""
    repo = Mock()
    repo.init = Mock()
    repo.clone_from = Mock()
    repo.create_head = Mock()
    repo.heads = Mock()
    repo.remotes = Mock()
    repo.index = Mock()
    repo.git = Mock()
    return repo


@pytest.fixture
def mock_terraform_cli():
    """Create a mock Terraform CLI for testing."""
    cli = AsyncMock()
    cli.init = AsyncMock()
    cli.plan = AsyncMock()
    cli.apply = AsyncMock()
    cli.destroy = AsyncMock()
    cli.validate = AsyncMock()
    cli.version = AsyncMock(return_value="1.0.0")
    return cli


@pytest.fixture
def mock_boto3_client():
    """Create a mock boto3 client for testing."""
    client = Mock()
    client.create_vpc = Mock()
    client.create_subnet = Mock()
    client.create_security_group = Mock()
    client.run_instances = Mock()
    client.create_db_instance = Mock()
    client.create_load_balancer = Mock()
    client.describe_vpcs = Mock()
    client.describe_instances = Mock()
    client.describe_db_instances = Mock()
    return client


@pytest.fixture
def mock_resource_response():
    """Create a mock AWS resource response for testing."""
    return {
        "resource_id": "test-resource-123",
        "state": "available",
        "tags": {"Name": "test-resource", "Environment": "test"},
        "created_at": "2023-01-01T00:00:00Z",
        "cost_estimate": 50.0,
    }


@pytest.fixture
def sample_terraform_config():
    """Create sample Terraform configuration for testing."""
    return {
        "terraform": {
            "required_version": ">= 1.0",
            "required_providers": {
                "aws": {"source": "hashicorp/aws", "version": "~> 5.0"}
            },
        },
        "provider": {"aws": {"region": "us-east-1"}},
        "resource": {
            "aws_vpc": {
                "test_vpc": {"cidr_block": "10.0.0.0/16", "tags": {"Name": "test-vpc"}}
            }
        },
    }
