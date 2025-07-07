"""
Configuration and fixtures for live integration tests.

This module provides shared fixtures for tests that run against real services
like LocalStack, Docker, databases, etc.
"""

import asyncio
import os
import time
from typing import AsyncGenerator, Generator

import pytest

# Optional imports for live testing
try:
    import boto3
    import redis
    from pymongo import MongoClient
    from testcontainers.compose import DockerCompose

    import docker

    LIVE_DEPS_AVAILABLE = True
except ImportError as e:
    print(f"Live test dependencies not available: {e}")
    LIVE_DEPS_AVAILABLE = False

    # Create stub classes to avoid NameError
    class DockerCompose:
        pass

    class redis:
        class Redis:
            pass

    class MongoClient:
        pass


from src.tools.aws import AWSCloudTool
from src.tools.base import ToolConfig
from src.tools.docker import DockerTool

# Test environment configuration
TEST_LOCALSTACK_ENDPOINT = os.getenv("LOCALSTACK_ENDPOINT", "http://localhost:4566")
TEST_POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
TEST_MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
TEST_REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
TEST_MONGODB_HOST = os.getenv("MONGODB_HOST", "localhost")


@pytest.fixture(scope="session")
def docker_compose_file():
    """Path to docker-compose file for tests."""
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "docker-compose.test.yml"
    )


@pytest.fixture(scope="session")
def test_infrastructure(docker_compose_file) -> Generator[str, None, None]:
    """
    Connect to existing test infrastructure.

    This fixture assumes the test infrastructure is already running
    (started externally) and just validates connectivity.
    """
    if not LIVE_DEPS_AVAILABLE:
        pytest.skip("Live test dependencies not available")

    # Wait for services to be ready
    max_retries = 10
    retry_delay = 2

    services_to_wait = [
        ("LocalStack", f"{TEST_LOCALSTACK_ENDPOINT}/_localstack/health"),
        ("PostgreSQL", f"tcp://{TEST_POSTGRES_HOST}:5432"),
        ("MySQL", f"tcp://{TEST_MYSQL_HOST}:3306"),
        ("Redis", f"tcp://{TEST_REDIS_HOST}:6379"),
        ("MongoDB", f"tcp://{TEST_MONGODB_HOST}:27017"),
    ]

    for service_name, endpoint in services_to_wait:
        for attempt in range(max_retries):
            try:
                if endpoint.startswith("http"):
                    import urllib.request

                    urllib.request.urlopen(endpoint, timeout=5)
                else:
                    # For TCP endpoints, use simple socket connection
                    import socket

                    host, port = endpoint.replace("tcp://", "").split(":")
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((host, int(port)))
                    sock.close()
                    if result != 0:
                        raise ConnectionError(f"Cannot connect to {endpoint}")

                print(f"✓ {service_name} is ready")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"❌ {service_name} failed to connect after {max_retries} attempts: {e}"
                    )
                print(
                    f"⏳ Waiting for {service_name}... (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_delay)

    print("🚀 All test infrastructure services are ready!")
    yield "infrastructure_ready"


@pytest.fixture
def aws_live_config() -> ToolConfig:
    """Configuration for AWS tool using LocalStack."""
    return ToolConfig(
        name="aws",
        version="1.0.0",
        timeout=60,
        max_retries=3,
        environment={
            "AWS_ENDPOINT_URL": TEST_LOCALSTACK_ENDPOINT,
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "AWS_DEFAULT_REGION": "us-east-1",
        },
    )


@pytest.fixture
async def aws_live_tool(
    test_infrastructure, aws_live_config
) -> AsyncGenerator[AWSCloudTool, None]:
    """AWS tool instance configured for LocalStack testing."""
    tool = AWSCloudTool(aws_live_config)

    # Override the client creation to use LocalStack endpoint
    tool._create_client

    async def _create_localstack_client():
        try:
            # Create AWS session with LocalStack endpoint
            tool._session = boto3.Session(
                region_name="us-east-1",
                aws_access_key_id="test",
                aws_secret_access_key="test",
            )

            # Initialize clients with LocalStack endpoint
            tool._clients = {
                "ec2": tool._session.client(
                    "ec2", endpoint_url=TEST_LOCALSTACK_ENDPOINT
                ),
                "rds": tool._session.client(
                    "rds", endpoint_url=TEST_LOCALSTACK_ENDPOINT
                ),
                "lambda": tool._session.client(
                    "lambda", endpoint_url=TEST_LOCALSTACK_ENDPOINT
                ),
                "iam": tool._session.client(
                    "iam", endpoint_url=TEST_LOCALSTACK_ENDPOINT
                ),
                "sts": tool._session.client(
                    "sts", endpoint_url=TEST_LOCALSTACK_ENDPOINT
                ),
                "s3": tool._session.client("s3", endpoint_url=TEST_LOCALSTACK_ENDPOINT),
            }

            # Test connection
            sts_client = tool._clients["sts"]
            identity = sts_client.get_caller_identity()
            tool.logger.info(
                f"LocalStack connection established for account: {identity.get('Account')}"
            )

            return tool._session

        except Exception as e:
            raise RuntimeError(f"Failed to initialize LocalStack session: {e}")

    tool._create_client = _create_localstack_client

    await tool.initialize()
    yield tool
    await tool.cleanup()


@pytest.fixture
def docker_live_config() -> ToolConfig:
    """Configuration for Docker tool using real Docker daemon."""
    return ToolConfig(
        name="docker",
        version="1.0.0",
        timeout=60,
        max_retries=3,
    )


@pytest.fixture
async def docker_live_tool(docker_live_config) -> AsyncGenerator[DockerTool, None]:
    """Docker tool instance configured for real Docker daemon testing."""
    tool = DockerTool(docker_live_config)
    await tool.initialize()
    yield tool
    await tool.cleanup()


@pytest.fixture
def localstack_boto3_client():
    """Direct boto3 client for LocalStack (for verification)."""
    return boto3.client(
        "ec2",
        endpoint_url=TEST_LOCALSTACK_ENDPOINT,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )


@pytest.fixture
def docker_client() -> Generator[docker.DockerClient, None, None]:
    """Direct Docker client for verification."""
    client = docker.from_env()
    yield client
    client.close()


@pytest.fixture
def redis_client(test_infrastructure) -> Generator[redis.Redis, None, None]:
    """Redis client for database testing."""
    client = redis.Redis(host=TEST_REDIS_HOST, port=6379, decode_responses=True)
    yield client
    client.close()


@pytest.fixture
def mongodb_client(test_infrastructure) -> Generator[MongoClient, None, None]:
    """MongoDB client for NoSQL database testing."""
    client = MongoClient(
        f"mongodb://test_user:test_password@{TEST_MONGODB_HOST}:27017/"
    )
    yield client
    client.close()


@pytest.fixture(autouse=True)
def cleanup_test_resources():
    """Auto-cleanup fixture that runs after each test."""
    # Store resources created during test
    created_resources = {
        "ec2_instances": [],
        "security_groups": [],
        "iam_roles": [],
        "docker_containers": [],
        "docker_images": [],
    }

    yield created_resources

    # Cleanup would happen here in a real implementation
    # For now, we rely on container restarts between test runs


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to handle live test markers.

    This automatically skips live tests unless specifically requested.
    """
    # Skip live tests if dependencies are not available
    if not LIVE_DEPS_AVAILABLE:
        skip_deps = pytest.mark.skip(reason="Live test dependencies not installed")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_deps)
        return

    if config.getoption("--live"):
        # If --live flag is provided, run live tests
        return

    # Otherwise, skip live tests by default
    skip_live = pytest.mark.skip(reason="Live tests require --live flag")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


def pytest_addoption(parser):
    """Add command line options for live testing."""
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run live integration tests against real services",
    )
    parser.addoption(
        "--infrastructure-timeout",
        action="store",
        default=300,
        type=int,
        help="Timeout in seconds for infrastructure startup",
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Utility functions for tests
def wait_for_resource_state(
    client, resource_type: str, resource_id: str, desired_state: str, timeout: int = 60
) -> bool:
    """
    Wait for an AWS resource to reach a desired state.

    Args:
        client: Boto3 client
        resource_type: Type of resource (e.g., 'instance', 'db_instance')
        resource_id: ID of the resource
        desired_state: State to wait for
        timeout: Maximum time to wait in seconds

    Returns:
        True if resource reached desired state, False if timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            if resource_type == "instance":
                response = client.describe_instances(InstanceIds=[resource_id])
                current_state = response["Reservations"][0]["Instances"][0]["State"][
                    "Name"
                ]
            elif resource_type == "db_instance":
                response = client.describe_db_instances(
                    DBInstanceIdentifier=resource_id
                )
                current_state = response["DBInstances"][0]["DBInstanceStatus"]
            else:
                raise ValueError(f"Unknown resource type: {resource_type}")

            if current_state == desired_state:
                return True

            time.sleep(2)

        except Exception as e:
            print(f"Error checking resource state: {e}")
            time.sleep(2)

    return False


def generate_unique_name(prefix: str) -> str:
    """Generate a unique name for test resources."""
    import uuid

    return f"{prefix}-{uuid.uuid4().hex[:8]}"
