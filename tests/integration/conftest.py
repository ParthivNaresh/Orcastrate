"""
Integration test specific fixtures and configurations.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from src.agent.base import Requirements
from src.executors.base import ExecutionStrategy, ExecutorConfig
from src.executors.concrete_executor import ConcreteExecutor
from src.planners.base import PlannerConfig, PlanningStrategy
from src.planners.template_planner import TemplatePlanner


@pytest.fixture
def temp_env_file():
    """Create a temporary environment file for integration testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("TEST_VAR=test_value\n")
        f.write("DB_HOST=localhost\n")
        f.write("DB_PORT=5432\n")
        temp_file = Path(f.name)

    yield temp_file

    # Cleanup
    if temp_file.exists():
        temp_file.unlink()


@pytest.fixture
def integration_temp_dir():
    """Create a temporary directory for integration testing."""
    with tempfile.TemporaryDirectory(prefix="orcastrate_integration_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def integration_planner_config() -> PlannerConfig:
    """Create planner configuration for integration testing."""
    return PlannerConfig(
        strategy=PlanningStrategy.TEMPLATE_MATCHING,
        max_plan_steps=20,
        max_planning_time=60,
        cost_optimization=True,
        risk_threshold=0.8,
        parallel_execution=False,  # Sequential for integration tests
    )


@pytest.fixture
def integration_executor_config() -> ExecutorConfig:
    """Create executor configuration for integration testing."""
    return ExecutorConfig(
        strategy=ExecutionStrategy.SEQUENTIAL,
        max_concurrent_steps=3,
        step_timeout=120,
        retry_policy={"max_retries": 2, "backoff_factor": 1.5, "max_delay": 30},
        enable_rollback=True,
    )


@pytest.fixture
async def integration_planner(integration_planner_config, mock_progress_tracker):
    """Create a real planner instance for integration testing."""
    planner = TemplatePlanner(
        integration_planner_config, progress_tracker=mock_progress_tracker
    )
    await planner.initialize()
    return planner


@pytest.fixture
async def integration_executor(integration_executor_config, mock_progress_tracker):
    """Create a real executor with mocked external dependencies for integration testing."""
    executor = ConcreteExecutor(
        integration_executor_config, progress_tracker=mock_progress_tracker
    )
    # Note: Not calling initialize() to avoid setting up real tools
    return executor


@pytest.fixture
def integration_requirements() -> Requirements:
    """Create requirements for integration testing."""
    return Requirements(
        description="Integration test application with multiple components",
        framework="fastapi",
        database="postgresql",
        cloud_provider="aws",
        scaling_requirements={"min_instances": 1, "max_instances": 3},
        security_requirements={"encryption": True, "authentication": "basic"},
        budget_constraints={"max_monthly_cost": 200.0},
        metadata={
            "test_mode": True,
            "integration_test": True,
            "timeout": 300,
        },
    )


@pytest.fixture
def multicloud_integration_config():
    """Create multi-cloud configuration for integration testing."""
    return {
        "providers": {
            "aws": {
                "region": "us-east-1",
                "profile": "test",
                "endpoint_url": "http://localhost:4566",  # LocalStack
            },
            "gcp": {
                "project": "test-project",
                "region": "us-central1",
                "credentials_path": None,
            },
            "azure": {
                "subscription_id": "test-subscription",
                "resource_group": "test-rg",
                "location": "eastus",
            },
        },
        "cost_tracking": {
            "enabled": True,
            "currency": "USD",
            "budget_alerts": True,
        },
        "security": {
            "encryption_at_rest": True,
            "encryption_in_transit": True,
            "access_logging": True,
        },
    }


@pytest.fixture
def sample_docker_compose():
    """Create sample docker-compose configuration for testing."""
    return {
        "version": "3.8",
        "services": {
            "app": {
                "build": ".",
                "ports": ["8000:8000"],
                "environment": ["DATABASE_URL=postgresql://user:pass@db:5432/testdb"],
                "depends_on": ["db"],
            },
            "db": {
                "image": "postgres:13",
                "environment": [
                    "POSTGRES_DB=testdb",
                    "POSTGRES_USER=user",
                    "POSTGRES_PASSWORD=pass",
                ],
                "volumes": ["postgres_data:/var/lib/postgresql/data"],
                "ports": ["5432:5432"],
            },
        },
        "volumes": {"postgres_data": {}},
    }


@pytest.fixture
def mock_external_service():
    """Create mock external service for integration testing."""
    service = Mock()
    service.health_check = AsyncMock(return_value={"status": "healthy"})
    service.authenticate = AsyncMock(return_value={"token": "test-token"})
    service.create_resource = AsyncMock(return_value={"id": "resource-123"})
    service.get_resource = AsyncMock(
        return_value={"id": "resource-123", "status": "active"}
    )
    service.delete_resource = AsyncMock(return_value={"status": "deleted"})
    return service


@pytest.fixture
def complex_plan_requirements():
    """Create complex requirements for comprehensive integration testing."""
    return Requirements(
        description="Complex multi-tier application with microservices, databases, caching, and monitoring",
        framework="fastapi",
        database="postgresql",
        cloud_provider="aws",
        scaling_requirements={
            "min_instances": 2,
            "max_instances": 10,
            "auto_scaling": True,
            "target_cpu_utilization": 70,
        },
        security_requirements={
            "encryption": True,
            "authentication": "oauth2",
            "authorization": "rbac",
            "ssl_termination": True,
            "waf_enabled": True,
        },
        budget_constraints={
            "max_monthly_cost": 1000.0,
            "cost_alerts": True,
            "cost_optimization": True,
        },
        metadata={
            "environment": "integration",
            "monitoring": True,
            "logging": True,
            "backup_retention": 30,
            "multi_az": True,
            "caching_enabled": True,
            "cdn_enabled": True,
        },
    )
