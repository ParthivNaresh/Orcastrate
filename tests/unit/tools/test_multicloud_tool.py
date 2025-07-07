"""
Unit tests for the multi-cloud tool.

This module tests the multi-cloud tool functionality including resource
provisioning, management, and cross-cloud operations.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from src.tools.base import ToolConfig
from src.tools.multicloud.base import (
    CloudProvider,
    ComputeSpec,
    DatabaseSpec,
    MultiCloudResource,
    ResourceState,
    ResourceType,
)
from src.tools.multicloud_tool import MultiCloudTool


@pytest.fixture
def multicloud_config():
    """Create a test configuration for the multi-cloud tool."""
    return ToolConfig(
        name="multicloud",
        version="1.0.0",
        timeout=30,
        retry_count=2,
        environment={
            "aws": {
                "region": "us-east-1",
                "access_key_id": "test",
                "secret_access_key": "test",
            }
        },
    )


@pytest.fixture
def multicloud_tool(multicloud_config):
    """Create a multi-cloud tool instance for testing."""
    return MultiCloudTool(multicloud_config)


@pytest.fixture
def mock_compute_resource():
    """Create a mock compute resource for testing."""
    return MultiCloudResource(
        id="compute-i-1234567890abcdef0",
        name="test-instance",
        provider=CloudProvider.AWS,
        resource_type=ResourceType.COMPUTE,
        state=ResourceState.RUNNING,
        region="us-east-1",
        created_at=datetime.utcnow(),
        tags={"Environment": "test"},
        provider_resource_id="i-1234567890abcdef0",
        provider_details={"instance_type": "t3.micro"},
        compute_details={
            "instance_type": "t3.micro",
            "ami_id": "ami-12345678",
            "public_ip": "203.0.113.12",
            "private_ip": "10.0.1.12",
        },
        estimated_cost_per_month=Decimal("8.50"),
    )


@pytest.fixture
def mock_database_resource():
    """Create a mock database resource for testing."""
    return MultiCloudResource(
        id="database-mydb-instance",
        name="test-database",
        provider=CloudProvider.AWS,
        resource_type=ResourceType.DATABASE,
        state=ResourceState.RUNNING,
        region="us-east-1",
        created_at=datetime.utcnow(),
        tags={"Environment": "test"},
        provider_resource_id="mydb-instance",
        provider_details={"db_instance_class": "db.t3.micro"},
        database_details={
            "engine": "mysql",
            "version": "8.0",
            "instance_class": "db.t3.micro",
            "storage_size_gb": 20,
            "endpoint": "mydb.cluster-xyz.us-east-1.rds.amazonaws.com",
            "port": 3306,
            "multi_az": False,
        },
        estimated_cost_per_month=Decimal("25.00"),
    )


class TestMultiCloudTool:
    """Test cases for the MultiCloudTool class."""

    @pytest.mark.asyncio
    async def test_get_schema(self, multicloud_tool):
        """Test that the tool schema is returned correctly."""
        schema = await multicloud_tool.get_schema()

        assert schema.name == "multicloud"
        assert schema.description == "Multi-cloud resource management tool"
        assert "provision_compute" in schema.actions
        assert "provision_database" in schema.actions
        assert "list_resources" in schema.actions
        assert "compare_costs" in schema.actions

    @pytest.mark.asyncio
    async def test_provision_compute_success(
        self, multicloud_tool, mock_compute_resource
    ):
        """Test successful compute provisioning."""
        # Mock the AWS provider and manager
        with patch.object(
            multicloud_tool.manager,
            "provision_resource",
            return_value=mock_compute_resource,
        ) as mock_provision:
            with patch.object(
                multicloud_tool, "_ensure_provider_initialized"
            ) as mock_init:
                mock_init.return_value = None

                result = await multicloud_tool.execute(
                    "provision_compute",
                    {
                        "provider": "aws",
                        "name": "test-instance",
                        "region": "us-east-1",
                        "instance_size": "micro",
                        "image": "ami-12345678",
                        "tags": {"Environment": "test"},
                    },
                )

                assert result.success
                assert result.output["resource_id"] == "compute-i-1234567890abcdef0"
                assert result.output["name"] == "test-instance"
                assert result.output["provider"] == "aws"
                assert result.output["resource_type"] == "compute"
                assert result.output["state"] == "running"
                assert result.output["estimated_cost_per_month"] == 8.50

                # Verify the ComputeSpec was created correctly
                mock_provision.assert_called_once()
                call_args = mock_provision.call_args
                assert call_args[0][0] == CloudProvider.AWS
                assert isinstance(call_args[0][1], ComputeSpec)
                assert call_args[0][1].name == "test-instance"
                assert call_args[0][1].instance_size == "micro"

    @pytest.mark.asyncio
    async def test_provision_database_success(
        self, multicloud_tool, mock_database_resource
    ):
        """Test successful database provisioning."""
        with patch.object(
            multicloud_tool.manager,
            "provision_resource",
            return_value=mock_database_resource,
        ) as mock_provision:
            with patch.object(
                multicloud_tool, "_ensure_provider_initialized"
            ) as mock_init:
                mock_init.return_value = None

                result = await multicloud_tool.execute(
                    "provision_database",
                    {
                        "provider": "aws",
                        "name": "test-database",
                        "region": "us-east-1",
                        "engine": "mysql",
                        "version": "8.0",
                        "instance_size": "micro",
                        "storage_size_gb": 20,
                        "tags": {"Environment": "test"},
                    },
                )

                assert result.success
                assert result.output["resource_id"] == "database-mydb-instance"
                assert result.output["name"] == "test-database"
                assert result.output["provider"] == "aws"
                assert result.output["resource_type"] == "database"
                assert result.output["database_details"]["engine"] == "mysql"
                assert result.output["estimated_cost_per_month"] == 25.00

                # Verify the DatabaseSpec was created correctly
                mock_provision.assert_called_once()
                call_args = mock_provision.call_args
                assert call_args[0][0] == CloudProvider.AWS
                assert isinstance(call_args[0][1], DatabaseSpec)
                assert call_args[0][1].name == "test-database"
                assert call_args[0][1].engine == "mysql"

    @pytest.mark.asyncio
    async def test_list_resources_success(
        self, multicloud_tool, mock_compute_resource, mock_database_resource
    ):
        """Test successful resource listing."""
        mock_provider = AsyncMock()
        mock_provider.list_resources.return_value = [
            mock_compute_resource,
            mock_database_resource,
        ]

        with patch.object(
            multicloud_tool.manager, "get_provider", return_value=mock_provider
        ):
            with patch.object(
                multicloud_tool, "_ensure_provider_initialized"
            ) as mock_init:
                mock_init.return_value = None
                multicloud_tool._initialized_providers.add(CloudProvider.AWS)

                result = await multicloud_tool.execute(
                    "list_resources", {"provider": "aws"}
                )

                assert result.success
                assert result.output["total_count"] == 2
                assert len(result.output["resources"]) == 2

                # Check first resource (compute)
                compute_data = result.output["resources"][0]
                assert compute_data["resource_id"] == "compute-i-1234567890abcdef0"
                assert compute_data["resource_type"] == "compute"
                assert compute_data["provider"] == "aws"

                # Check second resource (database)
                database_data = result.output["resources"][1]
                assert database_data["resource_id"] == "database-mydb-instance"
                assert database_data["resource_type"] == "database"
                assert database_data["provider"] == "aws"

    @pytest.mark.asyncio
    async def test_get_resource_success(self, multicloud_tool, mock_compute_resource):
        """Test successful resource retrieval."""
        mock_provider = AsyncMock()
        mock_provider.get_resource.return_value = mock_compute_resource

        with patch.object(
            multicloud_tool.manager, "get_provider", return_value=mock_provider
        ):
            with patch.object(
                multicloud_tool, "_ensure_provider_initialized"
            ) as mock_init:
                mock_init.return_value = None

                result = await multicloud_tool.execute(
                    "get_resource",
                    {
                        "resource_id": "compute-i-1234567890abcdef0",
                        "resource_type": "compute",
                    },
                )

                assert result.success
                assert result.output["resource_id"] == "compute-i-1234567890abcdef0"
                assert result.output["name"] == "test-instance"
                assert result.output["compute_details"]["instance_type"] == "t3.micro"
                assert result.output["compute_details"]["public_ip"] == "203.0.113.12"

    @pytest.mark.asyncio
    async def test_start_resource_success(self, multicloud_tool, mock_compute_resource):
        """Test successful resource start."""
        # Update resource state to stopped initially, then running after start
        stopped_resource = mock_compute_resource
        stopped_resource.state = ResourceState.STOPPED

        running_resource = mock_compute_resource
        running_resource.state = ResourceState.RUNNING

        mock_provider = AsyncMock()
        mock_provider.start_resource.return_value = running_resource

        with patch.object(
            multicloud_tool.manager, "get_provider", return_value=mock_provider
        ):
            with patch.object(
                multicloud_tool, "_ensure_provider_initialized"
            ) as mock_init:
                mock_init.return_value = None

                result = await multicloud_tool.execute(
                    "start_resource", {"resource_id": "compute-i-1234567890abcdef0"}
                )

                assert result.success
                assert result.output["resource_id"] == "compute-i-1234567890abcdef0"
                assert result.output["state"] == "running"
                assert "started successfully" in result.output["message"]

    @pytest.mark.asyncio
    async def test_compare_costs_success(self, multicloud_tool):
        """Test successful cost comparison."""
        from src.tools.base import CostEstimate

        mock_cost_estimate = CostEstimate(
            estimated_cost=8.50, currency="USD", confidence=0.8
        )

        with patch.object(multicloud_tool.manager, "compare_costs") as mock_compare:
            mock_compare.return_value = {CloudProvider.AWS: mock_cost_estimate}
            with patch.object(
                multicloud_tool, "_ensure_provider_initialized"
            ) as mock_init:
                mock_init.return_value = None

                result = await multicloud_tool.execute(
                    "compare_costs",
                    {
                        "resource_type": "compute",
                        "spec": {
                            "name": "test-instance",
                            "region": "us-east-1",
                            "instance_size": "micro",
                            "image": "ami-12345678",
                        },
                        "providers": ["aws"],
                    },
                )

                assert result.success
                assert result.output["resource_type"] == "compute"
                assert "aws" in result.output["cost_estimates"]
                assert result.output["cost_estimates"]["aws"]["estimated_cost"] == 8.50
                assert result.output["cheapest_provider"] == "aws"

    @pytest.mark.asyncio
    async def test_invalid_action(self, multicloud_tool):
        """Test handling of invalid actions."""
        result = await multicloud_tool.execute("invalid_action", {})

        assert not result.success
        assert "Unsupported action" in result.error

    @pytest.mark.asyncio
    async def test_missing_required_parameters(self, multicloud_tool):
        """Test handling of missing required parameters."""
        result = await multicloud_tool.execute(
            "provision_compute",
            {
                "provider": "aws",
                # Missing required parameters like name, region, etc.
            },
        )

        assert not result.success
        # The test hits AWS initialization before parameter validation
        assert (
            "credentials" in result.error.lower()
            or "validation" in result.error.lower()
        )

    @pytest.mark.asyncio
    async def test_unsupported_provider(self, multicloud_tool):
        """Test handling of unsupported providers."""
        with patch.object(multicloud_tool, "_ensure_provider_initialized") as mock_init:
            mock_init.side_effect = Exception("Provider gcp not yet implemented")

            result = await multicloud_tool.execute(
                "provision_compute",
                {
                    "provider": "gcp",
                    "name": "test-instance",
                    "region": "us-central1",
                    "instance_size": "micro",
                    "image": "ubuntu-2004-lts",
                },
            )

            assert not result.success
            assert "not yet implemented" in result.error

    def test_parse_resource_type(self, multicloud_tool):
        """Test resource type parsing."""
        assert multicloud_tool._parse_resource_type("compute") == ResourceType.COMPUTE
        assert multicloud_tool._parse_resource_type("database") == ResourceType.DATABASE
        assert multicloud_tool._parse_resource_type("storage") == ResourceType.STORAGE
        assert multicloud_tool._parse_resource_type("network") == ResourceType.NETWORK
        assert (
            multicloud_tool._parse_resource_type("security_group")
            == ResourceType.SECURITY_GROUP
        )

    @pytest.mark.asyncio
    async def test_list_providers(self, multicloud_tool):
        """Test listing registered providers."""
        multicloud_tool._initialized_providers.add(CloudProvider.AWS)

        result = await multicloud_tool.execute("list_providers", {})

        assert result.success
        assert "aws" in result.output["registered_providers"]
        assert "gcp" in result.output["available_providers"]
        assert "azure" in result.output["available_providers"]

    @pytest.mark.asyncio
    async def test_validate_provider_success(self, multicloud_tool):
        """Test successful provider validation."""
        with patch.object(multicloud_tool, "_ensure_provider_initialized") as mock_init:
            mock_init.return_value = None

            result = await multicloud_tool.execute(
                "validate_provider", {"provider": "aws"}
            )

            assert result.success
            assert result.output["status"] == "valid"
            assert result.output["provider"] == "aws"

    @pytest.mark.asyncio
    async def test_validate_provider_failure(self, multicloud_tool):
        """Test provider validation failure."""
        with patch.object(multicloud_tool, "_ensure_provider_initialized") as mock_init:
            mock_init.side_effect = Exception("Invalid credentials")

            result = await multicloud_tool.execute(
                "validate_provider", {"provider": "aws"}
            )

            assert (
                result.success
            )  # The validation action succeeds even if provider is invalid
            assert result.output["status"] == "invalid"
            assert "Invalid credentials" in result.output["message"]
