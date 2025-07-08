"""
Integration tests for the multi-cloud tool with mocked cloud providers.

This module tests the integration between the multi-cloud abstraction layer
and the underlying cloud provider tools using mocked services.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.base import ToolConfig
from src.tools.multicloud_tool import MultiCloudTool


@pytest.fixture
def multicloud_integration_config():
    """Create a test configuration for multi-cloud integration testing."""
    return ToolConfig(
        name="multicloud",
        version="1.0.0",
        timeout=60,
        retry_count=3,
        environment={
            "aws": {
                "region": "us-east-1",
                "access_key_id": "test_key",
                "secret_access_key": "test_secret",
            }
        },
    )


@pytest.fixture
def multicloud_tool_integration(multicloud_integration_config):
    """Create a multi-cloud tool instance for integration testing."""
    return MultiCloudTool(multicloud_integration_config)


class TestMultiCloudIntegration:
    """Integration test cases for the multi-cloud tool."""

    @pytest.mark.asyncio
    @patch("src.tools.multicloud.providers.aws_provider.AWSCloudTool")
    async def test_provision_compute_with_aws_integration(
        self, mock_aws_tool_class, multicloud_tool_integration
    ):
        """Test compute provisioning integrates correctly with AWS tool."""
        # Setup mock AWS tool
        mock_aws_tool = AsyncMock()
        mock_aws_tool_class.return_value = mock_aws_tool

        # Mock AWS tool responses
        mock_aws_tool.initialize.return_value = None

        def mock_execute_side_effect(action, params):
            if action == "create_ec2_instance":
                return MagicMock(
                    success=True,
                    output={
                        "instance_id": "i-1234567890abcdef0",
                        "instance_type": "t3.micro",
                        "state": "running",
                        "ami_id": "ami-12345678",
                        "public_ip_address": "203.0.113.12",
                        "private_ip_address": "10.0.1.12",
                        "launch_time": "2024-01-01T12:00:00Z",
                        "placement": {"availability_zone": "us-east-1a"},
                        "tags": {"Name": "test-multicloud-instance"},
                    },
                )
            elif action == "estimate_costs":
                return MagicMock(
                    success=True,
                    output={
                        "total_estimated_cost": 8.50,
                        "duration_hours": 720,
                        "currency": "USD",
                    },
                )
            return MagicMock(success=False, error="Unknown action")

        mock_aws_tool.execute.side_effect = mock_execute_side_effect

        # Test compute provisioning
        result = await multicloud_tool_integration.execute(
            "provision_compute",
            {
                "provider": "aws",
                "name": "test-multicloud-instance",
                "region": "us-east-1",
                "instance_size": "micro",
                "image": "ami-12345678",
                "tags": {"Environment": "integration-test"},
                "storage_size_gb": 20,
                "public_ip": True,
            },
        )

        # Verify the result
        assert result.success
        assert result.output["resource_id"] == "compute-i-1234567890abcdef0"
        assert result.output["name"] == "test-multicloud-instance"
        assert result.output["provider"] == "aws"
        assert result.output["resource_type"] == "compute"
        assert result.output["state"] == "running"
        assert result.output["provider_resource_id"] == "i-1234567890abcdef0"

        # Verify AWS tool was called correctly
        mock_aws_tool.initialize.assert_called_once()
        assert (
            mock_aws_tool.execute.call_count == 2
        )  # create_ec2_instance + estimate_costs

        # Check the AWS tool was called with correct parameters
        create_call = mock_aws_tool.execute.call_args_list[0]
        assert create_call[0][0] == "create_ec2_instance"
        aws_params = create_call[0][1]
        assert aws_params["instance_type"] == "t3.micro"
        assert aws_params["ami_id"] == "ami-12345678"
        assert aws_params["tags"]["Name"] == "test-multicloud-instance"

    @pytest.mark.asyncio
    @patch("src.tools.multicloud.providers.aws_provider.AWSCloudTool")
    async def test_provision_database_with_aws_integration(
        self, mock_aws_tool_class, multicloud_tool_integration
    ):
        """Test database provisioning integrates correctly with AWS tool."""
        # Setup mock AWS tool
        mock_aws_tool = AsyncMock()
        mock_aws_tool_class.return_value = mock_aws_tool

        # Mock AWS tool responses
        mock_aws_tool.initialize.return_value = None

        def mock_execute_side_effect(action, params):
            if action == "create_rds_instance":
                return MagicMock(
                    success=True,
                    output={
                        "db_instance_identifier": "test-multicloud-db",
                        "db_instance_class": "db.t4g.micro",
                        "engine": "mysql",
                        "engine_version": "8.0.35",
                        "db_instance_status": "available",
                        "allocated_storage": 20,
                        "endpoint": {
                            "address": "test-multicloud-db.cluster-xyz.us-east-1.rds.amazonaws.com",
                            "port": 3306,
                        },
                        "availability_zone": "us-east-1a",
                        "master_username": "admin",
                        "multi_az": False,
                        "instance_create_time": "2024-01-01T12:00:00Z",
                        "tags": {"Name": "test-multicloud-db"},
                    },
                )
            elif action == "estimate_costs":
                return MagicMock(success=True, output={"total_estimated_cost": 25.00})
            return MagicMock(success=False, error="Unknown action")

        mock_aws_tool.execute.side_effect = mock_execute_side_effect

        # Test database provisioning
        result = await multicloud_tool_integration.execute(
            "provision_database",
            {
                "provider": "aws",
                "name": "test-multicloud-db",
                "region": "us-east-1",
                "engine": "mysql",
                "version": "8.0.35",
                "instance_size": "micro",
                "storage_size_gb": 20,
                "backup_retention_days": 7,
                "multi_az": False,
                "tags": {"Environment": "integration-test"},
            },
        )

        # Verify the result
        assert result.success
        assert result.output["resource_id"] == "database-test-multicloud-db"
        assert result.output["name"] == "test-multicloud-db"
        assert result.output["provider"] == "aws"
        assert result.output["resource_type"] == "database"
        assert result.output["state"] == "running"  # "available" maps to "running"
        assert result.output["database_details"]["engine"] == "mysql"
        assert (
            result.output["database_details"]["endpoint"]
            == "test-multicloud-db.cluster-xyz.us-east-1.rds.amazonaws.com"
        )

        # Verify AWS tool was called correctly
        mock_aws_tool.initialize.assert_called_once()
        # AWS tool should be called twice: once for RDS creation, once for cost estimation
        assert mock_aws_tool.execute.call_count == 2

        # Check the AWS tool was called with correct parameters for RDS creation
        # The first call should be create_rds_instance
        first_call_args = mock_aws_tool.execute.call_args_list[0]
        assert first_call_args[0][0] == "create_rds_instance"
        aws_params = first_call_args[0][1]
        assert aws_params["engine"] == "mysql"
        assert aws_params["db_instance_class"] == "db.t4g.micro"
        assert aws_params["allocated_storage"] == 20

    @pytest.mark.asyncio
    @patch("src.tools.multicloud.providers.aws_provider.AWSCloudTool")
    async def test_list_resources_with_aws_integration(
        self, mock_aws_tool_class, multicloud_tool_integration
    ):
        """Test resource listing integrates correctly with AWS tool."""
        # Setup mock AWS tool
        mock_aws_tool = AsyncMock()
        mock_aws_tool_class.return_value = mock_aws_tool

        # Mock AWS tool responses for listing
        mock_aws_tool.initialize.return_value = None

        # Mock list_ec2_instances response
        def mock_execute_side_effect(action, params):
            if action == "list_ec2_instances":
                return MagicMock(
                    success=True,
                    output={
                        "instances": [
                            {
                                "instance_id": "i-1234567890abcdef0",
                                "instance_type": "t3.micro",
                                "state": "running",
                                "image_id": "ami-12345678",
                                "launch_time": "2024-01-01T12:00:00+00:00",
                                "placement": {"availability_zone": "us-east-1a"},
                                "tags": {"Name": "test-instance-1"},
                                "public_ip_address": "203.0.113.12",
                                "private_ip_address": "10.0.1.12",
                            },
                            {
                                "instance_id": "i-0987654321fedcba0",
                                "instance_type": "t3.small",
                                "state": "stopped",
                                "image_id": "ami-87654321",
                                "launch_time": "2024-01-01T13:00:00+00:00",
                                "placement": {"availability_zone": "us-east-1b"},
                                "tags": {"Name": "test-instance-2"},
                                "public_ip_address": None,
                                "private_ip_address": "10.0.2.12",
                            },
                        ]
                    },
                )
            elif action == "list_rds_instances":
                return MagicMock(
                    success=True,
                    output={
                        "db_instances": [
                            {
                                "db_instance_identifier": "test-db-1",
                                "db_instance_class": "db.t4g.micro",
                                "engine": "postgres",
                                "engine_version": "15.4",
                                "db_instance_status": "available",
                                "allocated_storage": 20,
                                "availability_zone": "us-east-1a",
                                "instance_create_time": "2024-01-01T12:00:00+00:00",
                                "endpoint": {
                                    "address": "test-db-1.cluster-xyz.us-east-1.rds.amazonaws.com",
                                    "port": 5432,
                                },
                                "tags": {"Name": "test-database"},
                            }
                        ]
                    },
                )
            return MagicMock(success=False, error="Unknown action")

        mock_aws_tool.execute.side_effect = mock_execute_side_effect

        # Initialize the provider first
        await multicloud_tool_integration._ensure_provider_initialized("aws")

        # Test resource listing
        result = await multicloud_tool_integration.execute(
            "list_resources", {"provider": "aws"}
        )

        # Verify the result
        assert result.success
        assert result.output["total_count"] == 3  # 2 compute + 1 database

        # Check compute resources
        compute_resources = [
            r for r in result.output["resources"] if r["resource_type"] == "compute"
        ]
        assert len(compute_resources) == 2

        running_instance = next(
            r
            for r in compute_resources
            if r["provider_resource_id"] == "i-1234567890abcdef0"
        )
        assert running_instance["state"] == "running"
        assert running_instance["name"] == "test-instance-1"

        stopped_instance = next(
            r
            for r in compute_resources
            if r["provider_resource_id"] == "i-0987654321fedcba0"
        )
        assert stopped_instance["state"] == "stopped"
        assert stopped_instance["name"] == "test-instance-2"

        # Check database resources
        database_resources = [
            r for r in result.output["resources"] if r["resource_type"] == "database"
        ]
        assert len(database_resources) == 1

        db_resource = database_resources[0]
        assert db_resource["provider_resource_id"] == "test-db-1"
        assert db_resource["state"] == "running"  # "available" maps to "running"
        assert db_resource["name"] == "test-database"

    @pytest.mark.asyncio
    @patch("src.tools.multicloud.providers.aws_provider.AWSCloudTool")
    async def test_resource_lifecycle_management(
        self, mock_aws_tool_class, multicloud_tool_integration
    ):
        """Test complete resource lifecycle: start, stop, terminate."""
        # Setup mock AWS tool
        mock_aws_tool = AsyncMock()
        mock_aws_tool_class.return_value = mock_aws_tool

        mock_aws_tool.initialize.return_value = None

        # Mock AWS tool responses for lifecycle operations
        def mock_execute_side_effect(action, params):
            if action == "start_ec2_instance":
                return MagicMock(
                    success=True,
                    output={
                        "instance_id": "i-1234567890abcdef0",
                        "previous_state": "stopped",
                        "current_state": "running",
                    },
                )
            elif action == "stop_ec2_instance":
                return MagicMock(
                    success=True,
                    output={
                        "instance_id": "i-1234567890abcdef0",
                        "previous_state": "running",
                        "current_state": "stopping",
                    },
                )
            elif action == "terminate_ec2_instance":
                return MagicMock(
                    success=True,
                    output={
                        "instance_id": "i-1234567890abcdef0",
                        "previous_state": "stopped",
                        "current_state": "shutting-down",
                    },
                )
            elif action == "list_ec2_instances":
                return MagicMock(
                    success=True,
                    output={
                        "instances": [
                            {
                                "instance_id": "i-1234567890abcdef0",
                                "instance_type": "t3.micro",
                                "state": "running",
                                "image_id": "ami-12345678",
                                "launch_time": "2024-01-01T12:00:00+00:00",
                                "placement": {"availability_zone": "us-east-1a"},
                                "tags": {"Name": "test-instance"},
                                "public_ip_address": "203.0.113.12",
                                "private_ip_address": "10.0.1.12",
                            }
                        ]
                    },
                )
            return MagicMock(success=False, error="Unknown action")

        mock_aws_tool.execute.side_effect = mock_execute_side_effect

        # Initialize the provider
        await multicloud_tool_integration._ensure_provider_initialized("aws")

        # Test start resource
        result = await multicloud_tool_integration.execute(
            "start_resource", {"resource_id": "compute-i-1234567890abcdef0"}
        )

        assert result.success
        assert result.output["resource_id"] == "compute-i-1234567890abcdef0"
        assert result.output["state"] == "running"
        assert "started successfully" in result.output["message"]

        # Test stop resource
        result = await multicloud_tool_integration.execute(
            "stop_resource", {"resource_id": "compute-i-1234567890abcdef0"}
        )

        assert result.success
        assert result.output["resource_id"] == "compute-i-1234567890abcdef0"
        assert result.output["state"] == "running"  # From the mocked get_resource call
        assert "stopped successfully" in result.output["message"]

        # Test terminate resource
        result = await multicloud_tool_integration.execute(
            "terminate_resource", {"resource_id": "compute-i-1234567890abcdef0"}
        )

        assert result.success
        assert result.output["resource_id"] == "compute-i-1234567890abcdef0"
        assert "terminated successfully" in result.output["message"]

    @pytest.mark.asyncio
    @patch("src.tools.multicloud.providers.aws_provider.AWSCloudTool")
    async def test_cost_estimation_integration(
        self, mock_aws_tool_class, multicloud_tool_integration
    ):
        """Test cost estimation integrates correctly with AWS tool."""
        # Setup mock AWS tool
        mock_aws_tool = AsyncMock()
        mock_aws_tool_class.return_value = mock_aws_tool

        mock_aws_tool.initialize.return_value = None
        mock_aws_tool.execute.return_value = MagicMock(
            success=True,
            output={
                "total_estimated_cost": 8.50,
                "duration_hours": 720,
                "currency": "USD",
                "resource_costs": [{"resource_type": "ec2", "cost": 8.50}],
            },
        )

        # Test cost comparison
        result = await multicloud_tool_integration.execute(
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

        # Verify the result
        assert result.success
        assert result.output["resource_type"] == "compute"
        assert "aws" in result.output["cost_estimates"]
        assert result.output["cost_estimates"]["aws"]["estimated_cost"] == 8.50
        assert result.output["cost_estimates"]["aws"]["currency"] == "USD"
        assert result.output["cheapest_provider"] == "aws"

    @pytest.mark.asyncio
    async def test_error_handling_invalid_provider(self, multicloud_tool_integration):
        """Test error handling for invalid provider configurations."""
        result = await multicloud_tool_integration.execute(
            "provision_compute",
            {
                "provider": "invalid_provider",
                "name": "test-instance",
                "region": "us-east-1",
                "instance_size": "micro",
                "image": "ami-12345678",
            },
        )

        assert not result.success
        assert (
            "invalid_provider" in result.error
            or "not supported" in result.error.lower()
        )

    @pytest.mark.asyncio
    @patch("src.tools.multicloud.providers.aws_provider.AWSCloudTool")
    async def test_error_handling_aws_failure(
        self, mock_aws_tool_class, multicloud_tool_integration
    ):
        """Test error handling when AWS operations fail."""
        # Setup mock AWS tool that fails
        mock_aws_tool = AsyncMock()
        mock_aws_tool_class.return_value = mock_aws_tool

        mock_aws_tool.initialize.return_value = None
        mock_aws_tool.execute.return_value = MagicMock(
            success=False, error="AWS API Error: Invalid AMI ID"
        )

        # Test compute provisioning failure
        result = await multicloud_tool_integration.execute(
            "provision_compute",
            {
                "provider": "aws",
                "name": "test-instance",
                "region": "us-east-1",
                "instance_size": "micro",
                "image": "ami-invalid",
                "tags": {"Environment": "test"},
            },
        )

        # Verify error handling
        assert not result.success
        assert "AWS API Error" in result.error or "Invalid AMI ID" in result.error
