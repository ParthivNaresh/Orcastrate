"""
Integration tests for AWS Cloud Provider tool.

This module contains integration tests that demonstrate
the AWS tool working in complete end-to-end scenarios.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.tools.aws import AWSCloudTool
from src.tools.base import ToolConfig


@pytest.mark.integration
class TestAWSIntegration:
    """Integration tests for AWS Cloud Provider tool."""

    @pytest.fixture
    def aws_config(self):
        """Create AWS tool configuration."""
        return ToolConfig(
            name="aws",
            version="1.0.0",
            timeout=300,
            max_retries=3,
        )

    @pytest.fixture
    def aws_tool(self, aws_config):
        """Create AWS tool instance."""
        return AWSCloudTool(aws_config)

    @pytest.mark.asyncio
    async def test_aws_tool_schema_integration(self, aws_tool):
        """Test AWS tool schema integration."""
        schema = await aws_tool.get_schema()

        # Verify comprehensive action coverage
        expected_actions = [
            "create_ec2_instance",
            "list_ec2_instances",
            "start_ec2_instance",
            "stop_ec2_instance",
            "terminate_ec2_instance",
            "create_rds_instance",
            "list_rds_instances",
            "delete_rds_instance",
            "create_lambda_function",
            "list_lambda_functions",
            "invoke_lambda_function",
            "delete_lambda_function",
            "create_iam_role",
            "attach_role_policy",
            "detach_role_policy",
            "create_security_group",
            "authorize_security_group_ingress",
            "get_account_info",
            "estimate_costs",
        ]

        for action in expected_actions:
            assert action in schema.actions, f"Missing action: {action}"

        # Verify parameter schemas are complete
        ec2_create = schema.actions["create_ec2_instance"]
        assert "instance_type" in ec2_create["parameters"]
        assert "ami_id" in ec2_create["parameters"]
        assert ec2_create["parameters"]["instance_type"]["required"] is True
        assert ec2_create["parameters"]["ami_id"]["required"] is True

    @pytest.mark.asyncio
    async def test_aws_cost_estimation_integration(self, aws_tool):
        """Test AWS cost estimation for multiple resource types."""
        # Test EC2 cost estimation
        ec2_cost = await aws_tool.estimate_cost(
            "create_ec2_instance", {"instance_type": "t3.large"}
        )
        assert ec2_cost.estimated_cost > 0
        assert "ec2_instance" in ec2_cost.cost_breakdown

        # Test RDS cost estimation
        rds_cost = await aws_tool.estimate_cost(
            "create_rds_instance",
            {"db_instance_class": "db.t3.medium", "allocated_storage": 100},
        )
        assert rds_cost.estimated_cost > 0
        assert "rds_instance" in rds_cost.cost_breakdown
        assert "rds_storage" in rds_cost.cost_breakdown

        # Test Lambda cost estimation
        lambda_cost = await aws_tool.estimate_cost(
            "create_lambda_function", {"memory_size": 512}
        )
        assert lambda_cost.estimated_cost > 0
        assert "lambda_requests" in lambda_cost.cost_breakdown
        assert "lambda_compute" in lambda_cost.cost_breakdown

    @pytest.mark.asyncio
    async def test_aws_parameter_validation_comprehensive(self, aws_tool):
        """Test comprehensive parameter validation across AWS services."""
        validator = await aws_tool._create_validator()

        # Test EC2 validation scenarios
        scenarios = [
            # Valid EC2 instance
            {
                "action": "create_ec2_instance",
                "params": {
                    "instance_type": "t3.micro",
                    "ami_id": "ami-12345678",
                    "key_name": "my-key",
                    "security_groups": ["sg-12345"],
                    "tags": {"Environment": "test"},
                },
                "should_be_valid": True,
            },
            # Invalid instance type
            {
                "action": "create_ec2_instance",
                "params": {"instance_type": "invalid-type", "ami_id": "ami-12345678"},
                "should_be_valid": False,
            },
            # Valid RDS instance
            {
                "action": "create_rds_instance",
                "params": {
                    "db_instance_identifier": "test-db",
                    "engine": "mysql",
                    "db_instance_class": "db.t3.micro",
                    "master_username": "admin",
                    "master_password": "secure123password",
                    "allocated_storage": 20,
                },
                "should_be_valid": True,
            },
            # RDS with insufficient storage
            {
                "action": "create_rds_instance",
                "params": {
                    "db_instance_identifier": "test-db",
                    "engine": "mysql",
                    "db_instance_class": "db.t3.micro",
                    "master_username": "admin",
                    "master_password": "secure123password",
                    "allocated_storage": 10,  # Too small
                },
                "should_be_valid": False,
            },
            # Valid Lambda function
            {
                "action": "create_lambda_function",
                "params": {
                    "function_name": "test-lambda-function",
                    "runtime": "python3.9",
                    "role": "arn:aws:iam::123456789012:role/lambda-role",
                    "handler": "lambda_function.lambda_handler",
                    "code": {
                        "ZipFile": b"def lambda_handler(event, context): return 'Hello'"
                    },
                },
                "should_be_valid": True,
            },
            # Lambda with invalid function name
            {
                "action": "create_lambda_function",
                "params": {
                    "function_name": "invalid function name!",  # Invalid characters
                    "runtime": "python3.9",
                    "role": "arn:aws:iam::123456789012:role/lambda-role",
                    "handler": "lambda_function.lambda_handler",
                    "code": {"ZipFile": b"test"},
                },
                "should_be_valid": False,
            },
        ]

        for scenario in scenarios:
            result = validator.validate(scenario["action"], scenario["params"])
            if scenario["should_be_valid"]:
                assert (
                    result.valid
                ), f"Expected {scenario['action']} to be valid but got errors: {result.errors}"
            else:
                assert (
                    not result.valid
                ), f"Expected {scenario['action']} to be invalid but it passed validation"

    @pytest.mark.asyncio
    async def test_aws_multi_resource_cost_estimation(self, aws_tool):
        """Test cost estimation for complex multi-resource deployments."""
        # Simulate a typical web application deployment
        resources = [
            {"type": "ec2", "params": {"instance_type": "t3.medium"}},
            {
                "type": "rds",
                "params": {"db_instance_class": "db.t3.small", "allocated_storage": 50},
            },
            {"type": "lambda", "params": {"memory_size": 256}},
        ]

        cost_result = await aws_tool._estimate_costs(
            {"resources": resources, "duration_hours": 24 * 7}  # One week
        )

        assert cost_result["total_estimated_cost"] > 0
        assert len(cost_result["resource_costs"]) == 3
        assert cost_result["duration_hours"] == 24 * 7
        assert cost_result["currency"] == "USD"

        # Verify individual resource costs
        resource_types = [rc["resource_type"] for rc in cost_result["resource_costs"]]
        assert "ec2" in resource_types
        assert "rds" in resource_types
        assert "lambda" in resource_types

    @pytest.mark.asyncio
    async def test_aws_end_to_end_workflow_simulation(self, aws_tool):
        """Test simulated end-to-end AWS workflow with mocked clients."""
        # Mock AWS clients for a complete workflow simulation
        mock_clients = {
            "ec2": MagicMock(),
            "iam": MagicMock(),
            "sts": MagicMock(),
        }

        # Mock successful EC2 instance creation
        mock_clients["ec2"].run_instances.return_value = {
            "Instances": [
                {
                    "InstanceId": "i-1234567890abcdef0",
                    "InstanceType": "t3.micro",
                    "State": {"Name": "pending"},
                    "LaunchTime": datetime(2024, 1, 1),
                    "PrivateIpAddress": "10.0.0.10",
                    "PublicIpAddress": "54.123.45.67",
                }
            ]
        }

        # Mock IAM role creation
        mock_clients["iam"].create_role.return_value = {
            "Role": {
                "RoleName": "ec2-role",
                "Arn": "arn:aws:iam::123456789012:role/ec2-role",
                "RoleId": "AROA123456789012",
                "Path": "/",
                "CreateDate": datetime(2024, 1, 1),
                "AssumeRolePolicyDocument": "%7B%22Version%22%3A%222012-10-17%22%7D",
            }
        }

        # Mock STS identity
        mock_clients["sts"].get_caller_identity.return_value = {
            "Account": "123456789012",
            "UserId": "AIDACKCEVSQ6C2EXAMPLE",
            "Arn": "arn:aws:iam::123456789012:user/test-user",
        }

        # Set up the tool with mocked clients
        aws_tool._clients = mock_clients

        # Simulate workflow: Create IAM role, then EC2 instance

        # Step 1: Create IAM role
        iam_result = await aws_tool._create_iam_role(
            {
                "role_name": "ec2-role",
                "assume_role_policy_document": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "ec2.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                },
            }
        )

        assert iam_result["role_name"] == "ec2-role"
        assert "arn:aws:iam::123456789012:role/ec2-role" in iam_result["role_arn"]

        # Step 2: Create EC2 instance
        ec2_result = await aws_tool._create_ec2_instance(
            {
                "instance_type": "t3.micro",
                "ami_id": "ami-0abcdef1234567890",
                "tags": {"Name": "test-instance", "Environment": "development"},
            }
        )

        assert ec2_result["instance_id"] == "i-1234567890abcdef0"
        assert ec2_result["instance_type"] == "t3.micro"
        assert ec2_result["state"] == "pending"

        # Step 3: Get account info
        account_info = await aws_tool._get_account_info({})
        assert account_info["account_id"] == "123456789012"

        # Verify all expected calls were made
        mock_clients["iam"].create_role.assert_called_once()
        mock_clients["ec2"].run_instances.assert_called_once()
        mock_clients["ec2"].create_tags.assert_called_once()

    @pytest.mark.asyncio
    async def test_aws_action_support_verification(self, aws_tool):
        """Test that all documented actions are actually supported."""
        supported_actions = await aws_tool._get_supported_actions()
        schema = await aws_tool.get_schema()

        # Verify all schema actions are supported
        for action_name in schema.actions.keys():
            assert (
                action_name in supported_actions
            ), f"Action {action_name} in schema but not supported"

        # Verify we have a good coverage of AWS services
        service_coverage = {
            "ec2": any("ec2" in action for action in supported_actions),
            "rds": any("rds" in action for action in supported_actions),
            "lambda": any("lambda" in action for action in supported_actions),
            "iam": any("iam" in action for action in supported_actions),
        }

        for service, covered in service_coverage.items():
            assert covered, f"No {service} actions found in supported actions"

    @pytest.mark.asyncio
    async def test_aws_error_handling_integration(self, aws_tool):
        """Test error handling across different failure scenarios."""
        # Test validation errors
        validator = await aws_tool._create_validator()

        # Missing required parameters
        result = validator.validate("create_ec2_instance", {})
        assert not result.valid
        assert len(result.errors) > 0

        # Test rollback functionality (placeholder implementation)
        rollback_result = await aws_tool._execute_rollback("test-execution-id")
        assert rollback_result["execution_id"] == "test-execution-id"
        assert rollback_result["rollback_status"] == "not_implemented"

    @pytest.mark.asyncio
    async def test_aws_security_validations(self, aws_tool):
        """Test security-focused parameter validations."""
        validator = await aws_tool._create_validator()

        # Test weak password rejection
        result = validator.validate(
            "create_rds_instance",
            {
                "db_instance_identifier": "test-db",
                "engine": "mysql",
                "db_instance_class": "db.t3.micro",
                "master_username": "admin",
                "master_password": "weak",  # Too short
                "allocated_storage": 20,
            },
        )

        assert not result.valid
        assert any("password" in error.lower() for error in result.errors)

        # Test valid strong password acceptance
        result = validator.validate(
            "create_rds_instance",
            {
                "db_instance_identifier": "test-db",
                "engine": "mysql",
                "db_instance_class": "db.t3.micro",
                "master_username": "admin",
                "master_password": "StrongPassword123!",
                "allocated_storage": 20,
            },
        )

        assert result.valid
