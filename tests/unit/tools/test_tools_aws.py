"""
Tests for AWS Cloud Provider tool.

This module contains unit tests for the AWS cloud provider tool,
testing AWS service integrations, parameter validation, and error handling.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import botocore.exceptions
import pytest

from src.tools.aws import AWSCloudTool
from src.tools.base import ToolConfig, ToolError


class TestAWSCloudTool:
    """Test suite for AWS Cloud Provider tool."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ToolConfig(
            name="aws",
            version="1.0.0",
            timeout=300,
            max_retries=3,
        )

    @pytest.fixture
    def aws_tool(self, config):
        """Create AWS tool instance."""
        return AWSCloudTool(config)

    @pytest.fixture
    def mock_clients(self):
        """Create mock AWS clients."""
        return {
            "ec2": MagicMock(),
            "rds": MagicMock(),
            "lambda": MagicMock(),
            "iam": MagicMock(),
            "sts": MagicMock(),
            "pricing": MagicMock(),
        }

    @pytest.mark.asyncio
    async def test_get_schema(self, aws_tool):
        """Test schema retrieval."""
        schema = await aws_tool.get_schema()

        assert schema.name == "aws"
        assert schema.description == "AWS Cloud Provider tool for resource management"
        assert "create_ec2_instance" in schema.actions
        assert "create_rds_instance" in schema.actions
        assert "create_lambda_function" in schema.actions
        assert "create_iam_role" in schema.actions

    @pytest.mark.asyncio
    async def test_initialization_success(self, aws_tool, mock_clients):
        """Test successful AWS client initialization."""
        with patch("boto3.Session") as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value = mock_session_instance

            # Mock client creation
            def mock_client(service, **kwargs):
                return mock_clients[service]

            mock_session_instance.client = mock_client

            # Mock STS identity call
            mock_clients["sts"].get_caller_identity.return_value = {
                "Account": "123456789012",
                "UserId": "AIDACKCEVSQ6C2EXAMPLE",
                "Arn": "arn:aws:iam::123456789012:user/test-user",
            }

            await aws_tool.initialize()

            assert aws_tool._session == mock_session_instance
            assert "ec2" in aws_tool._clients
            assert "rds" in aws_tool._clients

    @pytest.mark.asyncio
    async def test_initialization_no_credentials(self, aws_tool):
        """Test initialization with no credentials."""
        with patch("boto3.Session") as mock_session:
            mock_session.side_effect = botocore.exceptions.NoCredentialsError()

            with pytest.raises(ToolError) as exc_info:
                await aws_tool.initialize()

            assert "AWS credentials not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cost_estimation_ec2(self, aws_tool):
        """Test cost estimation for EC2 instances."""
        params = {
            "instance_type": "t3.medium",
        }

        cost_estimate = await aws_tool.estimate_cost("create_ec2_instance", params)

        assert cost_estimate.estimated_cost > 0
        assert "ec2_instance" in cost_estimate.cost_breakdown
        assert cost_estimate.confidence == 0.7

    @pytest.mark.asyncio
    async def test_cost_estimation_rds(self, aws_tool):
        """Test cost estimation for RDS instances."""
        params = {
            "db_instance_class": "db.t3.small",
            "allocated_storage": 100,
        }

        cost_estimate = await aws_tool.estimate_cost("create_rds_instance", params)

        assert cost_estimate.estimated_cost > 0
        assert "rds_instance" in cost_estimate.cost_breakdown
        assert "rds_storage" in cost_estimate.cost_breakdown

    @pytest.mark.asyncio
    async def test_cost_estimation_lambda(self, aws_tool):
        """Test cost estimation for Lambda functions."""
        params = {
            "memory_size": 256,
        }

        cost_estimate = await aws_tool.estimate_cost("create_lambda_function", params)

        assert cost_estimate.estimated_cost > 0
        assert "lambda_requests" in cost_estimate.cost_breakdown
        assert "lambda_compute" in cost_estimate.cost_breakdown

    @pytest.mark.asyncio
    async def test_parameter_validation_ec2(self, aws_tool):
        """Test parameter validation for EC2 operations."""
        validator = await aws_tool._create_validator()

        # Valid parameters
        result = validator.validate(
            "create_ec2_instance",
            {
                "instance_type": "t3.micro",
                "ami_id": "ami-12345678",
            },
        )
        assert result.valid is True

        # Missing required parameters
        result = validator.validate("create_ec2_instance", {})
        assert result.valid is False
        assert "instance_type is required" in result.errors
        assert "ami_id is required" in result.errors

        # Invalid instance type
        result = validator.validate(
            "create_ec2_instance",
            {
                "instance_type": "invalid.type",
                "ami_id": "ami-12345678",
            },
        )
        assert result.valid is False
        assert "Invalid instance type" in result.errors

    @pytest.mark.asyncio
    async def test_parameter_validation_rds(self, aws_tool):
        """Test parameter validation for RDS operations."""
        validator = await aws_tool._create_validator()

        # Valid parameters
        result = validator.validate(
            "create_rds_instance",
            {
                "db_instance_identifier": "test-db",
                "engine": "mysql",
                "db_instance_class": "db.t3.micro",
                "master_username": "admin",
                "master_password": "password123",
                "allocated_storage": 20,
            },
        )
        assert result.valid is True

        # Invalid storage size
        result = validator.validate(
            "create_rds_instance",
            {
                "db_instance_identifier": "test-db",
                "engine": "mysql",
                "db_instance_class": "db.t3.micro",
                "master_username": "admin",
                "master_password": "password123",
                "allocated_storage": 10,  # Too small
            },
        )
        assert result.valid is False
        assert "Minimum allocated storage is 20 GB" in result.errors

    @pytest.mark.asyncio
    async def test_parameter_validation_lambda(self, aws_tool):
        """Test parameter validation for Lambda operations."""
        validator = await aws_tool._create_validator()

        # Valid parameters
        result = validator.validate(
            "create_lambda_function",
            {
                "function_name": "test-function",
                "runtime": "python3.9",
                "role": "arn:aws:iam::123456789012:role/lambda-role",
                "handler": "index.handler",
                "code": {"ZipFile": b"test code"},
            },
        )
        assert result.valid is True

        # Missing required parameters
        result = validator.validate("create_lambda_function", {})
        assert result.valid is False
        assert "function_name is required for Lambda function creation" in result.errors

    @pytest.mark.asyncio
    async def test_parameter_validation_security(self, aws_tool):
        """Test security parameter validation."""
        validator = await aws_tool._create_validator()

        # Weak password
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
        assert result.valid is False
        assert "Master password must be at least 8 characters" in result.errors

    @pytest.mark.asyncio
    async def test_create_ec2_instance_success(self, aws_tool, mock_clients):
        """Test successful EC2 instance creation."""
        aws_tool._clients = mock_clients

        # Mock EC2 response
        mock_clients["ec2"].run_instances.return_value = {
            "Instances": [
                {
                    "InstanceId": "i-1234567890abcdef0",
                    "InstanceType": "t3.micro",
                    "State": {"Name": "pending"},
                    "LaunchTime": datetime.utcnow(),
                    "PrivateIpAddress": "10.0.0.10",
                    "PublicIpAddress": "54.123.45.67",
                }
            ]
        }

        params = {
            "instance_type": "t3.micro",
            "ami_id": "ami-12345678",
            "tags": {"Name": "test-instance"},
        }

        result = await aws_tool._create_ec2_instance(params)

        assert result["instance_id"] == "i-1234567890abcdef0"
        assert result["instance_type"] == "t3.micro"
        assert result["state"] == "pending"
        assert "launch_time" in result

        # Verify tagging was called
        mock_clients["ec2"].create_tags.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_ec2_instances_success(self, aws_tool, mock_clients):
        """Test successful EC2 instance listing."""
        aws_tool._clients = mock_clients

        # Mock EC2 response
        mock_clients["ec2"].describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-1234567890abcdef0",
                            "InstanceType": "t3.micro",
                            "State": {"Name": "running"},
                            "LaunchTime": datetime.utcnow(),
                            "PrivateIpAddress": "10.0.0.10",
                            "PublicIpAddress": "54.123.45.67",
                            "Tags": [{"Key": "Name", "Value": "test-instance"}],
                        }
                    ]
                }
            ]
        }

        result = await aws_tool._list_ec2_instances({})

        assert result["count"] == 1
        assert len(result["instances"]) == 1
        assert result["instances"][0]["instance_id"] == "i-1234567890abcdef0"
        assert result["instances"][0]["tags"]["Name"] == "test-instance"

    @pytest.mark.asyncio
    async def test_create_rds_instance_success(self, aws_tool, mock_clients):
        """Test successful RDS instance creation."""
        aws_tool._clients = mock_clients

        # Mock RDS response
        mock_clients["rds"].create_db_instance.return_value = {
            "DBInstance": {
                "DBInstanceIdentifier": "test-db",
                "Engine": "mysql",
                "DBInstanceClass": "db.t3.micro",
                "DBInstanceStatus": "creating",
                "AllocatedStorage": 20,
                "DBInstanceArn": "arn:aws:rds:us-east-1:123456789012:db:test-db",
                "Endpoint": {
                    "Address": "test-db.abcdefghijkl.us-east-1.rds.amazonaws.com",
                    "Port": 3306,
                },
            }
        }

        params = {
            "db_instance_identifier": "test-db",
            "engine": "mysql",
            "db_instance_class": "db.t3.micro",
            "master_username": "admin",
            "master_password": "password123",
            "allocated_storage": 20,
        }

        result = await aws_tool._create_rds_instance(params)

        assert result["db_instance_identifier"] == "test-db"
        assert result["engine"] == "mysql"
        assert result["db_instance_status"] == "creating"
        assert result["endpoint"] == "test-db.abcdefghijkl.us-east-1.rds.amazonaws.com"
        assert result["port"] == 3306

    @pytest.mark.asyncio
    async def test_create_lambda_function_success(self, aws_tool, mock_clients):
        """Test successful Lambda function creation."""
        aws_tool._clients = mock_clients

        # Mock Lambda response
        mock_clients["lambda"].create_function.return_value = {
            "FunctionName": "test-function",
            "FunctionArn": "arn:aws:lambda:us-east-1:123456789012:function:test-function",
            "Runtime": "python3.9",
            "Role": "arn:aws:iam::123456789012:role/lambda-role",
            "Handler": "index.handler",
            "CodeSize": 1024,
            "Description": "Test function",
            "Timeout": 30,
            "MemorySize": 128,
            "LastModified": "2023-01-01T00:00:00.000+0000",
            "State": "Active",
        }

        params = {
            "function_name": "test-function",
            "runtime": "python3.9",
            "role": "arn:aws:iam::123456789012:role/lambda-role",
            "handler": "index.handler",
            "code": {"ZipFile": b"test code"},
        }

        result = await aws_tool._create_lambda_function(params)

        assert result["function_name"] == "test-function"
        assert result["runtime"] == "python3.9"
        assert result["state"] == "Active"
        assert result["memory_size"] == 128

    @pytest.mark.asyncio
    async def test_invoke_lambda_function_success(self, aws_tool, mock_clients):
        """Test successful Lambda function invocation."""
        aws_tool._clients = mock_clients

        # Mock Lambda response
        response_payload = json.dumps({"result": "success"}).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = response_payload

        mock_clients["lambda"].invoke.return_value = {
            "StatusCode": 200,
            "Payload": mock_response,
            "ExecutedVersion": "$LATEST",
        }

        params = {
            "function_name": "test-function",
            "payload": {"input": "test"},
        }

        result = await aws_tool._invoke_lambda_function(params)

        assert result["status_code"] == 200
        assert result["payload"] == {"result": "success"}
        assert result["executed_version"] == "$LATEST"

    @pytest.mark.asyncio
    async def test_create_iam_role_success(self, aws_tool, mock_clients):
        """Test successful IAM role creation."""
        aws_tool._clients = mock_clients

        # Mock IAM response
        mock_clients["iam"].create_role.return_value = {
            "Role": {
                "RoleName": "test-role",
                "Arn": "arn:aws:iam::123456789012:role/test-role",
                "RoleId": "AROA123456789012",
                "Path": "/",
                "CreateDate": datetime.utcnow(),
                "AssumeRolePolicyDocument": "%7B%22Version%22%3A%222012-10-17%22%7D",
            }
        }

        params = {
            "role_name": "test-role",
            "assume_role_policy_document": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            },
        }

        result = await aws_tool._create_iam_role(params)

        assert result["role_name"] == "test-role"
        assert result["role_arn"] == "arn:aws:iam::123456789012:role/test-role"
        assert "create_date" in result

    @pytest.mark.asyncio
    async def test_get_account_info_success(self, aws_tool, mock_clients):
        """Test successful account info retrieval."""
        aws_tool._clients = mock_clients
        aws_tool._region = "us-west-2"

        # Mock STS response
        mock_clients["sts"].get_caller_identity.return_value = {
            "Account": "123456789012",
            "UserId": "AIDACKCEVSQ6C2EXAMPLE",
            "Arn": "arn:aws:iam::123456789012:user/test-user",
        }

        result = await aws_tool._get_account_info({})

        assert result["account_id"] == "123456789012"
        assert result["user_id"] == "AIDACKCEVSQ6C2EXAMPLE"
        assert result["region"] == "us-west-2"

    @pytest.mark.asyncio
    async def test_estimate_costs_multiple_resources(self, aws_tool):
        """Test cost estimation for multiple resources."""
        params = {
            "resources": [
                {"type": "ec2", "params": {"instance_type": "t3.micro"}},
                {
                    "type": "rds",
                    "params": {
                        "db_instance_class": "db.t3.micro",
                        "allocated_storage": 20,
                    },
                },
            ],
            "duration_hours": 24,
        }

        result = await aws_tool._estimate_costs(params)

        assert result["total_estimated_cost"] > 0
        assert len(result["resource_costs"]) == 2
        assert result["duration_hours"] == 24
        assert result["currency"] == "USD"

    @pytest.mark.asyncio
    async def test_aws_api_error_handling(self, aws_tool, mock_clients):
        """Test AWS API error handling."""
        aws_tool._clients = mock_clients

        # Mock AWS API error
        error_response = {
            "Error": {
                "Code": "InvalidInstanceType",
                "Message": "Invalid instance type",
            }
        }
        mock_clients["ec2"].run_instances.side_effect = botocore.exceptions.ClientError(
            error_response, "RunInstances"
        )

        params = {
            "instance_type": "invalid.type",
            "ami_id": "ami-12345678",
        }

        with pytest.raises(ToolError) as exc_info:
            await aws_tool._execute_action("create_ec2_instance", params)

        assert "AWS API error (InvalidInstanceType)" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unknown_action_error(self, aws_tool):
        """Test error handling for unknown actions."""
        with pytest.raises(ToolError) as exc_info:
            await aws_tool._execute_action("unknown_action", {})

        assert "Unknown action: unknown_action" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cleanup(self, aws_tool, mock_clients):
        """Test cleanup functionality."""
        aws_tool._clients = mock_clients
        aws_tool._session = MagicMock()

        await aws_tool.cleanup()

        assert len(aws_tool._clients) == 0
        assert aws_tool._session is None

    @pytest.mark.asyncio
    async def test_ec2_instance_lifecycle(self, aws_tool, mock_clients):
        """Test complete EC2 instance lifecycle operations."""
        aws_tool._clients = mock_clients

        # Mock start instance
        mock_clients["ec2"].start_instances.return_value = {
            "StartingInstances": [
                {
                    "InstanceId": "i-1234567890abcdef0",
                    "PreviousState": {"Name": "stopped"},
                    "CurrentState": {"Name": "pending"},
                }
            ]
        }

        start_result = await aws_tool._start_ec2_instance(
            {"instance_id": "i-1234567890abcdef0"}
        )
        assert start_result["current_state"] == "pending"

        # Mock stop instance
        mock_clients["ec2"].stop_instances.return_value = {
            "StoppingInstances": [
                {
                    "InstanceId": "i-1234567890abcdef0",
                    "PreviousState": {"Name": "running"},
                    "CurrentState": {"Name": "stopping"},
                }
            ]
        }

        stop_result = await aws_tool._stop_ec2_instance(
            {"instance_id": "i-1234567890abcdef0"}
        )
        assert stop_result["current_state"] == "stopping"

        # Mock terminate instance
        mock_clients["ec2"].terminate_instances.return_value = {
            "TerminatingInstances": [
                {
                    "InstanceId": "i-1234567890abcdef0",
                    "PreviousState": {"Name": "stopped"},
                    "CurrentState": {"Name": "shutting-down"},
                }
            ]
        }

        terminate_result = await aws_tool._terminate_ec2_instance(
            {"instance_id": "i-1234567890abcdef0"}
        )
        assert terminate_result["current_state"] == "shutting-down"

    @pytest.mark.asyncio
    async def test_security_group_operations(self, aws_tool, mock_clients):
        """Test security group creation and configuration."""
        aws_tool._clients = mock_clients

        # Mock create security group
        mock_clients["ec2"].create_security_group.return_value = {
            "GroupId": "sg-12345678"
        }

        create_params = {
            "group_name": "test-sg",
            "description": "Test security group",
            "vpc_id": "vpc-12345678",
        }

        create_result = await aws_tool._create_security_group(create_params)
        assert create_result["group_id"] == "sg-12345678"

        # Mock authorize ingress
        ingress_params = {
            "group_id": "sg-12345678",
            "ip_permissions": [
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                }
            ],
        }

        ingress_result = await aws_tool._authorize_security_group_ingress(
            ingress_params
        )
        assert ingress_result["rules_added"] == 1

    @pytest.mark.asyncio
    async def test_iam_policy_operations(self, aws_tool, mock_clients):
        """Test IAM policy attach/detach operations."""
        aws_tool._clients = mock_clients

        # Test attach policy
        attach_result = await aws_tool._attach_role_policy(
            {
                "role_name": "test-role",
                "policy_arn": "arn:aws:iam::aws:policy/PowerUserAccess",
            }
        )
        assert attach_result["attached"] is True

        # Test detach policy
        detach_result = await aws_tool._detach_role_policy(
            {
                "role_name": "test-role",
                "policy_arn": "arn:aws:iam::aws:policy/PowerUserAccess",
            }
        )
        assert detach_result["detached"] is True
