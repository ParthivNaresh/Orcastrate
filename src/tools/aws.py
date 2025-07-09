"""
AWS Cloud Provider tool for cloud resource management.

This module provides a concrete implementation of the Tool interface for
AWS cloud operations including EC2, ECS, RDS, Lambda, and other AWS services.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
import botocore.exceptions
from botocore.client import BaseClient

from ..config.settings import get_settings
from .base import (
    CostEstimate,
    Tool,
    ToolConfig,
    ToolError,
    ToolSchema,
    ValidationResult,
)


class AWSCloudTool(Tool):
    """
    AWS Cloud Provider tool for cloud resource management.

    Provides functionality for:
    - EC2 instances (create, list, start, stop, terminate)
    - RDS databases (create, list, start, stop, delete)
    - Lambda functions (create, deploy, invoke, delete)
    - ECS clusters and services (create, deploy, scale)
    - IAM roles and policies (create, attach, detach)
    - Security groups and VPC management
    - Cost estimation for AWS resources
    """

    def __init__(self, config: Optional[ToolConfig] = None):
        if config is None:
            config = ToolConfig(name="aws", version="1.0.0")
        super().__init__(config)

        # Load AWS configuration from centralized settings
        settings = get_settings()
        self._clients: Dict[str, BaseClient] = {}
        self._session: Optional[boto3.Session] = None

        # Get region from settings, with fallback to environment variable
        # This ensures AWS_DEFAULT_REGION is properly respected
        import os

        env_region = os.environ.get("AWS_DEFAULT_REGION")
        if env_region:
            self._region = env_region
        else:
            self._region = settings.cloud.aws_region

        # Initialize AWS session with credentials from settings
        self._initialize_session()

    def _initialize_session(self) -> None:
        """Initialize AWS session with credentials from settings."""
        try:
            settings = get_settings()

            # Create AWS session with credentials from settings
            session_kwargs = {"region_name": self._region}

            # Use explicit credentials if provided in settings
            if settings.cloud.aws_access_key_id:
                session_kwargs["aws_access_key_id"] = settings.cloud.aws_access_key_id
            if settings.cloud.aws_secret_access_key:
                session_kwargs["aws_secret_access_key"] = (
                    settings.cloud.aws_secret_access_key
                )
            if settings.cloud.aws_session_token:
                session_kwargs["aws_session_token"] = settings.cloud.aws_session_token

            self._session = boto3.Session(**session_kwargs)
            self.logger.info(f"AWS session initialized for region: {self._region}")

        except Exception as e:
            self.logger.warning(f"AWS session initialization failed: {e}")
            self._session = None

    async def get_schema(self) -> ToolSchema:
        """Return the AWS Cloud tool schema."""
        return ToolSchema(
            name="aws",
            description="AWS Cloud Provider tool for resource management",
            version=self.config.version,
            actions={
                # EC2 Actions
                "create_ec2_instance": {
                    "description": "Create an EC2 instance",
                    "parameters": {
                        "instance_type": {"type": "string", "required": True},
                        "ami_id": {"type": "string", "required": True},
                        "key_name": {"type": "string", "required": False},
                        "security_groups": {"type": "array", "required": False},
                        "subnet_id": {"type": "string", "required": False},
                        "user_data": {"type": "string", "required": False},
                        "tags": {"type": "object", "required": False},
                    },
                },
                "list_ec2_instances": {
                    "description": "List EC2 instances",
                    "parameters": {
                        "filters": {"type": "object", "required": False},
                        "instance_ids": {"type": "array", "required": False},
                    },
                },
                "start_ec2_instance": {
                    "description": "Start an EC2 instance",
                    "parameters": {
                        "instance_id": {"type": "string", "required": True},
                    },
                },
                "stop_ec2_instance": {
                    "description": "Stop an EC2 instance",
                    "parameters": {
                        "instance_id": {"type": "string", "required": True},
                    },
                },
                "terminate_ec2_instance": {
                    "description": "Terminate an EC2 instance",
                    "parameters": {
                        "instance_id": {"type": "string", "required": True},
                    },
                },
                # RDS Actions
                "create_rds_instance": {
                    "description": "Create an RDS database instance",
                    "parameters": {
                        "db_instance_identifier": {"type": "string", "required": True},
                        "engine": {"type": "string", "required": True},
                        "db_instance_class": {"type": "string", "required": True},
                        "master_username": {"type": "string", "required": True},
                        "master_password": {"type": "string", "required": True},
                        "allocated_storage": {"type": "integer", "required": True},
                        "vpc_security_group_ids": {"type": "array", "required": False},
                        "db_subnet_group_name": {"type": "string", "required": False},
                        "tags": {"type": "object", "required": False},
                    },
                },
                "list_rds_instances": {
                    "description": "List RDS database instances",
                    "parameters": {
                        "db_instance_identifier": {"type": "string", "required": False},
                    },
                },
                "delete_rds_instance": {
                    "description": "Delete an RDS database instance",
                    "parameters": {
                        "db_instance_identifier": {"type": "string", "required": True},
                        "skip_final_snapshot": {"type": "boolean", "required": False},
                        "final_db_snapshot_identifier": {
                            "type": "string",
                            "required": False,
                        },
                    },
                },
                # Lambda Actions
                "create_lambda_function": {
                    "description": "Create a Lambda function",
                    "parameters": {
                        "function_name": {"type": "string", "required": True},
                        "runtime": {"type": "string", "required": True},
                        "role": {"type": "string", "required": True},
                        "handler": {"type": "string", "required": True},
                        "code": {"type": "object", "required": True},
                        "description": {"type": "string", "required": False},
                        "timeout": {"type": "integer", "required": False},
                        "memory_size": {"type": "integer", "required": False},
                        "environment": {"type": "object", "required": False},
                        "tags": {"type": "object", "required": False},
                    },
                },
                "list_lambda_functions": {
                    "description": "List Lambda functions",
                    "parameters": {
                        "function_version": {"type": "string", "required": False},
                        "marker": {"type": "string", "required": False},
                        "max_items": {"type": "integer", "required": False},
                    },
                },
                "invoke_lambda_function": {
                    "description": "Invoke a Lambda function",
                    "parameters": {
                        "function_name": {"type": "string", "required": True},
                        "payload": {"type": "object", "required": False},
                        "invocation_type": {"type": "string", "required": False},
                    },
                },
                "delete_lambda_function": {
                    "description": "Delete a Lambda function",
                    "parameters": {
                        "function_name": {"type": "string", "required": True},
                        "qualifier": {"type": "string", "required": False},
                    },
                },
                # IAM Actions
                "create_iam_role": {
                    "description": "Create an IAM role",
                    "parameters": {
                        "role_name": {"type": "string", "required": True},
                        "assume_role_policy_document": {
                            "type": "object",
                            "required": True,
                        },
                        "description": {"type": "string", "required": False},
                        "max_session_duration": {"type": "integer", "required": False},
                        "path": {"type": "string", "required": False},
                        "tags": {"type": "object", "required": False},
                    },
                },
                "attach_role_policy": {
                    "description": "Attach a policy to an IAM role",
                    "parameters": {
                        "role_name": {"type": "string", "required": True},
                        "policy_arn": {"type": "string", "required": True},
                    },
                },
                "detach_role_policy": {
                    "description": "Detach a policy from an IAM role",
                    "parameters": {
                        "role_name": {"type": "string", "required": True},
                        "policy_arn": {"type": "string", "required": True},
                    },
                },
                # Security Group Actions
                "create_security_group": {
                    "description": "Create a security group",
                    "parameters": {
                        "group_name": {"type": "string", "required": True},
                        "description": {"type": "string", "required": True},
                        "vpc_id": {"type": "string", "required": False},
                        "tags": {"type": "object", "required": False},
                    },
                },
                "authorize_security_group_ingress": {
                    "description": "Add ingress rules to a security group",
                    "parameters": {
                        "group_id": {"type": "string", "required": True},
                        "ip_permissions": {"type": "array", "required": True},
                    },
                },
                # Utility Actions
                "get_account_info": {
                    "description": "Get AWS account information",
                    "parameters": {},
                },
                "estimate_costs": {
                    "description": "Estimate costs for AWS resources",
                    "parameters": {
                        "resources": {"type": "array", "required": True},
                        "duration_hours": {"type": "integer", "required": False},
                    },
                },
            },
        )

    async def estimate_cost(self, action: str, params: Dict[str, Any]) -> CostEstimate:
        """Estimate cost of AWS operations."""
        base_cost = 0.0
        breakdown = {}

        if action == "create_ec2_instance":
            instance_type = params.get("instance_type", "t3.micro")
            # Rough hourly cost estimates (simplified)
            instance_costs = {
                "t3.nano": 0.0052,
                "t3.micro": 0.0104,
                "t3.small": 0.0208,
                "t3.medium": 0.0416,
                "t3.large": 0.0832,
                "t3.xlarge": 0.1664,
                "m5.large": 0.096,
                "m5.xlarge": 0.192,
                "c5.large": 0.085,
                "c5.xlarge": 0.17,
            }
            hourly_cost = instance_costs.get(instance_type, 0.1)
            base_cost = hourly_cost * 24 * 30  # Monthly estimate
            breakdown["ec2_instance"] = base_cost

        elif action == "create_rds_instance":
            db_instance_class = params.get("db_instance_class", "db.t3.micro")
            allocated_storage = params.get("allocated_storage", 20)

            # Simplified RDS pricing
            instance_costs = {
                "db.t3.micro": 0.017,
                "db.t3.small": 0.034,
                "db.t3.medium": 0.068,
                "db.t3.large": 0.136,
                "db.m5.large": 0.192,
            }
            hourly_cost = instance_costs.get(db_instance_class, 0.1)
            instance_cost = hourly_cost * 24 * 30
            storage_cost = allocated_storage * 0.115  # $0.115 per GB/month
            base_cost = instance_cost + storage_cost
            breakdown["rds_instance"] = instance_cost
            breakdown["rds_storage"] = storage_cost

        elif action == "create_lambda_function":
            memory_size = params.get("memory_size", 128)
            # Simplified Lambda pricing estimate
            base_cost = 0.0000002 * memory_size * 100000  # Estimate for 100k executions
            breakdown["lambda_requests"] = base_cost * 0.4
            breakdown["lambda_compute"] = base_cost * 0.6

        return CostEstimate(
            estimated_cost=base_cost,
            cost_breakdown=breakdown,
            confidence=0.7,  # Medium confidence for estimates
            factors=["AWS pricing varies by region", "Estimates based on us-east-1"],
        )

    async def _create_client(self) -> Any:
        """Create AWS session and clients."""
        try:
            settings = get_settings()

            # Create AWS session with credentials from settings
            session_kwargs = {"region_name": self._region}

            # Use explicit credentials if provided in settings
            if settings.cloud.aws_access_key_id:
                session_kwargs["aws_access_key_id"] = settings.cloud.aws_access_key_id
            if settings.cloud.aws_secret_access_key:
                session_kwargs["aws_secret_access_key"] = (
                    settings.cloud.aws_secret_access_key
                )
            if settings.cloud.aws_session_token:
                session_kwargs["aws_session_token"] = settings.cloud.aws_session_token

            self._session = boto3.Session(**session_kwargs)

            # Initialize common clients
            self._clients = {
                "ec2": self._session.client("ec2"),
                "rds": self._session.client("rds"),
                "lambda": self._session.client("lambda"),
                "iam": self._session.client("iam"),
                "sts": self._session.client("sts"),
                "pricing": self._session.client(
                    "pricing", region_name="us-east-1"
                ),  # Pricing API only in us-east-1
            }

            # Test connection
            sts_client = self._clients["sts"]
            identity = sts_client.get_caller_identity()
            self.logger.info(
                f"AWS connection established for account: {identity.get('Account')}"
            )

            return self._session

        except botocore.exceptions.NoCredentialsError:
            raise ToolError(
                "AWS credentials not found. Please configure AWS credentials."
            )
        except botocore.exceptions.PartialCredentialsError:
            raise ToolError(
                "Incomplete AWS credentials. Please check your AWS configuration."
            )
        except Exception as e:
            raise ToolError(f"Failed to initialize AWS session: {e}")

    async def _create_validator(self) -> Any:
        """Create parameter validator for AWS operations."""

        class AWSValidator:
            def __init__(self, tool_instance):
                self.tool = tool_instance

            def validate(self, action: str, params: Dict[str, Any]) -> ValidationResult:
                errors: list[str] = []
                warnings: list[str] = []
                normalized_params = params.copy()

                # Validate common AWS parameters
                if action.startswith("create_ec2"):
                    errors.extend(self._validate_ec2_params(action, params))
                elif action.startswith("create_rds"):
                    errors.extend(self._validate_rds_params(action, params))
                elif action.startswith("create_lambda"):
                    errors.extend(self._validate_lambda_params(action, params))
                elif action.startswith("create_iam"):
                    errors.extend(self._validate_iam_params(action, params))

                # Validate resource names
                if "instance_id" in params:
                    if not self._validate_instance_id(params["instance_id"]):
                        errors.append("Invalid EC2 instance ID format")

                if "function_name" in params:
                    if not self._validate_lambda_name(params["function_name"]):
                        errors.append("Invalid Lambda function name")

                # Security validations
                if "master_password" in params:
                    if len(params["master_password"]) < 8:
                        errors.append("Master password must be at least 8 characters")

                return ValidationResult(
                    valid=len(errors) == 0,
                    errors=errors,
                    warnings=warnings,
                    normalized_params=normalized_params,
                )

            def _validate_ec2_params(
                self, action: str, params: Dict[str, Any]
            ) -> List[str]:
                errors = []

                if action == "create_ec2_instance":
                    if not params.get("instance_type"):
                        errors.append("instance_type is required")
                    elif not self._validate_instance_type(params["instance_type"]):
                        errors.append("Invalid instance type")

                    if not params.get("ami_id"):
                        errors.append("ami_id is required")
                    elif not params["ami_id"].startswith("ami-"):
                        errors.append("Invalid AMI ID format")

                return errors

            def _validate_rds_params(
                self, action: str, params: Dict[str, Any]
            ) -> List[str]:
                errors = []

                if action == "create_rds_instance":
                    required_fields = [
                        "db_instance_identifier",
                        "engine",
                        "db_instance_class",
                        "master_username",
                        "master_password",
                        "allocated_storage",
                    ]
                    for field in required_fields:
                        if not params.get(field):
                            errors.append(
                                f"{field} is required for RDS instance creation"
                            )

                    if (
                        params.get("allocated_storage")
                        and params["allocated_storage"] < 20
                    ):
                        errors.append("Minimum allocated storage is 20 GB")

                return errors

            def _validate_lambda_params(
                self, action: str, params: Dict[str, Any]
            ) -> List[str]:
                errors = []

                if action == "create_lambda_function":
                    required_fields = [
                        "function_name",
                        "runtime",
                        "role",
                        "handler",
                        "code",
                    ]
                    for field in required_fields:
                        if not params.get(field):
                            errors.append(
                                f"{field} is required for Lambda function creation"
                            )

                return errors

            def _validate_iam_params(
                self, action: str, params: Dict[str, Any]
            ) -> List[str]:
                errors = []

                if action == "create_iam_role":
                    if not params.get("role_name"):
                        errors.append("role_name is required")
                    if not params.get("assume_role_policy_document"):
                        errors.append("assume_role_policy_document is required")

                return errors

            def _validate_instance_type(self, instance_type: str) -> bool:
                """Validate EC2 instance type format."""
                # Simplified validation - real implementation would check against AWS API
                valid_families = ["t3", "t2", "m5", "m4", "c5", "c4", "r5", "r4"]
                valid_sizes = [
                    "nano",
                    "micro",
                    "small",
                    "medium",
                    "large",
                    "xlarge",
                    "2xlarge",
                ]

                parts = instance_type.split(".")
                return (
                    len(parts) == 2
                    and parts[0] in valid_families
                    and parts[1] in valid_sizes
                )

            def _validate_instance_id(self, instance_id: str) -> bool:
                """Validate EC2 instance ID format."""
                return instance_id.startswith("i-") and len(instance_id) >= 10

            def _validate_lambda_name(self, function_name: str) -> bool:
                """Validate Lambda function name."""
                import re

                # AWS Lambda function name pattern
                return bool(re.match(r"^[a-zA-Z0-9-_]{1,64}$", function_name))

        return AWSValidator(self)

    async def _execute_action(
        self, action: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute AWS action."""
        try:
            # EC2 Actions
            if action == "create_ec2_instance":
                return await self._create_ec2_instance(params)
            elif action == "list_ec2_instances":
                return await self._list_ec2_instances(params)
            elif action == "start_ec2_instance":
                return await self._start_ec2_instance(params)
            elif action == "stop_ec2_instance":
                return await self._stop_ec2_instance(params)
            elif action == "terminate_ec2_instance":
                return await self._terminate_ec2_instance(params)

            # RDS Actions
            elif action == "create_rds_instance":
                return await self._create_rds_instance(params)
            elif action == "list_rds_instances":
                return await self._list_rds_instances(params)
            elif action == "delete_rds_instance":
                return await self._delete_rds_instance(params)

            # Lambda Actions
            elif action == "create_lambda_function":
                return await self._create_lambda_function(params)
            elif action == "list_lambda_functions":
                return await self._list_lambda_functions(params)
            elif action == "invoke_lambda_function":
                return await self._invoke_lambda_function(params)
            elif action == "delete_lambda_function":
                return await self._delete_lambda_function(params)

            # IAM Actions
            elif action == "create_iam_role":
                return await self._create_iam_role(params)
            elif action == "attach_role_policy":
                return await self._attach_role_policy(params)
            elif action == "detach_role_policy":
                return await self._detach_role_policy(params)

            # Security Group Actions
            elif action == "create_security_group":
                return await self._create_security_group(params)
            elif action == "authorize_security_group_ingress":
                return await self._authorize_security_group_ingress(params)

            # Utility Actions
            elif action == "get_account_info":
                return await self._get_account_info(params)
            elif action == "estimate_costs":
                return await self._estimate_costs(params)

            else:
                raise ToolError(f"Unknown action: {action}")

        except botocore.exceptions.ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            raise ToolError(f"AWS API error ({error_code}): {error_message}")
        except Exception as e:
            raise ToolError(f"AWS operation failed: {e}")

    # EC2 Implementation Methods
    async def _create_ec2_instance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create an EC2 instance."""
        ec2 = self._clients["ec2"]

        create_params = {
            "ImageId": params["ami_id"],
            "InstanceType": params["instance_type"],
            "MinCount": 1,
            "MaxCount": 1,
        }

        # Optional parameters
        if params.get("key_name"):
            create_params["KeyName"] = params["key_name"]
        if params.get("security_groups"):
            create_params["SecurityGroups"] = params["security_groups"]
        if params.get("subnet_id"):
            create_params["SubnetId"] = params["subnet_id"]
        if params.get("user_data"):
            create_params["UserData"] = params["user_data"]

        response = ec2.run_instances(**create_params)
        instance = response["Instances"][0]
        instance_id = instance["InstanceId"]

        # Add tags if provided
        if params.get("tags"):
            ec2.create_tags(
                Resources=[instance_id],
                Tags=[{"Key": k, "Value": v} for k, v in params["tags"].items()],
            )

        return {
            "instance_id": instance_id,
            "instance_type": instance["InstanceType"],
            "state": instance["State"]["Name"],
            "launch_time": instance["LaunchTime"].isoformat(),
            "private_ip": instance.get("PrivateIpAddress"),
            "public_ip": instance.get("PublicIpAddress"),
        }

    async def _list_ec2_instances(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List EC2 instances."""
        ec2 = self._clients["ec2"]

        describe_params = {}
        if params.get("instance_ids"):
            describe_params["InstanceIds"] = params["instance_ids"]
        if params.get("filters"):
            describe_params["Filters"] = [
                {"Name": k, "Values": v if isinstance(v, list) else [v]}
                for k, v in params["filters"].items()
            ]

        response = ec2.describe_instances(**describe_params)

        instances = []
        for reservation in response["Reservations"]:
            for instance in reservation["Instances"]:
                instances.append(
                    {
                        "instance_id": instance["InstanceId"],
                        "instance_type": instance["InstanceType"],
                        "state": instance["State"]["Name"],
                        "launch_time": instance["LaunchTime"].isoformat(),
                        "private_ip": instance.get("PrivateIpAddress"),
                        "public_ip": instance.get("PublicIpAddress"),
                        "tags": {
                            tag["Key"]: tag["Value"] for tag in instance.get("Tags", [])
                        },
                    }
                )

        return {"instances": instances, "count": len(instances)}

    async def _start_ec2_instance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start an EC2 instance."""
        ec2 = self._clients["ec2"]
        instance_id = params["instance_id"]

        response = ec2.start_instances(InstanceIds=[instance_id])
        starting_instance = response["StartingInstances"][0]

        return {
            "instance_id": instance_id,
            "previous_state": starting_instance["PreviousState"]["Name"],
            "current_state": starting_instance["CurrentState"]["Name"],
        }

    async def _stop_ec2_instance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Stop an EC2 instance."""
        ec2 = self._clients["ec2"]
        instance_id = params["instance_id"]

        response = ec2.stop_instances(InstanceIds=[instance_id])
        stopping_instance = response["StoppingInstances"][0]

        return {
            "instance_id": instance_id,
            "previous_state": stopping_instance["PreviousState"]["Name"],
            "current_state": stopping_instance["CurrentState"]["Name"],
        }

    async def _terminate_ec2_instance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Terminate an EC2 instance."""
        ec2 = self._clients["ec2"]
        instance_id = params["instance_id"]

        response = ec2.terminate_instances(InstanceIds=[instance_id])
        terminating_instance = response["TerminatingInstances"][0]

        return {
            "instance_id": instance_id,
            "previous_state": terminating_instance["PreviousState"]["Name"],
            "current_state": terminating_instance["CurrentState"]["Name"],
        }

    # RDS Implementation Methods
    async def _create_rds_instance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create an RDS database instance."""
        rds = self._clients["rds"]

        create_params = {
            "DBInstanceIdentifier": params["db_instance_identifier"],
            "Engine": params["engine"],
            "DBInstanceClass": params["db_instance_class"],
            "MasterUsername": params["master_username"],
            "MasterUserPassword": params["master_password"],
            "AllocatedStorage": params["allocated_storage"],
        }

        # Optional parameters
        if params.get("vpc_security_group_ids"):
            create_params["VpcSecurityGroupIds"] = params["vpc_security_group_ids"]
        if params.get("db_subnet_group_name"):
            create_params["DBSubnetGroupName"] = params["db_subnet_group_name"]

        response = rds.create_db_instance(**create_params)
        db_instance = response["DBInstance"]

        # Add tags if provided
        if params.get("tags"):
            rds.add_tags_to_resource(
                ResourceName=db_instance["DBInstanceArn"],
                Tags=[{"Key": k, "Value": v} for k, v in params["tags"].items()],
            )

        return {
            "db_instance_identifier": db_instance["DBInstanceIdentifier"],
            "engine": db_instance["Engine"],
            "db_instance_class": db_instance["DBInstanceClass"],
            "db_instance_status": db_instance["DBInstanceStatus"],
            "allocated_storage": db_instance["AllocatedStorage"],
            "endpoint": db_instance.get("Endpoint", {}).get("Address"),
            "port": db_instance.get("Endpoint", {}).get("Port"),
        }

    async def _list_rds_instances(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List RDS database instances."""
        rds = self._clients["rds"]

        describe_params = {}
        if params.get("db_instance_identifier"):
            describe_params["DBInstanceIdentifier"] = params["db_instance_identifier"]

        response = rds.describe_db_instances(**describe_params)

        instances = []
        for db_instance in response["DBInstances"]:
            instances.append(
                {
                    "db_instance_identifier": db_instance["DBInstanceIdentifier"],
                    "engine": db_instance["Engine"],
                    "db_instance_class": db_instance["DBInstanceClass"],
                    "db_instance_status": db_instance["DBInstanceStatus"],
                    "allocated_storage": db_instance["AllocatedStorage"],
                    "endpoint": db_instance.get("Endpoint", {}).get("Address"),
                    "port": db_instance.get("Endpoint", {}).get("Port"),
                    "creation_time": db_instance["InstanceCreateTime"].isoformat(),
                }
            )

        return {"db_instances": instances, "count": len(instances)}

    async def _delete_rds_instance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete an RDS database instance."""
        rds = self._clients["rds"]

        delete_params = {
            "DBInstanceIdentifier": params["db_instance_identifier"],
            "SkipFinalSnapshot": params.get("skip_final_snapshot", True),
        }

        if not delete_params["SkipFinalSnapshot"] and params.get(
            "final_db_snapshot_identifier"
        ):
            delete_params["FinalDBSnapshotIdentifier"] = params[
                "final_db_snapshot_identifier"
            ]

        response = rds.delete_db_instance(**delete_params)
        db_instance = response["DBInstance"]

        return {
            "db_instance_identifier": db_instance["DBInstanceIdentifier"],
            "db_instance_status": db_instance["DBInstanceStatus"],
            "deletion_time": datetime.utcnow().isoformat(),
        }

    # Lambda Implementation Methods
    async def _create_lambda_function(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Lambda function."""
        lambda_client = self._clients["lambda"]

        create_params = {
            "FunctionName": params["function_name"],
            "Runtime": params["runtime"],
            "Role": params["role"],
            "Handler": params["handler"],
            "Code": params["code"],
        }

        # Optional parameters
        if params.get("description"):
            create_params["Description"] = params["description"]
        if params.get("timeout"):
            create_params["Timeout"] = params["timeout"]
        if params.get("memory_size"):
            create_params["MemorySize"] = params["memory_size"]
        if params.get("environment"):
            create_params["Environment"] = params["environment"]
        if params.get("tags"):
            create_params["Tags"] = params["tags"]

        response = lambda_client.create_function(**create_params)

        return {
            "function_name": response["FunctionName"],
            "function_arn": response["FunctionArn"],
            "runtime": response["Runtime"],
            "role": response["Role"],
            "handler": response["Handler"],
            "code_size": response["CodeSize"],
            "description": response.get("Description", ""),
            "timeout": response["Timeout"],
            "memory_size": response["MemorySize"],
            "last_modified": response["LastModified"],
            "state": response["State"],
        }

    async def _list_lambda_functions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List Lambda functions."""
        lambda_client = self._clients["lambda"]

        list_params = {}
        if params.get("function_version"):
            list_params["FunctionVersion"] = params["function_version"]
        if params.get("marker"):
            list_params["Marker"] = params["marker"]
        if params.get("max_items"):
            list_params["MaxItems"] = params["max_items"]

        response = lambda_client.list_functions(**list_params)

        functions = []
        for func in response["Functions"]:
            functions.append(
                {
                    "function_name": func["FunctionName"],
                    "function_arn": func["FunctionArn"],
                    "runtime": func["Runtime"],
                    "role": func["Role"],
                    "handler": func["Handler"],
                    "code_size": func["CodeSize"],
                    "description": func.get("Description", ""),
                    "timeout": func["Timeout"],
                    "memory_size": func["MemorySize"],
                    "last_modified": func["LastModified"],
                    "state": func.get(
                        "State", "Active"
                    ),  # Default to Active for LocalStack compatibility
                }
            )

        return {
            "functions": functions,
            "count": len(functions),
            "next_marker": response.get("NextMarker"),
        }

    async def _invoke_lambda_function(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a Lambda function."""
        lambda_client = self._clients["lambda"]

        invoke_params = {
            "FunctionName": params["function_name"],
        }

        if params.get("payload"):
            invoke_params["Payload"] = json.dumps(params["payload"])
        if params.get("invocation_type"):
            invoke_params["InvocationType"] = params["invocation_type"]

        response = lambda_client.invoke(**invoke_params)

        # Read the response payload
        payload = response["Payload"].read()
        if payload:
            try:
                payload = json.loads(payload.decode("utf-8"))
            except json.JSONDecodeError:
                payload = payload.decode("utf-8")

        return {
            "status_code": response["StatusCode"],
            "payload": payload,
            "executed_version": response.get("ExecutedVersion"),
            "log_result": response.get("LogResult"),
        }

    async def _delete_lambda_function(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a Lambda function."""
        lambda_client = self._clients["lambda"]

        delete_params = {
            "FunctionName": params["function_name"],
        }

        if params.get("qualifier"):
            delete_params["Qualifier"] = params["qualifier"]

        lambda_client.delete_function(**delete_params)

        return {
            "function_name": params["function_name"],
            "deleted": True,
            "deletion_time": datetime.utcnow().isoformat(),
        }

    # IAM Implementation Methods
    async def _create_iam_role(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create an IAM role."""
        iam = self._clients["iam"]

        create_params = {
            "RoleName": params["role_name"],
            "AssumeRolePolicyDocument": json.dumps(
                params["assume_role_policy_document"]
            ),
        }

        # Optional parameters
        if params.get("description"):
            create_params["Description"] = params["description"]
        if params.get("max_session_duration"):
            create_params["MaxSessionDuration"] = params["max_session_duration"]
        if params.get("path"):
            create_params["Path"] = params["path"]
        if params.get("tags"):
            create_params["Tags"] = [
                {"Key": k, "Value": v} for k, v in params["tags"].items()
            ]

        response = iam.create_role(**create_params)
        role = response["Role"]

        return {
            "role_name": role["RoleName"],
            "role_arn": role["Arn"],
            "role_id": role["RoleId"],
            "path": role["Path"],
            "create_date": role["CreateDate"].isoformat(),
            "assume_role_policy_document": role["AssumeRolePolicyDocument"],
        }

    async def _attach_role_policy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Attach a policy to an IAM role."""
        iam = self._clients["iam"]

        iam.attach_role_policy(
            RoleName=params["role_name"], PolicyArn=params["policy_arn"]
        )

        return {
            "role_name": params["role_name"],
            "policy_arn": params["policy_arn"],
            "attached": True,
        }

    async def _detach_role_policy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detach a policy from an IAM role."""
        iam = self._clients["iam"]

        iam.detach_role_policy(
            RoleName=params["role_name"], PolicyArn=params["policy_arn"]
        )

        return {
            "role_name": params["role_name"],
            "policy_arn": params["policy_arn"],
            "detached": True,
        }

    # Security Group Implementation Methods
    async def _create_security_group(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a security group."""
        ec2 = self._clients["ec2"]

        create_params = {
            "GroupName": params["group_name"],
            "Description": params["description"],
        }

        if params.get("vpc_id"):
            create_params["VpcId"] = params["vpc_id"]

        response = ec2.create_security_group(**create_params)
        group_id = response["GroupId"]

        # Add tags if provided
        if params.get("tags"):
            ec2.create_tags(
                Resources=[group_id],
                Tags=[{"Key": k, "Value": v} for k, v in params["tags"].items()],
            )

        return {
            "group_id": group_id,
            "group_name": params["group_name"],
            "description": params["description"],
            "vpc_id": params.get("vpc_id"),
        }

    async def _authorize_security_group_ingress(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add ingress rules to a security group."""
        ec2 = self._clients["ec2"]

        ec2.authorize_security_group_ingress(
            GroupId=params["group_id"], IpPermissions=params["ip_permissions"]
        )

        return {
            "group_id": params["group_id"],
            "rules_added": len(params["ip_permissions"]),
            "ip_permissions": params["ip_permissions"],
        }

    # Utility Implementation Methods
    async def _get_account_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get AWS account information."""
        sts = self._clients["sts"]

        identity = sts.get_caller_identity()

        return {
            "account_id": identity["Account"],
            "user_id": identity["UserId"],
            "arn": identity["Arn"],
            "region": self._region,
        }

    async def _estimate_costs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate costs for multiple AWS resources."""
        resources = params["resources"]
        duration_hours = params.get("duration_hours", 24 * 30)  # Default to 30 days

        total_cost = 0.0
        resource_costs = []

        for resource in resources:
            resource_type = resource.get("type")
            resource_params = resource.get("params", {})

            if resource_type == "ec2":
                cost_estimate = await self.estimate_cost(
                    "create_ec2_instance", resource_params
                )
            elif resource_type == "rds":
                cost_estimate = await self.estimate_cost(
                    "create_rds_instance", resource_params
                )
            elif resource_type == "lambda":
                cost_estimate = await self.estimate_cost(
                    "create_lambda_function", resource_params
                )
            else:
                continue

            # Scale by duration
            scaled_cost = cost_estimate.estimated_cost * (duration_hours / (24 * 30))
            total_cost += scaled_cost

            resource_costs.append(
                {
                    "resource_type": resource_type,
                    "estimated_cost": scaled_cost,
                    "cost_breakdown": cost_estimate.cost_breakdown,
                }
            )

        return {
            "total_estimated_cost": total_cost,
            "duration_hours": duration_hours,
            "resource_costs": resource_costs,
            "currency": "USD",
            "disclaimer": "Estimates are approximate and may vary based on actual usage",
        }

    async def _execute_rollback(self, execution_id: str) -> Dict[str, Any]:
        """Execute rollback operation for AWS resources."""
        # For AWS, rollback is complex and depends on the specific resource
        # This is a simplified implementation
        self.logger.info(f"Rollback requested for execution: {execution_id}")

        # In a real implementation, we would:
        # 1. Look up the execution details from a database
        # 2. Identify the resources created
        # 3. Delete/terminate them in reverse order

        return {
            "execution_id": execution_id,
            "rollback_status": "not_implemented",
            "message": "AWS rollback requires specific resource tracking implementation",
        }

    async def _get_supported_actions(self) -> List[str]:
        """Get list of supported AWS actions."""
        schema = await self.get_schema()
        return list(schema.actions.keys())

    async def cleanup(self) -> None:
        """Clean up AWS connections."""
        self._clients.clear()
        self._session = None
        self.logger.info("AWS tool cleanup completed")
