"""
AWS provider implementation for the multi-cloud abstraction layer.

This module adapts the existing AWS tool to the multi-cloud interface,
translating between common resource specifications and AWS-specific parameters.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from ...aws import AWSCloudTool
from ...base import CostEstimate, ToolConfig
from ..base import (
    CloudProvider,
    ComputeSpec,
    DatabaseSpec,
    MultiCloudProvider,
    MultiCloudResource,
    NetworkSpec,
    ResourceSpec,
    ResourceState,
    ResourceType,
    SecurityGroupSpec,
    StorageSpec,
)


class AWSProvider(MultiCloudProvider):
    """AWS implementation of the multi-cloud provider interface."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the AWS provider."""
        super().__init__(config)

        # Create AWS tool configuration
        tool_config = ToolConfig(
            name="aws",
            version="1.0.0",
            enabled=True,
            timeout=config.get("timeout", 60),
            retry_count=config.get("retry_count", 3),
            retry_delay=config.get("retry_delay", 5),
            environment=config.get("environment", {}),
            credentials=config.get("credentials", {}),
        )

        # Initialize the AWS tool
        self.aws_tool = AWSCloudTool(tool_config)

    async def initialize(self) -> None:
        """Initialize the AWS tool."""
        await self.aws_tool.initialize()

    def _get_provider_type(self) -> CloudProvider:
        """Return the AWS provider type."""
        return CloudProvider.AWS

    def _map_aws_state_to_common(self, aws_state: str) -> ResourceState:
        """Map AWS resource states to common resource states."""
        state_mapping = {
            "pending": ResourceState.PENDING,
            "running": ResourceState.RUNNING,
            "shutting-down": ResourceState.STOPPING,
            "terminated": ResourceState.TERMINATED,
            "stopping": ResourceState.STOPPING,
            "stopped": ResourceState.STOPPED,
            "rebooting": ResourceState.RUNNING,
            "creating": ResourceState.CREATING,
            "available": ResourceState.RUNNING,
            "deleting": ResourceState.TERMINATING,
            "deleted": ResourceState.TERMINATED,
            "failed": ResourceState.ERROR,
        }
        return state_mapping.get(aws_state.lower(), ResourceState.UNKNOWN)

    def _get_aws_instance_type(self, instance_size: str) -> str:
        """Get AWS instance type from common instance size."""
        return ComputeSpec.INSTANCE_SIZE_MAPPING.get(instance_size, {}).get(
            "aws", "t3.micro"
        )

    def _get_aws_engine(self, engine: str) -> str:
        """Get AWS database engine from common engine name."""
        return DatabaseSpec.ENGINE_MAPPING.get(engine, {}).get("aws", engine)

    async def provision_compute(self, spec: ComputeSpec) -> MultiCloudResource:
        """Provision an EC2 instance."""
        # Map common specification to AWS parameters
        aws_params = {
            "instance_type": self._get_aws_instance_type(spec.instance_size),
            "ami_id": spec.image,
            "tags": {"Name": spec.name, **spec.tags},
        }

        # Add optional parameters
        if spec.ssh_key_name:
            aws_params["key_name"] = spec.ssh_key_name

        if spec.user_data:
            aws_params["user_data"] = spec.user_data

        if spec.subnet_id:
            aws_params["subnet_id"] = spec.subnet_id

        if spec.security_groups:
            aws_params["security_groups"] = spec.security_groups

        # Add provider-specific parameters
        if spec.provider_specific:
            aws_params.update(spec.provider_specific)

        # Create the instance
        result = await self.aws_tool.execute("create_ec2_instance", aws_params)

        if not result.success:
            raise RuntimeError(f"Failed to create EC2 instance: {result.error}")

        # Create multi-cloud resource
        resource = MultiCloudResource(
            id=f"compute-{result.output['instance_id']}",
            name=spec.name,
            provider=CloudProvider.AWS,
            resource_type=ResourceType.COMPUTE,
            state=self._map_aws_state_to_common(result.output.get("state", "unknown")),
            region=spec.region,
            created_at=datetime.utcnow(),
            tags=spec.tags,
            provider_resource_id=result.output["instance_id"],
            provider_details=result.output,
            compute_details={
                "instance_type": aws_params["instance_type"],
                "ami_id": spec.image,
                "public_ip": spec.public_ip,
                "storage_size_gb": spec.storage_size_gb,
                "storage_type": spec.storage_type,
            },
        )

        # Estimate costs
        try:
            cost_estimate = await self.estimate_cost(spec)
            resource.estimated_cost_per_month = Decimal(
                str(cost_estimate.estimated_cost)
            )
        except Exception as e:
            print(f"Warning: Could not estimate costs: {e}")

        return resource

    async def provision_database(self, spec: DatabaseSpec) -> MultiCloudResource:
        """Provision an RDS instance."""
        # Map common specification to AWS parameters
        aws_engine = self._get_aws_engine(spec.engine)
        aws_instance_class = (
            f"db.{self._get_aws_instance_type(spec.instance_size).replace('t3', 't4g')}"
        )

        aws_params = {
            "db_instance_identifier": spec.name,
            "engine": aws_engine,
            "engine_version": spec.version,
            "db_instance_class": aws_instance_class,
            "allocated_storage": spec.storage_size_gb,
            "master_username": spec.username,
            "master_password": spec.password or self._generate_password(),
            "backup_retention_period": spec.backup_retention_days,
            "multi_az": spec.multi_az,
            "publicly_accessible": spec.publicly_accessible,
            "tags": {"Name": spec.name, **spec.tags},
        }

        # Add optional parameters
        if spec.database_name:
            aws_params["db_name"] = spec.database_name

        if spec.subnet_group:
            aws_params["db_subnet_group_name"] = spec.subnet_group

        if spec.security_groups:
            aws_params["vpc_security_group_ids"] = spec.security_groups

        # Add provider-specific parameters
        if spec.provider_specific:
            aws_params.update(spec.provider_specific)

        # Create the RDS instance
        result = await self.aws_tool.execute("create_rds_instance", aws_params)

        if not result.success:
            raise RuntimeError(f"Failed to create RDS instance: {result.error}")

        # Create multi-cloud resource
        resource = MultiCloudResource(
            id=f"database-{result.output['db_instance_identifier']}",
            name=spec.name,
            provider=CloudProvider.AWS,
            resource_type=ResourceType.DATABASE,
            state=self._map_aws_state_to_common(
                result.output.get("db_instance_status", "unknown")
            ),
            region=spec.region,
            created_at=datetime.utcnow(),
            tags=spec.tags,
            provider_resource_id=result.output["db_instance_identifier"],
            provider_details=result.output,
            database_details={
                "engine": aws_engine,
                "version": spec.version,
                "instance_class": aws_instance_class,
                "storage_size_gb": spec.storage_size_gb,
                "endpoint": result.output.get("endpoint", {}).get("address"),
                "port": result.output.get("endpoint", {}).get("port"),
                "multi_az": spec.multi_az,
            },
        )

        # Estimate costs
        try:
            cost_estimate = await self.estimate_cost(spec)
            resource.estimated_cost_per_month = Decimal(
                str(cost_estimate.estimated_cost)
            )
        except Exception as e:
            print(f"Warning: Could not estimate costs: {e}")

        return resource

    async def provision_storage(self, spec: StorageSpec) -> MultiCloudResource:
        """Provision an S3 bucket."""
        # For S3, we'll use the boto3 client directly since the current AWS tool
        # doesn't have S3 bucket creation methods
        # This is a placeholder implementation

        bucket_name = spec.name.lower().replace("_", "-")  # S3 naming requirements

        # Create multi-cloud resource (S3 creation would be implemented here)
        resource = MultiCloudResource(
            id=f"storage-{bucket_name}",
            name=spec.name,
            provider=CloudProvider.AWS,
            resource_type=ResourceType.STORAGE,
            state=ResourceState.RUNNING,  # S3 buckets are immediately available
            region=spec.region,
            created_at=datetime.utcnow(),
            tags=spec.tags,
            provider_resource_id=bucket_name,
            provider_details={"bucket_name": bucket_name},
            storage_details={
                "storage_class": spec.storage_class,
                "versioning": spec.versioning,
                "encryption": spec.encryption,
                "public_read": spec.public_read,
            },
        )

        return resource

    async def setup_networking(self, spec: NetworkSpec) -> MultiCloudResource:
        """Set up VPC networking."""
        # Create VPC (this would require extending the AWS tool with VPC methods)
        # This is a placeholder implementation

        vpc_id = f"vpc-{uuid.uuid4().hex[:8]}"

        resource = MultiCloudResource(
            id=f"network-{vpc_id}",
            name=spec.name,
            provider=CloudProvider.AWS,
            resource_type=ResourceType.NETWORK,
            state=ResourceState.RUNNING,
            region=spec.region,
            created_at=datetime.utcnow(),
            tags=spec.tags,
            provider_resource_id=vpc_id,
            provider_details={"vpc_id": vpc_id},
            network_details={
                "cidr_block": spec.cidr_block,
                "enable_dns_hostnames": spec.enable_dns_hostnames,
                "enable_dns_resolution": spec.enable_dns_resolution,
                "internet_gateway": spec.internet_gateway,
                "subnets": spec.subnets,
            },
        )

        return resource

    async def create_security_group(
        self, spec: SecurityGroupSpec
    ) -> MultiCloudResource:
        """Create a security group."""
        aws_params = {
            "group_name": spec.name,
            "description": spec.description,
            "tags": {"Name": spec.name, **spec.tags},
        }

        if spec.vpc_id:
            aws_params["vpc_id"] = spec.vpc_id

        # Add provider-specific parameters
        if spec.provider_specific:
            aws_params.update(spec.provider_specific)

        # Create the security group
        result = await self.aws_tool.execute("create_security_group", aws_params)

        if not result.success:
            raise RuntimeError(f"Failed to create security group: {result.error}")

        # Add ingress rules if specified
        if spec.ingress_rules:
            try:
                await self.aws_tool.execute(
                    "authorize_security_group_ingress",
                    {
                        "group_id": result.output["group_id"],
                        "ip_permissions": spec.ingress_rules,
                    },
                )
            except Exception as e:
                print(f"Warning: Failed to add ingress rules: {e}")

        resource = MultiCloudResource(
            id=f"security-group-{result.output['group_id']}",
            name=spec.name,
            provider=CloudProvider.AWS,
            resource_type=ResourceType.SECURITY_GROUP,
            state=ResourceState.RUNNING,
            region=spec.region,
            created_at=datetime.utcnow(),
            tags=spec.tags,
            provider_resource_id=result.output["group_id"],
            provider_details=result.output,
        )

        return resource

    async def get_resource(
        self, resource_id: str, resource_type: ResourceType
    ) -> MultiCloudResource:
        """Get details of a specific resource."""
        # Extract provider resource ID from multi-cloud resource ID
        provider_resource_id = (
            resource_id.split("-", 1)[1] if "-" in resource_id else resource_id
        )

        if resource_type == ResourceType.COMPUTE:
            result = await self.aws_tool.execute("list_ec2_instances", {})
            if result.success:
                for instance in result.output.get("instances", []):
                    if instance["instance_id"] == provider_resource_id:
                        return self._convert_aws_compute_to_multicloud(instance)

        elif resource_type == ResourceType.DATABASE:
            result = await self.aws_tool.execute("list_rds_instances", {})
            if result.success:
                for db in result.output.get("db_instances", []):
                    if db["db_instance_identifier"] == provider_resource_id:
                        return self._convert_aws_database_to_multicloud(db)

        raise ValueError(f"Resource {resource_id} not found")

    async def list_resources(
        self, resource_type: Optional[ResourceType] = None
    ) -> list[MultiCloudResource]:
        """List resources of a specific type or all resources."""
        resources = []

        if resource_type is None or resource_type == ResourceType.COMPUTE:
            # List EC2 instances
            result = await self.aws_tool.execute("list_ec2_instances", {})
            if result.success:
                for instance in result.output.get("instances", []):
                    resources.append(self._convert_aws_compute_to_multicloud(instance))

        if resource_type is None or resource_type == ResourceType.DATABASE:
            # List RDS instances
            result = await self.aws_tool.execute("list_rds_instances", {})
            if result.success:
                for db in result.output.get("db_instances", []):
                    resources.append(self._convert_aws_database_to_multicloud(db))

        return resources

    async def start_resource(self, resource_id: str) -> MultiCloudResource:
        """Start a stopped resource."""
        provider_resource_id = (
            resource_id.split("-", 1)[1] if "-" in resource_id else resource_id
        )

        if resource_id.startswith("compute-"):
            result = await self.aws_tool.execute(
                "start_ec2_instance", {"instance_id": provider_resource_id}
            )
            if result.success:
                return await self.get_resource(resource_id, ResourceType.COMPUTE)

        raise ValueError(f"Cannot start resource {resource_id}")

    async def stop_resource(self, resource_id: str) -> MultiCloudResource:
        """Stop a running resource."""
        provider_resource_id = (
            resource_id.split("-", 1)[1] if "-" in resource_id else resource_id
        )

        if resource_id.startswith("compute-"):
            result = await self.aws_tool.execute(
                "stop_ec2_instance", {"instance_id": provider_resource_id}
            )
            if result.success:
                return await self.get_resource(resource_id, ResourceType.COMPUTE)

        raise ValueError(f"Cannot stop resource {resource_id}")

    async def terminate_resource(self, resource_id: str) -> MultiCloudResource:
        """Terminate/delete a resource."""
        provider_resource_id = (
            resource_id.split("-", 1)[1] if "-" in resource_id else resource_id
        )

        if resource_id.startswith("compute-"):
            result = await self.aws_tool.execute(
                "terminate_ec2_instance", {"instance_id": provider_resource_id}
            )
            if result.success:
                return await self.get_resource(resource_id, ResourceType.COMPUTE)

        elif resource_id.startswith("database-"):
            result = await self.aws_tool.execute(
                "delete_rds_instance",
                {
                    "db_instance_identifier": provider_resource_id,
                    "skip_final_snapshot": True,
                },
            )
            if result.success:
                return await self.get_resource(resource_id, ResourceType.DATABASE)

        raise ValueError(f"Cannot terminate resource {resource_id}")

    async def estimate_cost(self, spec: ResourceSpec) -> CostEstimate:
        """Estimate the cost of a resource based on specification."""
        if isinstance(spec, ComputeSpec):
            # Use AWS tool's cost estimation for EC2
            resources = [
                {
                    "type": "ec2",
                    "params": {
                        "instance_type": self._get_aws_instance_type(spec.instance_size)
                    },
                }
            ]

            result = await self.aws_tool.execute(
                "estimate_costs",
                {"resources": resources, "duration_hours": 24 * 30},  # One month
            )

            if result.success:
                monthly_cost = Decimal(str(result.output["total_estimated_cost"]))

                return CostEstimate(
                    estimated_cost=float(monthly_cost), currency="USD", confidence=0.8
                )

        elif isinstance(spec, DatabaseSpec):
            # Use AWS tool's cost estimation for RDS
            resources = [
                {
                    "type": "rds",
                    "params": {
                        "db_instance_class": f"db.{self._get_aws_instance_type(spec.instance_size).replace('t3', 't4g')}",
                        "allocated_storage": spec.storage_size_gb,
                    },
                }
            ]

            result = await self.aws_tool.execute(
                "estimate_costs", {"resources": resources, "duration_hours": 24 * 30}
            )

            if result.success:
                monthly_cost = Decimal(str(result.output["total_estimated_cost"]))
                monthly_cost / Decimal("720")

                return CostEstimate(
                    estimated_cost=float(monthly_cost), currency="USD", confidence=0.8
                )

        # Default estimation
        return CostEstimate(estimated_cost=10.00, currency="USD", confidence=0.5)

    async def get_resource_cost(self, resource_id: str) -> Optional[Decimal]:
        """Get current cost of an existing resource."""
        # This would require cost tracking implementation
        return None

    async def validate_spec(self, spec: ResourceSpec) -> dict[str, Any]:
        """Validate a resource specification for AWS."""
        validation_result: dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        # Check region
        supported_regions = self.get_supported_regions()
        if spec.region not in supported_regions:
            validation_result["errors"].append(f"Region {spec.region} not supported")
            validation_result["valid"] = False

        if isinstance(spec, ComputeSpec):
            # Validate instance size
            if spec.instance_size not in ComputeSpec.INSTANCE_SIZE_MAPPING:
                validation_result["errors"].append(
                    f"Instance size {spec.instance_size} not supported"
                )
                validation_result["valid"] = False

            # Validate AMI format
            if not spec.image.startswith("ami-"):
                validation_result["warnings"].append(
                    "Image should be an AMI ID (ami-xxxxx)"
                )

        elif isinstance(spec, DatabaseSpec):
            # Validate database engine
            if spec.engine not in DatabaseSpec.ENGINE_MAPPING:
                validation_result["errors"].append(
                    f"Database engine {spec.engine} not supported"
                )
                validation_result["valid"] = False

        return validation_result

    def get_supported_regions(self) -> list[str]:
        """Get list of supported AWS regions."""
        return [
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "eu-central-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ap-northeast-1",
            "ca-central-1",
            "sa-east-1",
        ]

    def get_supported_instance_sizes(self, resource_type: ResourceType) -> list[str]:
        """Get supported instance sizes for a resource type."""
        if resource_type in [ResourceType.COMPUTE, ResourceType.DATABASE]:
            return list(ComputeSpec.INSTANCE_SIZE_MAPPING.keys())
        return []

    def _convert_aws_compute_to_multicloud(
        self, aws_instance: dict[str, Any]
    ) -> MultiCloudResource:
        """Convert AWS EC2 instance to multi-cloud resource."""
        return MultiCloudResource(
            id=f"compute-{aws_instance['instance_id']}",
            name=aws_instance.get("tags", {}).get("Name", aws_instance["instance_id"]),
            provider=CloudProvider.AWS,
            resource_type=ResourceType.COMPUTE,
            state=self._map_aws_state_to_common(aws_instance.get("state", "unknown")),
            region=aws_instance.get("placement", {}).get("availability_zone", "")[:-1],
            created_at=datetime.fromisoformat(
                aws_instance.get("launch_time", datetime.utcnow().isoformat())
            ),
            tags=aws_instance.get("tags", {}),
            provider_resource_id=aws_instance["instance_id"],
            provider_details=aws_instance,
            compute_details={
                "instance_type": aws_instance.get("instance_type"),
                "ami_id": aws_instance.get("image_id"),
                "public_ip": aws_instance.get("public_ip_address"),
                "private_ip": aws_instance.get("private_ip_address"),
            },
        )

    def _convert_aws_database_to_multicloud(
        self, aws_db: dict[str, Any]
    ) -> MultiCloudResource:
        """Convert AWS RDS instance to multi-cloud resource."""
        return MultiCloudResource(
            id=f"database-{aws_db['db_instance_identifier']}",
            name=aws_db.get("tags", {}).get("Name", aws_db["db_instance_identifier"]),
            provider=CloudProvider.AWS,
            resource_type=ResourceType.DATABASE,
            state=self._map_aws_state_to_common(
                aws_db.get("db_instance_status", "unknown")
            ),
            region=aws_db.get("availability_zone", "")[:-1],
            created_at=datetime.fromisoformat(
                aws_db.get("instance_create_time", datetime.utcnow().isoformat())
            ),
            tags=aws_db.get("tags", {}),
            provider_resource_id=aws_db["db_instance_identifier"],
            provider_details=aws_db,
            database_details={
                "engine": aws_db.get("engine"),
                "version": aws_db.get("engine_version"),
                "instance_class": aws_db.get("db_instance_class"),
                "storage_size_gb": aws_db.get("allocated_storage"),
                "endpoint": aws_db.get("endpoint", {}).get("address"),
                "port": aws_db.get("endpoint", {}).get("port"),
                "multi_az": aws_db.get("multi_az", False),
            },
        )

    def _generate_password(self) -> str:
        """Generate a secure password for database instances."""
        import secrets
        import string

        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = "".join(secrets.choice(alphabet) for _ in range(16))
        return password
