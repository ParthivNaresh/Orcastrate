"""
Multi-cloud tool implementation.

This module provides a unified tool interface for managing resources across
multiple cloud providers using the multi-cloud abstraction layer.
"""

from datetime import datetime
from typing import Any, Union

from .base import CostEstimate, Tool, ToolConfig, ToolError, ToolSchema
from .multicloud import (
    CloudProvider,
    ComputeSpec,
    DatabaseSpec,
    MultiCloudManager,
    NetworkSpec,
    ResourceType,
    SecurityGroupSpec,
    StorageSpec,
)
from .multicloud.providers import AWSProvider
from .multicloud.registry import register_provider


class MultiCloudTool(Tool):
    """
    Multi-cloud tool for unified resource management across cloud providers.

    This tool provides a single interface for provisioning and managing
    resources across AWS, GCP, and Azure using common abstractions.
    """

    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.manager = MultiCloudManager()
        self._initialized_providers: set[CloudProvider] = set()
        self._provider_configs: dict[str, dict[str, Any]] = {}

    async def get_schema(self) -> ToolSchema:
        """Return the multi-cloud tool schema."""
        return ToolSchema(
            name="multicloud",
            description="Multi-cloud resource management tool",
            version=self.config.version,
            actions={
                # Resource provisioning actions
                "provision_compute": {
                    "description": "Provision a compute instance on any cloud provider",
                    "parameters": {
                        "provider": {
                            "type": "string",
                            "required": True,
                            "enum": ["aws", "gcp", "azure"],
                            "description": "Cloud provider to use",
                        },
                        "name": {"type": "string", "required": True},
                        "region": {"type": "string", "required": True},
                        "instance_size": {
                            "type": "string",
                            "required": True,
                            "enum": [
                                "nano",
                                "micro",
                                "small",
                                "medium",
                                "large",
                                "xlarge",
                            ],
                        },
                        "image": {"type": "string", "required": True},
                        "security_groups": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                        "ssh_key_name": {"type": "string"},
                        "user_data": {"type": "string"},
                        "subnet_id": {"type": "string"},
                        "public_ip": {"type": "boolean", "default": True},
                        "storage_size_gb": {"type": "integer", "default": 20},
                        "storage_type": {"type": "string", "default": "standard"},
                        "tags": {"type": "object", "default": {}},
                        "provider_specific": {"type": "object", "default": {}},
                    },
                },
                "provision_database": {
                    "description": "Provision a managed database on any cloud provider",
                    "parameters": {
                        "provider": {
                            "type": "string",
                            "required": True,
                            "enum": ["aws", "gcp", "azure"],
                        },
                        "name": {"type": "string", "required": True},
                        "region": {"type": "string", "required": True},
                        "engine": {
                            "type": "string",
                            "required": True,
                            "enum": ["mysql", "postgres", "redis"],
                        },
                        "version": {"type": "string", "required": True},
                        "instance_size": {
                            "type": "string",
                            "required": True,
                            "enum": ["micro", "small", "medium", "large", "xlarge"],
                        },
                        "storage_size_gb": {"type": "integer", "required": True},
                        "storage_type": {"type": "string", "default": "standard"},
                        "backup_retention_days": {"type": "integer", "default": 7},
                        "multi_az": {"type": "boolean", "default": False},
                        "publicly_accessible": {"type": "boolean", "default": False},
                        "database_name": {"type": "string"},
                        "username": {"type": "string", "default": "admin"},
                        "password": {"type": "string"},
                        "subnet_group": {"type": "string"},
                        "security_groups": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                        "tags": {"type": "object", "default": {}},
                        "provider_specific": {"type": "object", "default": {}},
                    },
                },
                "provision_storage": {
                    "description": "Provision object storage on any cloud provider",
                    "parameters": {
                        "provider": {
                            "type": "string",
                            "required": True,
                            "enum": ["aws", "gcp", "azure"],
                        },
                        "name": {"type": "string", "required": True},
                        "region": {"type": "string", "required": True},
                        "storage_class": {"type": "string", "default": "standard"},
                        "versioning": {"type": "boolean", "default": False},
                        "encryption": {"type": "boolean", "default": True},
                        "public_read": {"type": "boolean", "default": False},
                        "tags": {"type": "object", "default": {}},
                        "provider_specific": {"type": "object", "default": {}},
                    },
                },
                "setup_networking": {
                    "description": "Set up networking resources on any cloud provider",
                    "parameters": {
                        "provider": {
                            "type": "string",
                            "required": True,
                            "enum": ["aws", "gcp", "azure"],
                        },
                        "name": {"type": "string", "required": True},
                        "region": {"type": "string", "required": True},
                        "cidr_block": {"type": "string", "required": True},
                        "enable_dns_hostnames": {"type": "boolean", "default": True},
                        "enable_dns_resolution": {"type": "boolean", "default": True},
                        "internet_gateway": {"type": "boolean", "default": True},
                        "subnets": {
                            "type": "array",
                            "items": {"type": "object"},
                            "default": [],
                        },
                        "tags": {"type": "object", "default": {}},
                        "provider_specific": {"type": "object", "default": {}},
                    },
                },
                "create_security_group": {
                    "description": "Create a security group on any cloud provider",
                    "parameters": {
                        "provider": {
                            "type": "string",
                            "required": True,
                            "enum": ["aws", "gcp", "azure"],
                        },
                        "name": {"type": "string", "required": True},
                        "region": {"type": "string", "required": True},
                        "description": {"type": "string", "required": True},
                        "vpc_id": {"type": "string"},
                        "ingress_rules": {
                            "type": "array",
                            "items": {"type": "object"},
                            "default": [],
                        },
                        "egress_rules": {
                            "type": "array",
                            "items": {"type": "object"},
                            "default": [],
                        },
                        "tags": {"type": "object", "default": {}},
                        "provider_specific": {"type": "object", "default": {}},
                    },
                },
                # Resource management actions
                "list_resources": {
                    "description": "List resources across all providers or for a specific provider",
                    "parameters": {
                        "provider": {"type": "string", "enum": ["aws", "gcp", "azure"]},
                        "resource_type": {
                            "type": "string",
                            "enum": [
                                "compute",
                                "database",
                                "storage",
                                "network",
                                "security_group",
                            ],
                        },
                        "region": {"type": "string"},
                    },
                },
                "get_resource": {
                    "description": "Get details of a specific resource",
                    "parameters": {
                        "resource_id": {"type": "string", "required": True},
                        "resource_type": {
                            "type": "string",
                            "required": True,
                            "enum": [
                                "compute",
                                "database",
                                "storage",
                                "network",
                                "security_group",
                            ],
                        },
                    },
                },
                "start_resource": {
                    "description": "Start a stopped resource",
                    "parameters": {
                        "resource_id": {"type": "string", "required": True},
                    },
                },
                "stop_resource": {
                    "description": "Stop a running resource",
                    "parameters": {
                        "resource_id": {"type": "string", "required": True},
                    },
                },
                "terminate_resource": {
                    "description": "Terminate/delete a resource",
                    "parameters": {
                        "resource_id": {"type": "string", "required": True},
                    },
                },
                # Multi-cloud operations
                "create_deployment": {
                    "description": "Create a multi-cloud deployment with resources across providers",
                    "parameters": {
                        "deployment_name": {"type": "string", "required": True},
                        "description": {"type": "string", "default": ""},
                        "resources": {
                            "type": "array",
                            "required": True,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "provider": {
                                        "type": "string",
                                        "enum": ["aws", "gcp", "azure"],
                                    },
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "compute",
                                            "database",
                                            "storage",
                                            "network",
                                        ],
                                    },
                                    "spec": {"type": "object"},
                                },
                            },
                        },
                    },
                },
                "compare_costs": {
                    "description": "Compare costs for a resource across multiple providers",
                    "parameters": {
                        "resource_type": {
                            "type": "string",
                            "required": True,
                            "enum": ["compute", "database", "storage"],
                        },
                        "spec": {"type": "object", "required": True},
                        "providers": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["aws", "gcp", "azure"],
                            },
                            "default": ["aws"],
                        },
                    },
                },
                "optimize_deployment": {
                    "description": "Analyze a deployment and provide cost optimization recommendations",
                    "parameters": {
                        "deployment_id": {"type": "string", "required": True},
                    },
                },
                # Provider management
                "register_provider": {
                    "description": "Register a cloud provider with credentials",
                    "parameters": {
                        "provider": {
                            "type": "string",
                            "required": True,
                            "enum": ["aws", "gcp", "azure"],
                        },
                        "credentials": {"type": "object", "required": True},
                        "config": {"type": "object", "default": {}},
                    },
                },
                "list_providers": {
                    "description": "List registered cloud providers",
                    "parameters": {},
                },
                "validate_provider": {
                    "description": "Validate provider credentials and connectivity",
                    "parameters": {
                        "provider": {
                            "type": "string",
                            "required": True,
                            "enum": ["aws", "gcp", "azure"],
                        },
                    },
                },
            },
        )

    async def _execute_action(
        self, action: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a multi-cloud action."""

        # Resource provisioning actions
        if action == "provision_compute":
            return await self._provision_compute(params)
        elif action == "provision_database":
            return await self._provision_database(params)
        elif action == "provision_storage":
            return await self._provision_storage(params)
        elif action == "setup_networking":
            return await self._setup_networking(params)
        elif action == "create_security_group":
            return await self._create_security_group(params)

        # Resource management actions
        elif action == "list_resources":
            return await self._list_resources(params)
        elif action == "get_resource":
            return await self._get_resource(params)
        elif action == "start_resource":
            return await self._start_resource(params)
        elif action == "stop_resource":
            return await self._stop_resource(params)
        elif action == "terminate_resource":
            return await self._terminate_resource(params)

        # Multi-cloud operations
        elif action == "create_deployment":
            return await self._create_deployment(params)
        elif action == "compare_costs":
            return await self._compare_costs(params)
        elif action == "optimize_deployment":
            return await self._optimize_deployment(params)

        # Provider management
        elif action == "register_provider":
            return await self._register_provider(params)
        elif action == "list_providers":
            return await self._list_providers(params)
        elif action == "validate_provider":
            return await self._validate_provider(params)

        else:
            raise ToolError(f"Unknown action: {action}")

    async def _ensure_provider_initialized(self, provider_name: str) -> None:
        """Ensure a cloud provider is initialized."""
        provider_enum = CloudProvider(provider_name)

        if provider_enum not in self._initialized_providers:
            if provider_enum == CloudProvider.AWS:
                # Register AWS provider if not already registered
                register_provider(CloudProvider.AWS, AWSProvider)

                # Create AWS provider instance using stored config if available
                aws_config = self.config.environment.get("aws", {})
                if provider_name in self._provider_configs:
                    stored_config = self._provider_configs[provider_name]
                    # Merge stored credentials and config with environment config
                    aws_config.update(stored_config.get("config", {}))
                    if "credentials" in stored_config:
                        aws_config["credentials"] = stored_config["credentials"]

                aws_provider = AWSProvider(aws_config)
                await aws_provider.initialize()
                self.manager.register_provider(aws_provider)

                self._initialized_providers.add(provider_enum)
            else:
                raise ToolError(f"Provider {provider_name} not yet implemented")

    def _parse_resource_type(self, resource_type_str: str) -> ResourceType:
        """Parse resource type string to enum."""
        type_mapping = {
            "compute": ResourceType.COMPUTE,
            "database": ResourceType.DATABASE,
            "storage": ResourceType.STORAGE,
            "network": ResourceType.NETWORK,
            "security_group": ResourceType.SECURITY_GROUP,
        }
        return type_mapping[resource_type_str]

    async def _provision_compute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Provision a compute instance."""
        provider_name = params["provider"]
        await self._ensure_provider_initialized(provider_name)

        spec = ComputeSpec(
            name=params["name"],
            region=params["region"],
            instance_size=params["instance_size"],
            image=params["image"],
            security_groups=params.get("security_groups", []),
            ssh_key_name=params.get("ssh_key_name"),
            user_data=params.get("user_data"),
            subnet_id=params.get("subnet_id"),
            public_ip=params.get("public_ip", True),
            storage_size_gb=params.get("storage_size_gb", 20),
            storage_type=params.get("storage_type", "standard"),
            tags=params.get("tags", {}),
            provider_specific=params.get("provider_specific", {}),
        )

        provider_enum = CloudProvider(provider_name)
        resource = await self.manager.provision_resource(provider_enum, spec)

        return {
            "resource_id": resource.id,
            "name": resource.name,
            "provider": resource.provider.value,
            "resource_type": resource.resource_type.value,
            "state": resource.state.value,
            "region": resource.region,
            "provider_resource_id": resource.provider_resource_id,
            "compute_details": resource.compute_details,
            "estimated_cost_per_month": (
                float(resource.estimated_cost_per_month)
                if resource.estimated_cost_per_month
                else None
            ),
            "created_at": resource.created_at.isoformat(),
        }

    async def _provision_database(self, params: dict[str, Any]) -> dict[str, Any]:
        """Provision a database instance."""
        provider_name = params["provider"]
        await self._ensure_provider_initialized(provider_name)

        spec = DatabaseSpec(
            name=params["name"],
            region=params["region"],
            engine=params["engine"],
            version=params["version"],
            instance_size=params["instance_size"],
            storage_size_gb=params["storage_size_gb"],
            storage_type=params.get("storage_type", "standard"),
            backup_retention_days=params.get("backup_retention_days", 7),
            multi_az=params.get("multi_az", False),
            publicly_accessible=params.get("publicly_accessible", False),
            database_name=params.get("database_name"),
            username=params.get("username", "admin"),
            password=params.get("password"),
            subnet_group=params.get("subnet_group"),
            security_groups=params.get("security_groups", []),
            tags=params.get("tags", {}),
            provider_specific=params.get("provider_specific", {}),
        )

        provider_enum = CloudProvider(provider_name)
        resource = await self.manager.provision_resource(provider_enum, spec)

        return {
            "resource_id": resource.id,
            "name": resource.name,
            "provider": resource.provider.value,
            "resource_type": resource.resource_type.value,
            "state": resource.state.value,
            "region": resource.region,
            "provider_resource_id": resource.provider_resource_id,
            "database_details": resource.database_details,
            "estimated_cost_per_month": (
                float(resource.estimated_cost_per_month)
                if resource.estimated_cost_per_month
                else None
            ),
            "created_at": resource.created_at.isoformat(),
        }

    async def _provision_storage(self, params: dict[str, Any]) -> dict[str, Any]:
        """Provision storage."""
        provider_name = params["provider"]
        await self._ensure_provider_initialized(provider_name)

        spec = StorageSpec(
            name=params["name"],
            region=params["region"],
            storage_class=params.get("storage_class", "standard"),
            versioning=params.get("versioning", False),
            encryption=params.get("encryption", True),
            public_read=params.get("public_read", False),
            tags=params.get("tags", {}),
            provider_specific=params.get("provider_specific", {}),
        )

        provider_enum = CloudProvider(provider_name)
        resource = await self.manager.provision_resource(provider_enum, spec)

        return {
            "resource_id": resource.id,
            "name": resource.name,
            "provider": resource.provider.value,
            "resource_type": resource.resource_type.value,
            "state": resource.state.value,
            "region": resource.region,
            "provider_resource_id": resource.provider_resource_id,
            "storage_details": resource.storage_details,
            "created_at": resource.created_at.isoformat(),
        }

    async def _setup_networking(self, params: dict[str, Any]) -> dict[str, Any]:
        """Set up networking."""
        provider_name = params["provider"]
        await self._ensure_provider_initialized(provider_name)

        spec = NetworkSpec(
            name=params["name"],
            region=params["region"],
            cidr_block=params["cidr_block"],
            enable_dns_hostnames=params.get("enable_dns_hostnames", True),
            enable_dns_resolution=params.get("enable_dns_resolution", True),
            internet_gateway=params.get("internet_gateway", True),
            subnets=params.get("subnets", []),
            tags=params.get("tags", {}),
            provider_specific=params.get("provider_specific", {}),
        )

        provider_enum = CloudProvider(provider_name)
        resource = await self.manager.provision_resource(provider_enum, spec)

        return {
            "resource_id": resource.id,
            "name": resource.name,
            "provider": resource.provider.value,
            "resource_type": resource.resource_type.value,
            "state": resource.state.value,
            "region": resource.region,
            "provider_resource_id": resource.provider_resource_id,
            "network_details": resource.network_details,
            "created_at": resource.created_at.isoformat(),
        }

    async def _create_security_group(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create a security group."""
        provider_name = params["provider"]
        await self._ensure_provider_initialized(provider_name)

        spec = SecurityGroupSpec(
            name=params["name"],
            region=params["region"],
            description=params["description"],
            vpc_id=params.get("vpc_id"),
            ingress_rules=params.get("ingress_rules", []),
            egress_rules=params.get("egress_rules", []),
            tags=params.get("tags", {}),
            provider_specific=params.get("provider_specific", {}),
        )

        provider_enum = CloudProvider(provider_name)
        resource = await self.manager.provision_resource(provider_enum, spec)

        return {
            "resource_id": resource.id,
            "name": resource.name,
            "provider": resource.provider.value,
            "resource_type": resource.resource_type.value,
            "state": resource.state.value,
            "region": resource.region,
            "provider_resource_id": resource.provider_resource_id,
            "created_at": resource.created_at.isoformat(),
        }

    async def _list_resources(self, params: dict[str, Any]) -> dict[str, Any]:
        """List resources."""
        provider_filter = params.get("provider")
        resource_type_filter = params.get("resource_type")

        all_resources = []

        # If provider specified, only list from that provider
        if provider_filter:
            await self._ensure_provider_initialized(provider_filter)
            provider_enum = CloudProvider(provider_filter)
            provider = self.manager.get_provider(provider_enum)

            if provider:
                resource_type_enum = None
                if resource_type_filter:
                    resource_type_enum = self._parse_resource_type(resource_type_filter)

                resources = await provider.list_resources(resource_type_enum)
                all_resources.extend(resources)
        else:
            # List from all registered providers
            for provider_enum in self._initialized_providers:
                provider = self.manager.get_provider(provider_enum)
                if provider:
                    resource_type_enum = None
                    if resource_type_filter:
                        resource_type_enum = self._parse_resource_type(
                            resource_type_filter
                        )

                    resources = await provider.list_resources(resource_type_enum)
                    all_resources.extend(resources)

        # Convert to serializable format
        resources_data = []
        for resource in all_resources:
            resources_data.append(
                {
                    "resource_id": resource.id,
                    "name": resource.name,
                    "provider": resource.provider.value,
                    "resource_type": resource.resource_type.value,
                    "state": resource.state.value,
                    "region": resource.region,
                    "provider_resource_id": resource.provider_resource_id,
                    "created_at": resource.created_at.isoformat(),
                    "tags": resource.tags,
                    "estimated_cost_per_month": (
                        float(resource.estimated_cost_per_month)
                        if resource.estimated_cost_per_month
                        else None
                    ),
                }
            )

        return {
            "resources": resources_data,
            "total_count": len(resources_data),
            "filters": {
                "provider": provider_filter,
                "resource_type": resource_type_filter,
            },
        }

    async def _get_resource(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get resource details."""
        resource_id = params["resource_id"]
        resource_type_str = params["resource_type"]
        resource_type = self._parse_resource_type(resource_type_str)

        # Determine provider from resource ID prefix
        if resource_id.startswith("compute-") or resource_id.startswith("database-"):
            provider_enum = CloudProvider.AWS  # For now, assume AWS
            await self._ensure_provider_initialized("aws")
            provider = self.manager.get_provider(provider_enum)

            if provider:
                resource = await provider.get_resource(resource_id, resource_type)

                return {
                    "resource_id": resource.id,
                    "name": resource.name,
                    "provider": resource.provider.value,
                    "resource_type": resource.resource_type.value,
                    "state": resource.state.value,
                    "region": resource.region,
                    "provider_resource_id": resource.provider_resource_id,
                    "provider_details": resource.provider_details,
                    "compute_details": resource.compute_details,
                    "database_details": resource.database_details,
                    "storage_details": resource.storage_details,
                    "network_details": resource.network_details,
                    "estimated_cost_per_month": (
                        float(resource.estimated_cost_per_month)
                        if resource.estimated_cost_per_month
                        else None
                    ),
                    "created_at": resource.created_at.isoformat(),
                    "tags": resource.tags,
                }

        raise ToolError(f"Resource {resource_id} not found")

    async def _start_resource(self, params: dict[str, Any]) -> dict[str, Any]:
        """Start a resource."""
        resource_id = params["resource_id"]

        # Determine provider from resource ID
        provider_enum = CloudProvider.AWS  # For now, assume AWS
        await self._ensure_provider_initialized("aws")
        provider = self.manager.get_provider(provider_enum)

        if provider:
            resource = await provider.start_resource(resource_id)
            return {
                "resource_id": resource.id,
                "name": resource.name,
                "state": resource.state.value,
                "message": f"Resource {resource_id} started successfully",
            }

        raise ToolError(f"Cannot start resource {resource_id}")

    async def _stop_resource(self, params: dict[str, Any]) -> dict[str, Any]:
        """Stop a resource."""
        resource_id = params["resource_id"]

        # Determine provider from resource ID
        provider_enum = CloudProvider.AWS  # For now, assume AWS
        await self._ensure_provider_initialized("aws")
        provider = self.manager.get_provider(provider_enum)

        if provider:
            resource = await provider.stop_resource(resource_id)
            return {
                "resource_id": resource.id,
                "name": resource.name,
                "state": resource.state.value,
                "message": f"Resource {resource_id} stopped successfully",
            }

        raise ToolError(f"Cannot stop resource {resource_id}")

    async def _terminate_resource(self, params: dict[str, Any]) -> dict[str, Any]:
        """Terminate a resource."""
        resource_id = params["resource_id"]

        # Determine provider from resource ID
        provider_enum = CloudProvider.AWS  # For now, assume AWS
        await self._ensure_provider_initialized("aws")
        provider = self.manager.get_provider(provider_enum)

        if provider:
            resource = await provider.terminate_resource(resource_id)
            return {
                "resource_id": resource.id,
                "name": resource.name,
                "state": resource.state.value,
                "message": f"Resource {resource_id} terminated successfully",
            }

        raise ToolError(f"Cannot terminate resource {resource_id}")

    async def _create_deployment(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create a multi-cloud deployment."""
        deployment_name = params["deployment_name"]
        description = params.get("description", "")
        resources_spec = params["resources"]

        # Parse resource specifications
        resources_to_deploy: list[
            tuple[
                CloudProvider,
                Union[ComputeSpec, DatabaseSpec, StorageSpec, NetworkSpec],
            ]
        ] = []
        for resource_spec in resources_spec:
            provider_name = resource_spec["provider"]
            resource_type = resource_spec["type"]
            spec_data = resource_spec["spec"]

            await self._ensure_provider_initialized(provider_name)
            provider_enum = CloudProvider(provider_name)

            # Create appropriate spec object based on type
            spec: Union[ComputeSpec, DatabaseSpec, StorageSpec, NetworkSpec]
            if resource_type == "compute":
                spec = ComputeSpec(**spec_data)
            elif resource_type == "database":
                spec = DatabaseSpec(**spec_data)
            elif resource_type == "storage":
                spec = StorageSpec(**spec_data)
            elif resource_type == "network":
                spec = NetworkSpec(**spec_data)
            else:
                raise ToolError(f"Unsupported resource type: {resource_type}")

            resources_to_deploy.append((provider_enum, spec))

        # Create deployment
        deployment = await self.manager.create_deployment(
            deployment_name, resources_to_deploy, description
        )

        return {
            "deployment_id": deployment.id,
            "deployment_name": deployment.name,
            "description": deployment.description,
            "providers": [p.value for p in deployment.providers],
            "resource_count": len(deployment.resources),
            "total_estimated_cost_per_month": (
                float(deployment.total_estimated_cost_per_month)
                if deployment.total_estimated_cost_per_month
                else None
            ),
            "created_at": deployment.created_at.isoformat(),
            "resources": [
                {
                    "resource_id": r.id,
                    "name": r.name,
                    "provider": r.provider.value,
                    "resource_type": r.resource_type.value,
                    "state": r.state.value,
                }
                for r in deployment.resources
            ],
        }

    async def _compare_costs(self, params: dict[str, Any]) -> dict[str, Any]:
        """Compare costs across providers."""
        resource_type = params["resource_type"]
        spec_data = params["spec"]
        providers = params.get("providers", ["aws"])

        # Create spec object
        spec: Union[ComputeSpec, DatabaseSpec, StorageSpec]
        if resource_type == "compute":
            spec = ComputeSpec(**spec_data)
        elif resource_type == "database":
            spec = DatabaseSpec(**spec_data)
        elif resource_type == "storage":
            spec = StorageSpec(**spec_data)
        else:
            raise ToolError(
                f"Cost comparison not supported for resource type: {resource_type}"
            )

        # Ensure providers are initialized
        provider_enums = []
        for provider_name in providers:
            await self._ensure_provider_initialized(provider_name)
            provider_enums.append(CloudProvider(provider_name))

        # Compare costs
        cost_estimates = await self.manager.compare_costs(spec, provider_enums)

        # Convert to serializable format
        results = {}
        for provider_enum, estimate in cost_estimates.items():
            results[provider_enum.value] = {
                "estimated_cost": float(estimate.estimated_cost),
                "currency": estimate.currency,
                "confidence": estimate.confidence,
                "cost_breakdown": estimate.cost_breakdown,
            }

        # Find cheapest option
        cheapest_provider = (
            min(cost_estimates.items(), key=lambda x: x[1].estimated_cost)[0].value
            if cost_estimates
            else None
        )

        return {
            "resource_type": resource_type,
            "cost_estimates": results,
            "cheapest_provider": cheapest_provider,
            "comparison_timestamp": datetime.utcnow().isoformat(),
        }

    async def _optimize_deployment(self, params: dict[str, Any]) -> dict[str, Any]:
        """Optimize a deployment."""
        deployment_id = params["deployment_id"]
        optimization_result = await self.manager.optimize_deployment(deployment_id)
        return optimization_result

    async def _register_provider(self, params: dict[str, Any]) -> dict[str, Any]:
        """Register a cloud provider."""
        provider_name = params["provider"]
        credentials = params["credentials"]
        config = params.get("config", {})

        # Provider configuration is handled during initialization
        # The credentials are passed during provider initialization

        # Store credentials and config for provider initialization
        self._provider_configs[provider_name] = {
            "credentials": credentials,
            "config": config,
        }

        # Initialize provider
        await self._ensure_provider_initialized(provider_name)

        return {
            "provider": provider_name,
            "status": "registered",
            "message": f"Provider {provider_name} registered successfully",
        }

    async def _list_providers(self, params: dict[str, Any]) -> dict[str, Any]:
        """List registered providers."""
        return {
            "registered_providers": [p.value for p in self._initialized_providers],
            "available_providers": ["aws", "gcp", "azure"],
        }

    async def _validate_provider(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate a provider."""
        provider_name = params["provider"]

        try:
            await self._ensure_provider_initialized(provider_name)
            return {
                "provider": provider_name,
                "status": "valid",
                "message": f"Provider {provider_name} is configured and accessible",
            }
        except Exception as e:
            return {
                "provider": provider_name,
                "status": "invalid",
                "message": f"Provider validation failed: {str(e)}",
            }

    async def cleanup(self) -> None:
        """Clean up multi-cloud tool resources."""
        self._initialized_providers.clear()
        self.logger.info("Multi-cloud tool cleanup completed")

    async def estimate_cost(self, action: str, params: dict[str, Any]) -> CostEstimate:
        """Estimate the cost of executing an action."""
        # Default cost estimation - could be enhanced with actual cost calculation
        return CostEstimate(estimated_cost=10.00, currency="USD", confidence=0.5)

    async def _create_client(self) -> Any:
        """Create and configure the underlying client."""
        # Multi-cloud tool doesn't use a single client
        return None

    async def _create_validator(self) -> Any:
        """Create and configure the parameter validator."""
        # Use built-in validation based on schema
        return None

    async def _execute_rollback(self, execution_id: str) -> dict[str, Any]:
        """Execute rollback operation."""
        # Rollback not implemented for multi-cloud operations
        return {"message": f"Rollback not supported for execution {execution_id}"}

    async def _get_supported_actions(self) -> list[str]:
        """Get list of supported actions."""
        schema = await self.get_schema()
        return list(schema.actions.keys())
