"""
Base classes and interfaces for multi-cloud abstraction.

This module defines the core abstractions that enable unified management
of resources across different cloud providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional, Sequence


class CloudProvider(Enum):
    """Supported cloud providers."""

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class ResourceState(Enum):
    """Common resource states across cloud providers."""

    PENDING = "pending"
    CREATING = "creating"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    ERROR = "error"
    UNKNOWN = "unknown"


class ResourceType(Enum):
    """Types of cloud resources."""

    COMPUTE = "compute"
    DATABASE = "database"
    STORAGE = "storage"
    NETWORK = "network"
    SECURITY_GROUP = "security_group"
    LOAD_BALANCER = "load_balancer"


@dataclass
class ResourceSpec:
    """Base specification for cloud resources."""

    name: str
    region: str
    tags: dict[str, str] = field(default_factory=dict)
    provider_specific: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComputeSpec(ResourceSpec):
    """Specification for compute resources (VMs/instances)."""

    instance_size: str = "micro"  # small, medium, large, xlarge
    image: str = ""
    # Optional fields with defaults
    public_ip: bool = True
    storage_size_gb: int = 20
    storage_type: str = "standard"  # standard, ssd, premium
    security_groups: list[str] = field(default_factory=list)
    ssh_key_name: Optional[str] = None
    user_data: Optional[str] = None
    subnet_id: Optional[str] = None

    # Common instance sizes mapped to provider-specific types
    INSTANCE_SIZE_MAPPING = {
        "nano": {"aws": "t3.nano", "gcp": "e2-micro", "azure": "Standard_B1ls"},
        "micro": {"aws": "t3.micro", "gcp": "e2-small", "azure": "Standard_B1s"},
        "small": {"aws": "t3.small", "gcp": "e2-medium", "azure": "Standard_B2s"},
        "medium": {
            "aws": "t3.medium",
            "gcp": "e2-standard-2",
            "azure": "Standard_B4ms",
        },
        "large": {
            "aws": "t3.large",
            "gcp": "e2-standard-4",
            "azure": "Standard_D4s_v3",
        },
        "xlarge": {
            "aws": "t3.xlarge",
            "gcp": "e2-standard-8",
            "azure": "Standard_D8s_v3",
        },
    }


@dataclass
class DatabaseSpec(ResourceSpec):
    """Specification for managed database resources."""

    engine: str = "mysql"  # mysql, postgres, mongodb, redis
    version: str = "8.0"
    instance_size: str = "micro"  # small, medium, large
    storage_size_gb: int = 20
    storage_type: str = "standard"  # standard, ssd, premium
    backup_retention_days: int = 7
    multi_az: bool = False
    publicly_accessible: bool = False
    username: str = "admin"
    database_name: Optional[str] = None
    password: Optional[str] = None  # Will be auto-generated if None
    subnet_group: Optional[str] = None
    security_groups: list[str] = field(default_factory=list)

    # Database engine mapping
    ENGINE_MAPPING = {
        "mysql": {"aws": "mysql", "gcp": "MYSQL_8_0", "azure": "mysql"},
        "postgres": {"aws": "postgres", "gcp": "POSTGRES_13", "azure": "postgres"},
        "redis": {"aws": "redis", "gcp": "redis", "azure": "redis"},
    }


@dataclass
class StorageSpec(ResourceSpec):
    """Specification for object storage resources."""

    storage_class: str = "standard"  # standard, cold, archive
    versioning: bool = False
    encryption: bool = True
    public_read: bool = False
    lifecycle_rules: list[dict[str, Any]] = field(default_factory=list)
    cors_rules: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class NetworkSpec(ResourceSpec):
    """Specification for network resources (VPC/VNet)."""

    cidr_block: str = "10.0.0.0/16"
    enable_dns_hostnames: bool = True
    enable_dns_resolution: bool = True
    subnets: list[dict[str, Any]] = field(default_factory=list)
    internet_gateway: bool = True


@dataclass
class SecurityGroupSpec(ResourceSpec):
    """Specification for security groups."""

    description: str = "Security group"
    vpc_id: Optional[str] = None
    ingress_rules: list[dict[str, Any]] = field(default_factory=list)
    egress_rules: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MultiCloudResource:
    """Represents a cloud resource with provider-agnostic information."""

    id: str
    name: str
    provider: CloudProvider
    resource_type: ResourceType
    state: ResourceState
    region: str
    created_at: datetime
    tags: dict[str, str] = field(default_factory=dict)

    # Provider-specific details
    provider_resource_id: str = ""
    provider_details: dict[str, Any] = field(default_factory=dict)

    # Resource-specific attributes
    compute_details: Optional[dict[str, Any]] = None
    database_details: Optional[dict[str, Any]] = None
    storage_details: Optional[dict[str, Any]] = None
    network_details: Optional[dict[str, Any]] = None

    # Cost information
    estimated_cost_per_hour: Optional[Decimal] = None
    estimated_cost_per_month: Optional[Decimal] = None


@dataclass
class MultiCloudDeployment:
    """Represents a multi-cloud deployment with resources across providers."""

    id: str
    name: str
    description: str
    resources: list[MultiCloudResource] = field(default_factory=list)
    providers: list[CloudProvider] = field(default_factory=list)
    total_estimated_cost_per_month: Optional[Decimal] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class MultiCloudProvider(ABC):
    """Abstract base class for cloud provider implementations."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the cloud provider with configuration."""
        self.config = config
        self.provider = self._get_provider_type()

    @abstractmethod
    def _get_provider_type(self) -> CloudProvider:
        """Return the cloud provider type."""

    @abstractmethod
    async def provision_compute(self, spec: ComputeSpec) -> MultiCloudResource:
        """Provision a compute resource."""

    @abstractmethod
    async def provision_database(self, spec: DatabaseSpec) -> MultiCloudResource:
        """Provision a database resource."""

    @abstractmethod
    async def provision_storage(self, spec: StorageSpec) -> MultiCloudResource:
        """Provision a storage resource."""

    @abstractmethod
    async def setup_networking(self, spec: NetworkSpec) -> MultiCloudResource:
        """Set up networking resources."""

    @abstractmethod
    async def create_security_group(
        self, spec: SecurityGroupSpec
    ) -> MultiCloudResource:
        """Create a security group."""

    # Resource management operations
    @abstractmethod
    async def get_resource(
        self, resource_id: str, resource_type: ResourceType
    ) -> MultiCloudResource:
        """Get details of a specific resource."""

    @abstractmethod
    async def list_resources(
        self, resource_type: Optional[ResourceType] = None
    ) -> list[MultiCloudResource]:
        """List resources of a specific type or all resources."""

    @abstractmethod
    async def start_resource(self, resource_id: str) -> MultiCloudResource:
        """Start a stopped resource."""

    @abstractmethod
    async def stop_resource(self, resource_id: str) -> MultiCloudResource:
        """Stop a running resource."""

    @abstractmethod
    async def terminate_resource(self, resource_id: str) -> MultiCloudResource:
        """Terminate/delete a resource."""

    # Cost estimation
    @abstractmethod
    async def estimate_cost(self, spec: ResourceSpec) -> Any:
        """Estimate the cost of a resource based on specification."""

    @abstractmethod
    async def get_resource_cost(self, resource_id: str) -> Optional[Decimal]:
        """Get current cost of an existing resource."""

    # Validation and capability checks
    @abstractmethod
    async def validate_spec(self, spec: ResourceSpec) -> dict[str, Any]:
        """Validate a resource specification for this provider."""

    @abstractmethod
    def get_supported_regions(self) -> list[str]:
        """Get list of supported regions for this provider."""

    @abstractmethod
    def get_supported_instance_sizes(self, resource_type: ResourceType) -> list[str]:
        """Get supported instance sizes for a resource type."""


class MultiCloudManager:
    """Main manager class for multi-cloud operations."""

    def __init__(self):
        """Initialize the multi-cloud manager."""
        self.providers: dict[CloudProvider, MultiCloudProvider] = {}
        self.deployments: dict[str, MultiCloudDeployment] = {}

    def register_provider(self, provider: MultiCloudProvider) -> None:
        """Register a cloud provider."""
        self.providers[provider.provider] = provider

    def get_provider(
        self, provider_type: CloudProvider
    ) -> Optional[MultiCloudProvider]:
        """Get a registered cloud provider."""
        return self.providers.get(provider_type)

    async def provision_resource(
        self, provider_type: CloudProvider, spec: ResourceSpec
    ) -> MultiCloudResource:
        """Provision a resource on a specific cloud provider."""
        provider = self.get_provider(provider_type)
        if not provider:
            raise ValueError(f"Provider {provider_type.value} not registered")

        if isinstance(spec, ComputeSpec):
            return await provider.provision_compute(spec)
        elif isinstance(spec, DatabaseSpec):
            return await provider.provision_database(spec)
        elif isinstance(spec, StorageSpec):
            return await provider.provision_storage(spec)
        elif isinstance(spec, NetworkSpec):
            return await provider.setup_networking(spec)
        elif isinstance(spec, SecurityGroupSpec):
            return await provider.create_security_group(spec)
        else:
            raise ValueError(f"Unsupported resource specification type: {type(spec)}")

    async def create_deployment(
        self,
        deployment_name: str,
        resources: Sequence[tuple[CloudProvider, ResourceSpec]],
        description: str = "",
    ) -> MultiCloudDeployment:
        """Create a multi-cloud deployment with resources across providers."""
        deployment_id = f"deploy-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        deployment = MultiCloudDeployment(
            id=deployment_id, name=deployment_name, description=description
        )

        provisioned_resources = []
        total_cost = Decimal("0")

        for provider_type, spec in resources:
            resource = await self.provision_resource(provider_type, spec)
            provisioned_resources.append(resource)

            if resource.estimated_cost_per_month:
                total_cost += resource.estimated_cost_per_month

        deployment.resources = provisioned_resources
        deployment.providers = list(set([r.provider for r in provisioned_resources]))
        deployment.total_estimated_cost_per_month = total_cost

        self.deployments[deployment_id] = deployment
        return deployment

    async def compare_costs(
        self, spec: ResourceSpec, providers: Optional[list[CloudProvider]] = None
    ) -> dict[CloudProvider, Any]:
        """Compare costs for a resource across multiple providers."""
        if providers is None:
            providers = list(self.providers.keys())

        cost_estimates = {}
        for provider_type in providers:
            provider = self.get_provider(provider_type)
            if provider:
                try:
                    estimate = await provider.estimate_cost(spec)
                    cost_estimates[provider_type] = estimate
                except Exception as e:
                    # Log error but continue with other providers
                    print(
                        f"Failed to get cost estimate from {provider_type.value}: {e}"
                    )

        return cost_estimates

    async def optimize_deployment(self, deployment_id: str) -> dict[str, Any]:
        """Analyze a deployment and provide optimization recommendations."""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        recommendations = []
        potential_savings = Decimal("0")

        for resource in deployment.resources:
            # Check if resource could be cheaper on another provider
            if hasattr(resource, "spec"):
                cost_comparison = await self.compare_costs(resource.spec)
                current_cost = resource.estimated_cost_per_month or Decimal("0")

                cheapest_provider = min(
                    cost_comparison.items(),
                    key=lambda x: x[1].estimated_cost,
                    default=(resource.provider, None),
                )

                if (
                    cheapest_provider[1]
                    and cheapest_provider[0] != resource.provider
                    and cheapest_provider[1].estimated_cost < current_cost
                ):
                    savings = current_cost - cheapest_provider[1].estimated_cost
                    recommendations.append(
                        {
                            "resource_id": resource.id,
                            "current_provider": resource.provider.value,
                            "recommended_provider": cheapest_provider[0].value,
                            "current_cost": float(current_cost),
                            "recommended_cost": float(
                                cheapest_provider[1].estimated_cost
                            ),
                            "potential_savings": float(savings),
                        }
                    )
                    potential_savings += savings

        return {
            "deployment_id": deployment_id,
            "current_monthly_cost": float(
                deployment.total_estimated_cost_per_month or 0
            ),
            "potential_monthly_savings": float(potential_savings),
            "recommendations": recommendations,
        }
