"""
Docker tool for container management operations.

This module provides a concrete implementation of the Tool interface for Docker
operations including image management, container lifecycle, and Docker Compose.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

from .base import (
    CostEstimate,
    Tool,
    ToolConfig,
    ToolError,
    ToolSchema,
    ValidationResult,
)


class DockerTool(Tool):
    """
    Docker tool for container management operations.

    Provides functionality for:
    - Image operations (build, pull, push, list)
    - Container lifecycle (run, stop, remove, logs)
    - Docker Compose operations (up, down, scale)
    - Volume and network management
    """

    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self._docker_available = False
        self._compose_available = False
        self._docker_version: Optional[str] = None
        self._compose_cmd: Optional[List[str]] = None

    async def get_schema(self) -> ToolSchema:
        """Return the Docker tool schema."""
        return ToolSchema(
            name="docker",
            description="Docker container management tool",
            version=self.config.version,
            actions={
                "build_image": {
                    "description": "Build a Docker image from Dockerfile",
                    "parameters": {
                        "path": {"type": "string", "required": True},  # Context path
                        "tag": {"type": "string", "required": True},  # Image name/tag
                        "dockerfile": {"type": "string", "default": "Dockerfile"},
                        "build_args": {"type": "object", "default": {}},
                        "no_cache": {"type": "boolean", "default": False},
                        "labels": {"type": "object", "default": {}},
                        # Keep old parameter names for compatibility
                        "context_path": {"type": "string"},
                        "image_name": {"type": "string"},
                    },
                },
                "create_container": {
                    "description": "Create a Docker container without starting it",
                    "parameters": {
                        "image": {"type": "string", "required": True},
                        "name": {"type": "string"},
                        "ports": {"type": "object", "default": {}},
                        "volumes": {"type": "array", "items": {"type": "string"}},
                        "environment": {"type": "object", "default": {}},
                        "labels": {"type": "object", "default": {}},
                        "command": {"type": "string"},
                        "working_dir": {"type": "string"},
                        "user": {"type": "string"},
                    },
                },
                "start_container": {
                    "description": "Start an existing Docker container",
                    "parameters": {
                        "container_id": {"type": "string", "required": True},
                    },
                },
                "run_container": {
                    "description": "Run a Docker container (create and start)",
                    "parameters": {
                        "image": {"type": "string", "required": True},
                        "name": {"type": "string"},
                        "ports": {"type": "array", "items": {"type": "string"}},
                        "volumes": {"type": "array", "items": {"type": "string"}},
                        "environment": {"type": "object", "default": {}},
                        "detached": {"type": "boolean", "default": True},
                        "remove": {"type": "boolean", "default": False},
                        "command": {"type": "string"},
                    },
                },
                "stop_container": {
                    "description": "Stop a running container",
                    "parameters": {
                        "container_id": {"type": "string", "required": True},
                        "timeout": {"type": "integer", "default": 10},
                    },
                },
                "remove_container": {
                    "description": "Remove a container",
                    "parameters": {
                        "container_id": {"type": "string", "required": True},
                        "force": {"type": "boolean", "default": False},
                        "volumes": {"type": "boolean", "default": False},
                    },
                },
                "list_containers": {
                    "description": "List containers",
                    "parameters": {
                        "all": {"type": "boolean", "default": False},
                        "filter": {"type": "string"},
                    },
                },
                "get_logs": {
                    "description": "Get container logs",
                    "parameters": {
                        "container_id": {"type": "string", "required": True},
                        "tail": {"type": "integer", "default": 100},
                        "follow": {"type": "boolean", "default": False},
                        "timestamps": {"type": "boolean", "default": True},
                    },
                },
                "get_container_logs": {
                    "description": "Get container logs (alias for get_logs)",
                    "parameters": {
                        "container_id": {"type": "string", "required": True},
                        "tail": {"type": "integer", "default": 100},
                        "follow": {"type": "boolean", "default": False},
                        "timestamps": {"type": "boolean", "default": True},
                    },
                },
                "execute_command": {
                    "description": "Execute a command in a running container",
                    "parameters": {
                        "container_id": {"type": "string", "required": True},
                        "command": {
                            "type": "array",
                            "items": {"type": "string"},
                            "required": True,
                        },
                        "detach": {"type": "boolean", "default": False},
                        "user": {"type": "string"},
                        "working_dir": {"type": "string"},
                    },
                },
                "compose_up": {
                    "description": "Start services with Docker Compose",
                    "parameters": {
                        "compose_file": {
                            "type": "string",
                            "default": "docker-compose.yml",
                        },
                        "project_name": {"type": "string"},
                        "services": {"type": "array", "items": {"type": "string"}},
                        "detached": {"type": "boolean", "default": True},
                        "build": {"type": "boolean", "default": False},
                    },
                },
                "compose_down": {
                    "description": "Stop and remove services with Docker Compose",
                    "parameters": {
                        "compose_file": {
                            "type": "string",
                            "default": "docker-compose.yml",
                        },
                        "project_name": {"type": "string"},
                        "volumes": {"type": "boolean", "default": False},
                        "remove_orphans": {"type": "boolean", "default": True},
                    },
                },
                "pull_image": {
                    "description": "Pull a Docker image",
                    "parameters": {
                        "image": {"type": "string", "required": True},
                        "tag": {"type": "string", "default": "latest"},
                    },
                },
                "list_images": {
                    "description": "List Docker images",
                    "parameters": {
                        "all": {"type": "boolean", "default": False},
                        "filter": {"type": "string"},
                    },
                },
                "remove_image": {
                    "description": "Remove a Docker image",
                    "parameters": {
                        "image": {"type": "string", "required": True},
                        "force": {"type": "boolean", "default": False},
                        "no_prune": {"type": "boolean", "default": False},
                    },
                },
                "create_network": {
                    "description": "Create a Docker network",
                    "parameters": {
                        "name": {"type": "string", "required": True},
                        "driver": {"type": "string", "default": "bridge"},
                        "subnet": {"type": "string"},
                        "gateway": {"type": "string"},
                        "labels": {"type": "object", "default": {}},
                    },
                },
                "remove_network": {
                    "description": "Remove a Docker network",
                    "parameters": {
                        "network_id": {"type": "string", "required": True},
                    },
                },
                "list_networks": {
                    "description": "List Docker networks",
                    "parameters": {
                        "filter": {"type": "string"},
                    },
                },
                "create_volume": {
                    "description": "Create a Docker volume",
                    "parameters": {
                        "name": {"type": "string", "required": True},
                        "driver": {"type": "string", "default": "local"},
                        "labels": {"type": "object", "default": {}},
                    },
                },
                "remove_volume": {
                    "description": "Remove a Docker volume",
                    "parameters": {
                        "volume_id": {"type": "string"},
                        "name": {"type": "string"},
                        "force": {"type": "boolean", "default": False},
                    },
                },
                "list_volumes": {
                    "description": "List Docker volumes",
                    "parameters": {
                        "filter": {"type": "string"},
                    },
                },
                "inspect_image": {
                    "description": "Inspect a Docker image",
                    "parameters": {
                        "image": {"type": "string", "required": True},
                    },
                },
            },
            required_permissions=["docker"],
            dependencies=["docker", "docker-compose"],
            cost_model={
                "base_cost": 0.0,  # Docker operations are typically free locally
                "image_build_cost_per_mb": 0.001,
                "container_runtime_cost_per_hour": 0.01,
            },
        )

    async def estimate_cost(self, action: str, params: Dict[str, Any]) -> CostEstimate:
        """Estimate cost for Docker operations."""
        base_cost = 0.0
        breakdown = {}

        if action == "build_image":
            # Estimate based on potential image size and build time
            estimated_size_mb = 500  # Default estimate
            base_cost = estimated_size_mb * 0.001
            breakdown["image_build"] = base_cost

        elif action in ["run_container", "compose_up"]:
            # Estimate runtime cost (minimal for local development)
            estimated_hours = 1.0
            base_cost = estimated_hours * 0.01
            breakdown["container_runtime"] = base_cost

        return CostEstimate(
            estimated_cost=base_cost,
            cost_breakdown=breakdown,
            confidence=0.7,
            factors=["Local execution", "Development environment"],
        )

    async def _create_client(self) -> Any:
        """Create Docker client (validate Docker availability)."""
        try:
            # Check if Docker is available
            result = await self._run_command(["docker", "--version"])
            if result["returncode"] != 0:
                raise ToolError("Docker is not available or not running")

            self._docker_version = result["stdout"].strip()
            self._docker_available = True

            # Check Docker Compose availability (try both modern and legacy)
            # Try modern docker compose first
            compose_result = await self._run_command(["docker", "compose", "version"])
            if compose_result["returncode"] == 0:
                self._compose_available = True
                self._compose_cmd = ["docker", "compose"]
            else:
                # Fallback to legacy docker-compose
                compose_result = await self._run_command(
                    ["docker-compose", "--version"]
                )
                if compose_result["returncode"] == 0:
                    self._compose_available = True
                    self._compose_cmd = ["docker-compose"]
                else:
                    self._compose_cmd = None

            return {"docker_available": True, "version": self._docker_version}

        except Exception as e:
            raise ToolError(f"Failed to initialize Docker client: {e}")

    async def _create_validator(self) -> Any:
        """Create parameter validator."""

        class DockerValidator:
            def validate(self, action: str, params: Dict[str, Any]) -> ValidationResult:
                errors = []
                warnings = []
                normalized_params = params.copy()

                if action == "build_image":
                    # Support both old and new parameter names
                    context_path = params.get("path") or params.get("context_path")
                    tag_or_image = params.get("tag") or params.get("image_name")

                    if not context_path:
                        errors.append("context_path is required for build_image")
                    if not tag_or_image:
                        errors.append("tag or image_name is required for build_image")

                    # Normalize parameters for consistency
                    if context_path:
                        normalized_params["context_path"] = context_path
                        normalized_params["path"] = context_path

                    # Don't double-set both image_name and tag if only one was provided
                    # Let the build_image method handle the logic of combining them

                elif action == "create_container":
                    if not params.get("image"):
                        errors.append("image is required for create_container")

                elif action == "start_container":
                    if not params.get("container_id"):
                        errors.append("container_id is required for start_container")

                elif action == "run_container":
                    if not params.get("image"):
                        errors.append("image is required for run_container")

                    # Validate port format
                    ports = params.get("ports", [])
                    for port in ports:
                        if not isinstance(port, str) or ":" not in port:
                            warnings.append(
                                f"Port '{port}' should be in format 'host:container'"
                            )

                elif action in [
                    "stop_container",
                    "remove_container",
                    "get_logs",
                    "get_container_logs",
                    "execute_command",
                ]:
                    # Support both parameter names for backward compatibility
                    container_id = params.get("container_id") or params.get("container")
                    if not container_id:
                        errors.append(f"container_id is required for {action}")
                    else:
                        # Normalize parameter names
                        normalized_params["container_id"] = container_id

                elif action == "execute_command":
                    if not params.get("command"):
                        errors.append("command is required for execute_command")

                elif action == "pull_image":
                    if not params.get("image"):
                        errors.append("image is required for pull_image")

                elif action == "remove_image":
                    if not params.get("image"):
                        errors.append("image is required for remove_image")

                elif action == "create_network":
                    if not params.get("name"):
                        errors.append("name is required for create_network")

                elif action == "remove_network":
                    if not params.get("network_id"):
                        errors.append("network_id is required for remove_network")

                elif action == "create_volume":
                    if not params.get("name"):
                        errors.append("name is required for create_volume")

                elif action == "remove_volume":
                    volume_id = params.get("volume_id") or params.get("name")
                    if not volume_id:
                        errors.append("volume_id or name is required for remove_volume")
                    # Normalize for implementation
                    if volume_id:
                        normalized_params["volume_id"] = volume_id

                elif action == "inspect_image":
                    if not params.get("image"):
                        errors.append("image is required for inspect_image")

                return ValidationResult(
                    valid=len(errors) == 0,
                    errors=errors,
                    warnings=warnings,
                    normalized_params=normalized_params,
                )

        return DockerValidator()

    async def _execute_action(
        self, action: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Docker action."""
        if not self._docker_available:
            raise ToolError("Docker is not available")

        if action == "build_image":
            return await self._build_image(params)
        elif action == "create_container":
            return await self._create_container(params)
        elif action == "start_container":
            return await self._start_container(params)
        elif action == "run_container":
            return await self._run_container(params)
        elif action == "stop_container":
            return await self._stop_container(params)
        elif action == "remove_container":
            return await self._remove_container(params)
        elif action == "list_containers":
            return await self._list_containers(params)
        elif action == "get_logs" or action == "get_container_logs":
            return await self._get_logs(params)
        elif action == "execute_command":
            return await self._execute_command(params)
        elif action == "compose_up":
            return await self._compose_up(params)
        elif action == "compose_down":
            return await self._compose_down(params)
        elif action == "pull_image":
            return await self._pull_image(params)
        elif action == "list_images":
            return await self._list_images(params)
        elif action == "remove_image":
            return await self._remove_image(params)
        elif action == "create_network":
            return await self._create_network(params)
        elif action == "remove_network":
            return await self._remove_network(params)
        elif action == "list_networks":
            return await self._list_networks(params)
        elif action == "create_volume":
            return await self._create_volume(params)
        elif action == "remove_volume":
            return await self._remove_volume(params)
        elif action == "list_volumes":
            return await self._list_volumes(params)
        elif action == "inspect_image":
            return await self._inspect_image(params)
        else:
            raise ToolError(f"Unknown action: {action}")

    async def _build_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build a Docker image."""
        # Support both parameter formats
        context_path = params.get("path") or params.get("context_path")
        image_name = params.get("image_name")
        tag = params.get("tag")

        # Build the full image tag
        if image_name and tag:
            image_tag = f"{image_name}:{tag}"
            user_facing_name = image_tag
        else:
            image_tag = tag or image_name or ""
            user_facing_name = image_tag  # Return what user provided

        if not context_path:
            raise ToolError("context_path or path parameter is required")
        if not image_tag:
            raise ToolError("tag or image_name parameter is required")

        cmd = [
            "docker",
            "build",
            "-t",
            image_tag,
        ]

        # Only add -f if dockerfile is not the default
        dockerfile = params.get("dockerfile", "Dockerfile")
        if dockerfile != "Dockerfile":
            cmd.extend(["-f", dockerfile])

        if params.get("no_cache"):
            cmd.append("--no-cache")

        # Add build args
        for key, value in params.get("build_args", {}).items():
            cmd.extend(["--build-arg", f"{key}={value}"])

        # Add labels
        for key, value in params.get("labels", {}).items():
            cmd.extend(["--label", f"{key}={value}"])

        cmd.append(context_path)

        result = await self._run_command(cmd)

        if result["returncode"] != 0:
            raise ToolError(f"Docker build failed: {result['stderr']}")

        return {
            "image_name": user_facing_name,  # Return what user can use to reference the image
            "tag": user_facing_name,
            "build_output": result["stdout"],
            "success": True,
        }

    async def _run_container(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a Docker container."""
        cmd = ["docker", "run"]

        if params.get("detached", True):
            cmd.append("-d")

        if params.get("remove"):
            cmd.append("--rm")

        if params.get("name"):
            cmd.extend(["--name", params["name"]])

        # Add port mappings
        for port in params.get("ports", []):
            cmd.extend(["-p", port])

        # Add volume mounts
        for volume in params.get("volumes", []):
            cmd.extend(["-v", volume])

        # Add environment variables
        for key, value in params.get("environment", {}).items():
            cmd.extend(["-e", f"{key}={value}"])

        cmd.append(params["image"])

        if params.get("command"):
            cmd.extend(params["command"].split())

        result = await self._run_command(cmd)

        if result["returncode"] != 0:
            raise ToolError(f"Failed to run container: {result['stderr']}")

        container_id = result["stdout"].strip()

        return {
            "container_id": container_id,
            "image": params["image"],
            "success": True,
            "output": result["stdout"],
        }

    async def _create_container(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Docker container without starting it."""
        cmd = ["docker", "create"]

        if params.get("name"):
            cmd.extend(["--name", params["name"]])

        # Add port mappings (expecting dict format like {"80/tcp": None})
        for container_port, host_port in params.get("ports", {}).items():
            if host_port:
                cmd.extend(["-p", f"{host_port}:{container_port.split('/')[0]}"])
            else:
                cmd.extend(["-p", container_port.split("/")[0]])

        # Add volume mounts - handle both array and dict formats
        volumes = params.get("volumes", [])
        if isinstance(volumes, dict):
            # Format: {"volume_name": {"bind": "/path", "mode": "rw"}}
            for volume_name, mount_config in volumes.items():
                if isinstance(mount_config, dict):
                    bind_path = mount_config.get("bind", "")
                    mode = mount_config.get("mode", "rw")
                    cmd.extend(["-v", f"{volume_name}:{bind_path}:{mode}"])
                else:
                    cmd.extend(["-v", f"{volume_name}:{mount_config}"])
        else:
            # Array format: ["volume_name:/path", "host_path:/container_path"]
            for volume in volumes:
                cmd.extend(["-v", volume])

        # Add environment variables
        for key, value in params.get("environment", {}).items():
            cmd.extend(["-e", f"{key}={value}"])

        # Add labels
        for key, value in params.get("labels", {}).items():
            cmd.extend(["--label", f"{key}={value}"])

        # Add working directory
        if params.get("working_dir"):
            cmd.extend(["-w", params["working_dir"]])

        # Add user
        if params.get("user"):
            cmd.extend(["-u", params["user"]])

        cmd.append(params["image"])

        if params.get("command"):
            command = params["command"]
            if isinstance(command, str):
                cmd.extend(command.split())
            elif isinstance(command, list):
                cmd.extend(command)
            else:
                cmd.append(str(command))

        result = await self._run_command(cmd)

        if result["returncode"] != 0:
            raise ToolError(f"Failed to create container: {result['stderr']}")

        container_id = result["stdout"].strip()

        return {
            "container_id": container_id,
            "name": params.get("name"),
            "image": params["image"],
            "success": True,
            "output": result["stdout"],
        }

    async def _start_container(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start an existing Docker container."""
        cmd = ["docker", "start", params["container_id"]]

        result = await self._run_command(cmd)

        if result["returncode"] != 0:
            raise ToolError(f"Failed to start container: {result['stderr']}")

        return {
            "container_id": params["container_id"],
            "success": True,
            "output": result["stdout"],
        }

    async def _stop_container(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Stop a container."""
        cmd = ["docker", "stop"]

        if params.get("timeout"):
            cmd.extend(["-t", str(params["timeout"])])

        # Support both parameter names for backward compatibility
        container_id = params.get("container_id") or params.get("container")
        if not container_id:
            raise ToolError("container_id or container parameter is required")

        cmd.append(container_id)

        result = await self._run_command(cmd)

        return {
            "container_id": container_id,
            "container": container_id,  # For backward compatibility
            "success": result["returncode"] == 0,
            "output": result["stdout"],
        }

    async def _remove_container(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove a container."""
        cmd = ["docker", "rm"]

        if params.get("force"):
            cmd.append("-f")
        if params.get("volumes"):
            cmd.append("-v")

        cmd.append(params["container_id"])

        result = await self._run_command(cmd)

        if result["returncode"] != 0:
            raise ToolError(f"Failed to remove container: {result['stderr']}")

        return {
            "container_id": params["container_id"],
            "success": True,
            "output": result["stdout"],
        }

    async def _list_containers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List containers."""
        cmd = ["docker", "ps", "--format", "json"]

        if params.get("all"):
            cmd.append("-a")
        if params.get("filter"):
            cmd.extend(["--filter", params["filter"]])

        result = await self._run_command(cmd)

        containers = []
        if result["returncode"] == 0 and result["stdout"]:
            for line in result["stdout"].strip().split("\n"):
                if line:
                    try:
                        container_data = json.loads(line)
                        # Normalize field names for consistent API
                        # Docker returns Names as a string, extract the actual name
                        names_str = container_data.get("Names", "")
                        # Names typically comes as "/container-name" so strip the leading slash
                        name = names_str.lstrip("/") if names_str else ""

                        normalized_container = {
                            "container_id": container_data.get("ID", ""),
                            "image": container_data.get("Image", ""),
                            "command": container_data.get("Command", ""),
                            "created": container_data.get("CreatedAt", ""),
                            "status": container_data.get("Status", ""),
                            "state": container_data.get("State", ""),
                            "ports": container_data.get("Ports", ""),
                            "names": names_str,  # Keep original for compatibility
                            "name": name,  # Add parsed name field
                            "size": container_data.get("Size", ""),
                            "labels": container_data.get("Labels", ""),
                            "mounts": container_data.get("Mounts", ""),
                            "networks": container_data.get("Networks", ""),
                        }
                        containers.append(normalized_container)
                    except json.JSONDecodeError:
                        continue

        return {
            "containers": containers,
            "success": result["returncode"] == 0,
        }

    async def _get_logs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get container logs."""
        cmd = ["docker", "logs"]

        if params.get("timestamps"):
            cmd.append("-t")
        if params.get("tail"):
            cmd.extend(["--tail", str(params["tail"])])
        if params.get("follow"):
            cmd.append("-f")

        # Support both parameter names for backward compatibility
        container_id = params.get("container_id") or params.get("container")
        if not container_id:
            raise ToolError("container_id or container parameter is required")

        cmd.append(container_id)

        result = await self._run_command(cmd)

        return {
            "container_id": container_id,
            "container": container_id,  # For backward compatibility
            "logs": result["stdout"],
            "success": result["returncode"] == 0,
        }

    async def _execute_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command in a running container."""
        cmd = ["docker", "exec"]

        if params.get("detach", False):
            cmd.append("-d")

        if params.get("user"):
            cmd.extend(["-u", params["user"]])

        if params.get("working_dir"):
            cmd.extend(["-w", params["working_dir"]])

        cmd.append(params["container_id"])

        # Add the command to execute
        command = params["command"]
        if isinstance(command, str):
            cmd.extend(command.split())
        else:
            cmd.extend(command)

        result = await self._run_command(cmd)

        return {
            "container_id": params["container_id"],
            "command": params["command"],
            "output": result["stdout"],
            "success": result["returncode"] == 0,
            "error": result["stderr"] if result["returncode"] != 0 else None,
        }

    async def _compose_up(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start services with Docker Compose."""
        if not self._compose_available:
            raise ToolError("Docker Compose is not available")
        if not self._compose_cmd:
            raise ToolError("Docker Compose command not available")

        cmd = self._compose_cmd + [
            "-f",
            params.get("compose_file", "docker-compose.yml"),
        ]

        if params.get("project_name"):
            cmd.extend(["-p", params["project_name"]])

        cmd.append("up")

        if params.get("detached", True):
            cmd.append("-d")
        if params.get("build"):
            cmd.append("--build")

        # Add specific services if provided
        if params.get("services"):
            cmd.extend(params["services"])

        result = await self._run_command(cmd)

        return {
            "compose_file": params.get("compose_file", "docker-compose.yml"),
            "services": params.get("services", "all"),
            "success": result["returncode"] == 0,
            "output": result["stdout"],
        }

    async def _compose_down(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Stop and remove services with Docker Compose."""
        if not self._compose_available:
            raise ToolError("Docker Compose is not available")
        if not self._compose_cmd:
            raise ToolError("Docker Compose command not available")

        cmd = self._compose_cmd + [
            "-f",
            params.get("compose_file", "docker-compose.yml"),
        ]

        if params.get("project_name"):
            cmd.extend(["-p", params["project_name"]])

        cmd.append("down")

        if params.get("volumes"):
            cmd.append("-v")
        if params.get("remove_orphans", True):
            cmd.append("--remove-orphans")

        result = await self._run_command(cmd)

        return {
            "compose_file": params.get("compose_file", "docker-compose.yml"),
            "success": result["returncode"] == 0,
            "output": result["stdout"],
        }

    async def _pull_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Pull a Docker image."""
        image = params["image"]

        # Only add tag if image doesn't already have one
        if ":" not in image:
            image_tag = f"{image}:{params.get('tag', 'latest')}"
        else:
            image_tag = image

        cmd = ["docker", "pull", image_tag]

        result = await self._run_command(cmd)

        if result["returncode"] != 0:
            raise ToolError(f"Failed to pull image: {result['stderr']}")

        return {
            "image": image_tag,
            "success": True,
            "output": result["stdout"],
        }

    async def _list_images(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List Docker images."""
        cmd = ["docker", "images", "--format", "json"]

        if params.get("all"):
            cmd.append("-a")
        if params.get("filter"):
            cmd.extend(["--filter", params["filter"]])

        result = await self._run_command(cmd)

        images = []
        if result["returncode"] == 0 and result["stdout"]:
            for line in result["stdout"].strip().split("\n"):
                if line:
                    try:
                        image_data = json.loads(line)
                        # Normalize image format to include tags field
                        repository = image_data.get("Repository", "")
                        tag = image_data.get("Tag", "")

                        # Create full image name
                        if repository and tag and tag != "<none>":
                            full_name = f"{repository}:{tag}"
                            tags = [full_name]
                        else:
                            tags = []

                        normalized_image = {
                            "id": image_data.get("ID", ""),
                            "repository": repository,
                            "tag": tag,
                            "tags": tags,  # This is what the test expects
                            "created": image_data.get("CreatedAt", ""),
                            "size": image_data.get("Size", ""),
                            "virtual_size": image_data.get("VirtualSize", ""),
                        }
                        images.append(normalized_image)
                    except json.JSONDecodeError:
                        continue

        return {
            "images": images,
            "success": result["returncode"] == 0,
        }

    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run a shell command asynchronously."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            return {
                "returncode": process.returncode,
                "stdout": stdout.decode("utf-8"),
                "stderr": stderr.decode("utf-8"),
                "command": " ".join(cmd),
            }

        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "command": " ".join(cmd),
            }

    async def _execute_rollback(self, execution_id: str) -> Dict[str, Any]:
        """Execute rollback operation."""
        # For Docker, rollback might involve stopping/removing containers
        # This would depend on the specific operation that was performed
        return {
            "rollback_type": "docker_operation",
            "execution_id": execution_id,
            "message": "Docker rollback operations are action-specific",
        }

    async def _remove_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove a Docker image."""
        cmd = ["docker", "rmi"]

        if params.get("force"):
            cmd.append("-f")
        if params.get("no_prune"):
            cmd.append("--no-prune")

        cmd.append(params["image"])

        result = await self._run_command(cmd)

        return {
            "image": params["image"],
            "success": result["returncode"] == 0,
            "output": result["stdout"],
            "error": result["stderr"] if result["returncode"] != 0 else None,
        }

    async def _create_network(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Docker network."""
        cmd = ["docker", "network", "create"]

        if params.get("driver"):
            cmd.extend(["--driver", params["driver"]])
        if params.get("subnet"):
            cmd.extend(["--subnet", params["subnet"]])
        if params.get("gateway"):
            cmd.extend(["--gateway", params["gateway"]])

        # Add labels
        for key, value in params.get("labels", {}).items():
            cmd.extend(["--label", f"{key}={value}"])

        cmd.append(params["name"])

        result = await self._run_command(cmd)

        if result["returncode"] != 0:
            raise ToolError(f"Failed to create network: {result['stderr']}")

        network_id = result["stdout"].strip()

        return {
            "network_id": network_id,
            "name": params["name"],
            "success": True,
            "output": result["stdout"],
        }

    async def _remove_network(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove a Docker network."""
        cmd = ["docker", "network", "rm", params["network_id"]]

        result = await self._run_command(cmd)

        if result["returncode"] != 0:
            raise ToolError(f"Failed to remove network: {result['stderr']}")

        return {
            "network_id": params["network_id"],
            "success": True,
            "output": result["stdout"],
        }

    async def _list_networks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List Docker networks."""
        cmd = ["docker", "network", "ls", "--format", "json"]

        if params.get("filter"):
            cmd.extend(["--filter", params["filter"]])

        result = await self._run_command(cmd)

        networks = []
        if result["returncode"] == 0 and result["stdout"]:
            for line in result["stdout"].strip().split("\n"):
                if line:
                    try:
                        network_data = json.loads(line)
                        # Normalize field names
                        normalized_network = {
                            "network_id": network_data.get("ID", ""),
                            "name": network_data.get("Name", ""),
                            "driver": network_data.get("Driver", ""),
                            "scope": network_data.get("Scope", ""),
                            "created": network_data.get("CreatedAt", ""),
                        }
                        networks.append(normalized_network)
                    except json.JSONDecodeError:
                        continue

        return {
            "networks": networks,
            "success": result["returncode"] == 0,
        }

    async def _create_volume(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Docker volume."""
        cmd = ["docker", "volume", "create"]

        if params.get("driver"):
            cmd.extend(["--driver", params["driver"]])

        # Add labels
        for key, value in params.get("labels", {}).items():
            cmd.extend(["--label", f"{key}={value}"])

        cmd.append(params["name"])

        result = await self._run_command(cmd)

        if result["returncode"] != 0:
            raise ToolError(f"Failed to create volume: {result['stderr']}")

        volume_id = result["stdout"].strip()

        return {
            "volume_id": volume_id,
            "name": params["name"],
            "success": True,
            "output": result["stdout"],
        }

    async def _remove_volume(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove a Docker volume."""
        cmd = ["docker", "volume", "rm"]

        if params.get("force"):
            cmd.append("-f")

        cmd.append(params["volume_id"])

        result = await self._run_command(cmd)

        return {
            "volume_id": params["volume_id"],
            "success": result["returncode"] == 0,
            "output": result["stdout"],
            "error": result["stderr"] if result["returncode"] != 0 else None,
        }

    async def _list_volumes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List Docker volumes."""
        cmd = ["docker", "volume", "ls", "--format", "json"]

        if params.get("filter"):
            cmd.extend(["--filter", params["filter"]])

        result = await self._run_command(cmd)

        volumes = []
        if result["returncode"] == 0 and result["stdout"]:
            for line in result["stdout"].strip().split("\n"):
                if line:
                    try:
                        volume_data = json.loads(line)
                        # Normalize field names
                        normalized_volume = {
                            "volume_id": volume_data.get(
                                "Name", ""
                            ),  # Docker volumes use Name as ID
                            "name": volume_data.get("Name", ""),
                            "driver": volume_data.get("Driver", ""),
                            "mountpoint": volume_data.get("Mountpoint", ""),
                            "created": volume_data.get("CreatedAt", ""),
                        }
                        volumes.append(normalized_volume)
                    except json.JSONDecodeError:
                        continue

        return {
            "volumes": volumes,
            "success": result["returncode"] == 0,
        }

    async def _inspect_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect a Docker image."""
        cmd = ["docker", "inspect", params["image"]]

        result = await self._run_command(cmd)

        if result["returncode"] == 0:
            try:
                import json as json_module

                inspect_data = json_module.loads(result["stdout"])
                image_info = inspect_data[0] if inspect_data else {}

                # Flatten the structure to match test expectations
                output = {
                    "image": params["image"],
                    "success": True,
                    "inspect_data": image_info,  # Keep full data for completeness
                }

                # Add specific fields that tests expect
                if image_info:
                    output["architecture"] = image_info.get("Architecture", "")
                    output["config"] = image_info.get("Config", {})
                    output["created"] = image_info.get("Created", "")
                    output["id"] = image_info.get("Id", "")
                    output["size"] = image_info.get("Size", 0)

                return output
            except json_module.JSONDecodeError:
                return {
                    "image": params["image"],
                    "success": False,
                    "error": "Failed to parse inspect output",
                }
        else:
            return {
                "image": params["image"],
                "success": False,
                "error": result["stderr"],
            }

    async def cleanup(self) -> None:
        """Clean up Docker connections and resources."""
        # For Docker, no persistent connections to clean up
        # Reset any cached state
        self._docker_available = False
        self._compose_available = False
        self._docker_version = None
        self.logger.info("Docker tool cleanup completed")

    async def _get_supported_actions(self) -> List[str]:
        """Get list of supported actions."""
        return [
            "build_image",
            "create_container",
            "start_container",
            "run_container",
            "stop_container",
            "remove_container",
            "list_containers",
            "get_logs",
            "get_container_logs",
            "execute_command",
            "compose_up",
            "compose_down",
            "pull_image",
            "list_images",
            "remove_image",
            "create_network",
            "remove_network",
            "list_networks",
            "create_volume",
            "remove_volume",
            "list_volumes",
            "inspect_image",
        ]
