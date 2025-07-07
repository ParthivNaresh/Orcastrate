"""
Terraform Infrastructure as Code tool implementation.

High-performance Terraform integration with HCL generation, state management,
plan/apply operations, and comprehensive infrastructure management.
"""

import asyncio
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from .base import CostEstimate, Tool, ToolConfig, ToolError, ToolSchema


class TerraformConfig:
    """Configuration for Terraform operations."""

    def __init__(
        self,
        working_dir: Optional[str] = None,
        terraform_version: str = "latest",
        backend_config: Optional[Dict[str, Any]] = None,
        var_files: Optional[List[str]] = None,
        variables: Optional[Dict[str, Any]] = None,
        parallelism: int = 10,
        auto_approve: bool = False,
    ):
        self.working_dir = working_dir or tempfile.mkdtemp(prefix="terraform_")
        self.terraform_version = terraform_version
        self.backend_config = backend_config or {}
        self.var_files = var_files or []
        self.variables = variables or {}
        self.parallelism = parallelism
        self.auto_approve = auto_approve


class TerraformPlan:
    """Represents a Terraform execution plan."""

    def __init__(
        self,
        plan_file: str,
        changes_summary: Dict[str, int],
        resource_changes: List[Dict[str, Any]],
        plan_output: str,
    ):
        self.plan_file = plan_file
        self.changes_summary = changes_summary  # {"add": 5, "change": 2, "destroy": 1}
        self.resource_changes = resource_changes
        self.plan_output = plan_output
        self.created_at = time.time()

    @property
    def has_changes(self) -> bool:
        """Check if the plan contains any changes."""
        return sum(self.changes_summary.values()) > 0

    @property
    def total_changes(self) -> int:
        """Get total number of changes in the plan."""
        return sum(self.changes_summary.values())


class TerraformState:
    """Represents Terraform state information."""

    def __init__(
        self,
        version: str,
        terraform_version: str,
        resources: List[Dict[str, Any]],
        outputs: Dict[str, Any],
    ):
        self.version = version
        self.terraform_version = terraform_version
        self.resources = resources
        self.outputs = outputs

    @property
    def resource_count(self) -> int:
        """Get the number of resources in state."""
        return len(self.resources)

    def get_resources_by_type(self, resource_type: str) -> List[Dict[str, Any]]:
        """Get all resources of a specific type."""
        return [r for r in self.resources if r.get("type") == resource_type]


class HCLGenerator:
    """Generates HCL (HashiCorp Configuration Language) configurations."""

    @staticmethod
    def generate_provider_config(provider: str, config: Dict[str, Any]) -> str:
        """Generate HCL for a provider configuration."""
        hcl_lines = [f'provider "{provider}" {{']

        for key, value in config.items():
            if isinstance(value, str):
                hcl_lines.append(f'  {key} = "{value}"')
            elif isinstance(value, bool):
                hcl_lines.append(f"  {key} = {str(value).lower()}")
            elif isinstance(value, (int, float)):
                hcl_lines.append(f"  {key} = {value}")
            elif isinstance(value, list):
                formatted_list = "[" + ", ".join(f'"{item}"' for item in value) + "]"
                hcl_lines.append(f"  {key} = {formatted_list}")
            elif isinstance(value, dict):
                hcl_lines.append(f"  {key} = {{")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str):
                        hcl_lines.append(f'    {sub_key} = "{sub_value}"')
                    else:
                        hcl_lines.append(f"    {sub_key} = {sub_value}")
                hcl_lines.append("  }")

        hcl_lines.append("}")
        return "\n".join(hcl_lines)

    @staticmethod
    def generate_resource_config(
        resource_type: str, resource_name: str, config: Dict[str, Any]
    ) -> str:
        """Generate HCL for a resource configuration."""
        hcl_lines = [f'resource "{resource_type}" "{resource_name}" {{']

        for key, value in config.items():
            formatted_value = HCLGenerator._format_hcl_value(value)
            hcl_lines.append(f"  {key} = {formatted_value}")

        hcl_lines.append("}")
        return "\n".join(hcl_lines)

    @staticmethod
    def generate_data_source_config(
        data_type: str, data_name: str, config: Dict[str, Any]
    ) -> str:
        """Generate HCL for a data source configuration."""
        hcl_lines = [f'data "{data_type}" "{data_name}" {{']

        for key, value in config.items():
            formatted_value = HCLGenerator._format_hcl_value(value)
            hcl_lines.append(f"  {key} = {formatted_value}")

        hcl_lines.append("}")
        return "\n".join(hcl_lines)

    @staticmethod
    def generate_output_config(output_name: str, config: Dict[str, Any]) -> str:
        """Generate HCL for an output configuration."""
        hcl_lines = [f'output "{output_name}" {{']

        for key, value in config.items():
            formatted_value = HCLGenerator._format_hcl_value(value)
            hcl_lines.append(f"  {key} = {formatted_value}")

        hcl_lines.append("}")
        return "\n".join(hcl_lines)

    @staticmethod
    def _format_hcl_value(value: Any) -> str:
        """Format a Python value as HCL syntax."""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            formatted_items = [HCLGenerator._format_hcl_value(item) for item in value]
            return "[" + ", ".join(formatted_items) + "]"
        elif isinstance(value, dict):
            lines = ["{"]
            for k, v in value.items():
                formatted_v = HCLGenerator._format_hcl_value(v)
                lines.append(f"    {k} = {formatted_v}")
            lines.append("  }")
            return "\n  ".join(lines)
        else:
            return f'"{str(value)}"'


class TerraformTool(Tool):
    """High-performance Terraform Infrastructure as Code tool."""

    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.terraform_config = self._create_terraform_config()
        self.hcl_generator = HCLGenerator()
        self._terraform_binary: Optional[str] = None

    def _create_terraform_config(self) -> TerraformConfig:
        """Create Terraform configuration from tool config."""
        env = self.config.environment

        return TerraformConfig(
            working_dir=env.get("working_dir"),
            terraform_version=env.get("terraform_version", "latest"),
            backend_config=env.get("backend_config", {}),
            var_files=env.get("var_files", []),
            variables=env.get("variables", {}),
            parallelism=env.get("parallelism", 10),
            auto_approve=env.get("auto_approve", False),
        )

    async def initialize(self) -> None:
        """Initialize the Terraform tool."""
        await super().initialize()

        # Ensure working directory exists
        os.makedirs(self.terraform_config.working_dir, exist_ok=True)

        # Check for Terraform binary
        await self._ensure_terraform_binary()

        self.logger.info(
            f"Terraform tool initialized with working directory: {self.terraform_config.working_dir}"
        )

    async def _ensure_terraform_binary(self) -> None:
        """Ensure Terraform binary is available."""
        try:
            result = await self._run_command(["terraform", "version"])
            if result["success"]:
                self._terraform_binary = "terraform"
                self.logger.info(f"Found Terraform binary: {result['output']}")
            else:
                raise ToolError("Terraform binary not found in PATH")
        except Exception as e:
            raise ToolError(f"Failed to find Terraform binary: {e}")

    async def get_schema(self) -> ToolSchema:
        """Return Terraform tool schema."""
        return ToolSchema(
            name="terraform",
            description="Infrastructure as Code tool for managing cloud resources",
            version=self.config.version,
            actions={
                "init": {
                    "description": "Initialize Terraform working directory",
                    "parameters": {
                        "backend_config": {"type": "object", "default": {}},
                        "upgrade": {"type": "boolean", "default": False},
                    },
                },
                "plan": {
                    "description": "Create an execution plan",
                    "parameters": {
                        "var_file": {"type": "string"},
                        "variables": {"type": "object", "default": {}},
                        "target": {"type": "array"},
                        "destroy": {"type": "boolean", "default": False},
                        "out": {"type": "string"},
                    },
                },
                "apply": {
                    "description": "Apply the execution plan",
                    "parameters": {
                        "plan_file": {"type": "string"},
                        "auto_approve": {"type": "boolean", "default": False},
                        "variables": {"type": "object", "default": {}},
                        "target": {"type": "array"},
                    },
                },
                "destroy": {
                    "description": "Destroy infrastructure",
                    "parameters": {
                        "auto_approve": {"type": "boolean", "default": False},
                        "variables": {"type": "object", "default": {}},
                        "target": {"type": "array"},
                    },
                },
                "validate": {
                    "description": "Validate configuration files",
                    "parameters": {},
                },
                "show": {
                    "description": "Show current state or plan",
                    "parameters": {
                        "plan_file": {"type": "string"},
                        "json": {"type": "boolean", "default": True},
                    },
                },
                "state_list": {
                    "description": "List resources in state",
                    "parameters": {},
                },
                "state_show": {
                    "description": "Show a resource in state",
                    "parameters": {
                        "address": {"type": "string", "required": True},
                    },
                },
                "import": {
                    "description": "Import existing infrastructure",
                    "parameters": {
                        "address": {"type": "string", "required": True},
                        "id": {"type": "string", "required": True},
                    },
                },
                "output": {
                    "description": "Show output values",
                    "parameters": {
                        "json": {"type": "boolean", "default": True},
                    },
                },
                "generate_hcl": {
                    "description": "Generate HCL configuration",
                    "parameters": {
                        "resources": {"type": "array", "required": True},
                        "providers": {"type": "object", "default": {}},
                        "outputs": {"type": "object", "default": {}},
                    },
                },
                "workspace_list": {
                    "description": "List workspaces",
                    "parameters": {},
                },
                "workspace_select": {
                    "description": "Select a workspace",
                    "parameters": {
                        "workspace": {"type": "string", "required": True},
                    },
                },
                "workspace_new": {
                    "description": "Create a new workspace",
                    "parameters": {
                        "workspace": {"type": "string", "required": True},
                    },
                },
                "format": {
                    "description": "Format configuration files",
                    "parameters": {
                        "check": {"type": "boolean", "default": False},
                        "diff": {"type": "boolean", "default": False},
                    },
                },
                "get_state": {
                    "description": "Get current Terraform state",
                    "parameters": {},
                },
            },
        )

    async def _execute_action(
        self, action: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Terraform action."""
        action_methods = {
            "init": self._init_action,
            "plan": self._plan_action,
            "apply": self._apply_action,
            "destroy": self._destroy_action,
            "validate": self._validate_action,
            "show": self._show_action,
            "state_list": self._state_list_action,
            "state_show": self._state_show_action,
            "import": self._import_action,
            "output": self._output_action,
            "generate_hcl": self._generate_hcl_action,
            "workspace_list": self._workspace_list_action,
            "workspace_select": self._workspace_select_action,
            "workspace_new": self._workspace_new_action,
            "format": self._format_action,
            "get_state": self._get_state_action,
        }

        if action not in action_methods:
            raise ToolError(f"Unknown action: {action}")

        return await action_methods[action](params)

    async def _init_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize Terraform working directory."""
        backend_config = params.get("backend_config", {})
        upgrade = params.get("upgrade", False)

        cmd = ["terraform", "init"]

        if upgrade:
            cmd.append("-upgrade")

        for key, value in backend_config.items():
            cmd.extend(["-backend-config", f"{key}={value}"])

        result = await self._run_terraform_command(cmd)

        return {
            "success": result["success"],
            "message": "Terraform initialized successfully"
            if result["success"]
            else None,
            "output": result["output"],
            "error": result.get("error"),
        }

    async def _plan_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create Terraform execution plan."""
        cmd = ["terraform", "plan"]

        # Add variables
        variables = params.get("variables", {})
        for key, value in variables.items():
            cmd.extend(["-var", f"{key}={value}"])

        # Add variable files
        var_file = params.get("var_file")
        if var_file:
            cmd.extend(["-var-file", var_file])

        # Add targets
        targets = params.get("target", [])
        for target in targets:
            cmd.extend(["-target", target])

        # Add destroy flag
        if params.get("destroy", False):
            cmd.append("-destroy")

        # Add output file
        out_file = params.get("out")
        if out_file:
            cmd.extend(["-out", out_file])

        # Always add JSON output for parsing
        cmd.extend(["-json"])

        result = await self._run_terraform_command(cmd)

        if result["success"]:
            # Parse plan output to extract changes summary
            plan_data = self._parse_plan_output(result["output"])

            return {
                "success": True,
                "plan": plan_data,
                "output": result["output"],
                "has_changes": plan_data.has_changes,
                "changes_summary": plan_data.changes_summary,
            }
        else:
            return {
                "success": False,
                "error": result.get("error"),
                "output": result["output"],
            }

    async def _apply_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Terraform execution plan."""
        cmd = ["terraform", "apply"]

        # Check for plan file
        plan_file = params.get("plan_file")
        if plan_file:
            cmd.append(plan_file)
        else:
            # Add variables if no plan file
            variables = params.get("variables", {})
            for key, value in variables.items():
                cmd.extend(["-var", f"{key}={value}"])

            # Add targets
            targets = params.get("target", [])
            for target in targets:
                cmd.extend(["-target", target])

        # Add auto-approve
        if params.get("auto_approve", self.terraform_config.auto_approve):
            cmd.append("-auto-approve")

        result = await self._run_terraform_command(cmd)

        return {
            "success": result["success"],
            "message": "Terraform apply completed successfully"
            if result["success"]
            else None,
            "output": result["output"],
            "error": result.get("error"),
        }

    async def _destroy_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Destroy Terraform-managed infrastructure."""
        cmd = ["terraform", "destroy"]

        # Add variables
        variables = params.get("variables", {})
        for key, value in variables.items():
            cmd.extend(["-var", f"{key}={value}"])

        # Add targets
        targets = params.get("target", [])
        for target in targets:
            cmd.extend(["-target", target])

        # Add auto-approve
        if params.get("auto_approve", self.terraform_config.auto_approve):
            cmd.append("-auto-approve")

        result = await self._run_terraform_command(cmd)

        return {
            "success": result["success"],
            "message": "Terraform destroy completed successfully"
            if result["success"]
            else None,
            "output": result["output"],
            "error": result.get("error"),
        }

    async def _validate_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Terraform configuration files."""
        result = await self._run_terraform_command(["terraform", "validate", "-json"])

        if result["success"]:
            try:
                validation_data = json.loads(result["output"])
                return {
                    "success": True,
                    "valid": validation_data.get("valid", False),
                    "error_count": validation_data.get("error_count", 0),
                    "warning_count": validation_data.get("warning_count", 0),
                    "diagnostics": validation_data.get("diagnostics", []),
                }
            except json.JSONDecodeError:
                return {
                    "success": True,
                    "valid": True,
                    "output": result["output"],
                }
        else:
            return {
                "success": False,
                "error": result.get("error"),
                "output": result["output"],
            }

    async def _show_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Show current state or plan."""
        cmd = ["terraform", "show"]

        plan_file = params.get("plan_file")
        if plan_file:
            cmd.append(plan_file)

        if params.get("json", True):
            cmd.append("-json")

        result = await self._run_terraform_command(cmd)

        if result["success"]:
            try:
                if params.get("json", True):
                    show_data = json.loads(result["output"])
                    return {
                        "success": True,
                        "data": show_data,
                        "format": "json",
                    }
                else:
                    return {
                        "success": True,
                        "data": result["output"],
                        "format": "text",
                    }
            except json.JSONDecodeError:
                return {
                    "success": True,
                    "data": result["output"],
                    "format": "text",
                }
        else:
            return {
                "success": False,
                "error": result.get("error"),
            }

    async def _state_list_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List resources in Terraform state."""
        result = await self._run_terraform_command(["terraform", "state", "list"])

        if result["success"]:
            resources = [
                line.strip() for line in result["output"].split("\n") if line.strip()
            ]
            return {
                "success": True,
                "resources": resources,
                "count": len(resources),
            }
        else:
            return {
                "success": False,
                "error": result.get("error"),
            }

    async def _state_show_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Show a resource in Terraform state."""
        address = params["address"]
        result = await self._run_terraform_command(
            ["terraform", "state", "show", address]
        )

        return {
            "success": result["success"],
            "resource": result["output"] if result["success"] else None,
            "error": result.get("error"),
        }

    async def _import_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Import existing infrastructure into Terraform state."""
        address = params["address"]
        resource_id = params["id"]

        result = await self._run_terraform_command(
            ["terraform", "import", address, resource_id]
        )

        return {
            "success": result["success"],
            "message": f"Resource {address} imported successfully"
            if result["success"]
            else None,
            "output": result["output"],
            "error": result.get("error"),
        }

    async def _output_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Show Terraform output values."""
        cmd = ["terraform", "output"]

        if params.get("json", True):
            cmd.append("-json")

        result = await self._run_terraform_command(cmd)

        if result["success"]:
            try:
                if params.get("json", True):
                    outputs = json.loads(result["output"])
                    return {
                        "success": True,
                        "outputs": outputs,
                        "format": "json",
                    }
                else:
                    return {
                        "success": True,
                        "outputs": result["output"],
                        "format": "text",
                    }
            except json.JSONDecodeError:
                return {
                    "success": True,
                    "outputs": result["output"],
                    "format": "text",
                }
        else:
            return {
                "success": False,
                "error": result.get("error"),
            }

    async def _generate_hcl_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate HCL configuration from parameters."""
        try:
            hcl_parts = []

            # Generate provider configurations
            providers = params.get("providers", {})
            for provider_name, provider_config in providers.items():
                hcl_parts.append(
                    self.hcl_generator.generate_provider_config(
                        provider_name, provider_config
                    )
                )

            # Generate resource configurations
            resources = params["resources"]
            for resource in resources:
                resource_type = resource["type"]
                resource_name = resource["name"]
                resource_config = resource["config"]

                hcl_parts.append(
                    self.hcl_generator.generate_resource_config(
                        resource_type, resource_name, resource_config
                    )
                )

            # Generate data source configurations
            data_sources = params.get("data_sources", [])
            for data_source in data_sources:
                data_type = data_source["type"]
                data_name = data_source["name"]
                data_config = data_source["config"]

                hcl_parts.append(
                    self.hcl_generator.generate_data_source_config(
                        data_type, data_name, data_config
                    )
                )

            # Generate output configurations
            outputs = params.get("outputs", {})
            for output_name, output_config in outputs.items():
                hcl_parts.append(
                    self.hcl_generator.generate_output_config(
                        output_name, output_config
                    )
                )

            # Combine all HCL parts
            full_hcl = "\n\n".join(hcl_parts)

            # Write to file if working directory is available
            config_file = os.path.join(self.terraform_config.working_dir, "main.tf")
            with open(config_file, "w") as f:
                f.write(full_hcl)

            return {
                "success": True,
                "hcl_configuration": full_hcl,
                "config_file": config_file,
                "lines": len(full_hcl.split("\n")),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def _workspace_list_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List Terraform workspaces."""
        result = await self._run_terraform_command(["terraform", "workspace", "list"])

        if result["success"]:
            workspaces = []
            current_workspace = None

            for line in result["output"].split("\n"):
                line = line.strip()
                if line:
                    if line.startswith("*"):
                        current_workspace = line[1:].strip()
                        workspaces.append(current_workspace)
                    else:
                        workspaces.append(line)

            return {
                "success": True,
                "workspaces": workspaces,
                "current_workspace": current_workspace,
                "count": len(workspaces),
            }
        else:
            return {
                "success": False,
                "error": result.get("error"),
            }

    async def _workspace_select_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Select a Terraform workspace."""
        workspace = params["workspace"]
        result = await self._run_terraform_command(
            ["terraform", "workspace", "select", workspace]
        )

        return {
            "success": result["success"],
            "workspace": workspace,
            "message": f"Switched to workspace '{workspace}'"
            if result["success"]
            else None,
            "error": result.get("error"),
        }

    async def _workspace_new_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Terraform workspace."""
        workspace = params["workspace"]
        result = await self._run_terraform_command(
            ["terraform", "workspace", "new", workspace]
        )

        return {
            "success": result["success"],
            "workspace": workspace,
            "message": f"Created and switched to workspace '{workspace}'"
            if result["success"]
            else None,
            "error": result.get("error"),
        }

    async def _format_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Format Terraform configuration files."""
        cmd = ["terraform", "fmt"]

        if params.get("check", False):
            cmd.append("-check")

        if params.get("diff", False):
            cmd.append("-diff")

        result = await self._run_terraform_command(cmd)

        return {
            "success": result["success"],
            "message": "Configuration formatted successfully"
            if result["success"]
            else None,
            "output": result["output"],
            "error": result.get("error"),
        }

    async def _get_state_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get current Terraform state."""
        result = await self._run_terraform_command(["terraform", "show", "-json"])

        if result["success"]:
            try:
                state_data = json.loads(result["output"])

                # Extract state information
                terraform_state = TerraformState(
                    version=state_data.get("format_version", "unknown"),
                    terraform_version=state_data.get("terraform_version", "unknown"),
                    resources=state_data.get("values", {})
                    .get("root_module", {})
                    .get("resources", []),
                    outputs=state_data.get("values", {}).get("outputs", {}),
                )

                return {
                    "success": True,
                    "state": {
                        "version": terraform_state.version,
                        "terraform_version": terraform_state.terraform_version,
                        "resource_count": terraform_state.resource_count,
                        "resources": terraform_state.resources,
                        "outputs": terraform_state.outputs,
                    },
                }
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Failed to parse state JSON: {e}",
                }
        else:
            return {
                "success": False,
                "error": result.get("error"),
            }

    async def _run_terraform_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run a Terraform command with proper working directory."""
        return await self._run_command(cmd, cwd=self.terraform_config.working_dir)

    async def _run_command(
        self, cmd: List[str], cwd: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run a command and return the result."""
        try:
            self.logger.debug(f"Running command: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            stdout, stderr = await process.communicate()

            success = process.returncode == 0
            output = stdout.decode("utf-8") if stdout else ""
            error = stderr.decode("utf-8") if stderr else ""

            return {
                "success": success,
                "return_code": process.returncode,
                "output": output,
                "error": error,
            }

        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": "",
            }

    def _parse_plan_output(self, plan_output: str) -> TerraformPlan:
        """Parse Terraform plan output to extract changes summary."""
        changes_summary = {"add": 0, "change": 0, "destroy": 0}
        resource_changes = []

        # Parse JSON plan output
        try:
            for line in plan_output.split("\n"):
                if line.strip() and line.startswith('{"'):
                    try:
                        plan_data = json.loads(line)
                        if plan_data.get("type") == "resource_drift":
                            # Handle resource changes
                            change = plan_data.get("change", {})
                            actions = change.get("actions", [])

                            if "create" in actions:
                                changes_summary["add"] += 1
                            elif "update" in actions:
                                changes_summary["change"] += 1
                            elif "delete" in actions:
                                changes_summary["destroy"] += 1

                            resource_changes.append(
                                {
                                    "address": change.get("resource", {}).get(
                                        "addr", ""
                                    ),
                                    "actions": actions,
                                    "before": change.get("before"),
                                    "after": change.get("after"),
                                }
                            )
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            self.logger.warning(f"Failed to parse plan output: {e}")

        # Generate a temporary plan file name
        plan_file = os.path.join(
            self.terraform_config.working_dir, f"tfplan-{int(time.time())}"
        )

        return TerraformPlan(
            plan_file=plan_file,
            changes_summary=changes_summary,
            resource_changes=resource_changes,
            plan_output=plan_output,
        )

    async def estimate_cost(self, action: str, params: Dict[str, Any]) -> CostEstimate:
        """Estimate cost of Terraform operations."""
        # Terraform operations are generally free, but infrastructure costs vary
        base_cost = 0.0

        if action in ["plan", "validate", "show", "format"]:
            base_cost = 0.0  # Read-only operations
        elif action in ["init", "workspace_list", "workspace_select"]:
            base_cost = 0.001  # Light setup operations
        elif action in ["apply", "destroy", "import"]:
            # These operations affect actual infrastructure
            # Cost depends on the resources being managed
            resource_count = len(params.get("target", []))
            if resource_count == 0:
                # Estimate based on typical small deployment
                base_cost = 0.10
            else:
                base_cost = 0.02 * resource_count
        elif action == "generate_hcl":
            # Cost based on complexity of generated configuration
            resource_count = len(params.get("resources", []))
            base_cost = 0.001 * resource_count

        return CostEstimate(
            estimated_cost=base_cost,
            currency="USD",
            confidence=0.5,  # Low confidence since actual infrastructure costs vary widely
        )

    async def _get_supported_actions(self) -> List[str]:
        """Get list of supported Terraform actions."""
        return [
            "init",
            "plan",
            "apply",
            "destroy",
            "validate",
            "show",
            "state_list",
            "state_show",
            "import",
            "output",
            "generate_hcl",
            "workspace_list",
            "workspace_select",
            "workspace_new",
            "format",
            "get_state",
        ]

    async def _execute_rollback(self, execution_id: str) -> Dict[str, Any]:
        """Execute rollback operation for Terraform."""
        # Terraform rollback typically involves applying a previous state
        # This is a complex operation that requires careful state management
        return {
            "message": f"Terraform rollback requires manual state management for execution {execution_id}",
            "recommendation": "Use 'terraform plan -destroy' and 'terraform apply' to rollback changes",
        }

    async def _create_client(self) -> None:
        """Create Terraform client (not applicable for command-line tool)."""
        return None

    async def _create_validator(self) -> None:
        """Create parameter validator (not needed for Terraform)."""
        return None
