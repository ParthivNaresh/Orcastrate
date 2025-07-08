"""
Unit tests for Terraform Infrastructure as Code tool.

Tests the Terraform tool functionality with mocked command execution.
"""

import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from src.tools.base import ToolConfig, ToolError
from src.tools.terraform import (
    HCLGenerator,
    TerraformConfig,
    TerraformPlan,
    TerraformState,
    TerraformTool,
)


class TestTerraformConfig:
    """Test Terraform configuration functionality."""

    def test_config_initialization_default(self):
        """Test Terraform config initialization with defaults."""
        config = TerraformConfig()

        assert "terraform_" in config.working_dir
        assert config.terraform_version == "latest"
        assert config.backend_config == {}
        assert config.var_files == []
        assert config.variables == {}
        assert config.parallelism == 10
        assert config.auto_approve is False

    def test_config_initialization_custom(self):
        """Test Terraform config initialization with custom values."""
        config = TerraformConfig(
            working_dir="/custom/path",
            terraform_version="1.5.0",
            backend_config={"bucket": "my-bucket"},
            var_files=["vars.tfvars"],
            variables={"env": "prod"},
            parallelism=5,
            auto_approve=True,
        )

        assert config.working_dir == "/custom/path"
        assert config.terraform_version == "1.5.0"
        assert config.backend_config == {"bucket": "my-bucket"}
        assert config.var_files == ["vars.tfvars"]
        assert config.variables == {"env": "prod"}
        assert config.parallelism == 5
        assert config.auto_approve is True


class TestTerraformPlan:
    """Test Terraform plan functionality."""

    def test_plan_initialization(self):
        """Test Terraform plan initialization."""
        plan = TerraformPlan(
            plan_file="/path/to/plan",
            changes_summary={"add": 3, "change": 1, "destroy": 0},
            resource_changes=[{"address": "aws_instance.test", "actions": ["create"]}],
            plan_output="Plan output here",
        )

        assert plan.plan_file == "/path/to/plan"
        assert plan.changes_summary == {"add": 3, "change": 1, "destroy": 0}
        assert len(plan.resource_changes) == 1
        assert plan.plan_output == "Plan output here"
        assert plan.created_at > 0

    def test_plan_has_changes(self):
        """Test plan has_changes property."""
        # Plan with changes
        plan_with_changes = TerraformPlan(
            plan_file="/path/to/plan",
            changes_summary={"add": 1, "change": 0, "destroy": 0},
            resource_changes=[],
            plan_output="",
        )
        assert plan_with_changes.has_changes

        # Plan without changes
        plan_no_changes = TerraformPlan(
            plan_file="/path/to/plan",
            changes_summary={"add": 0, "change": 0, "destroy": 0},
            resource_changes=[],
            plan_output="",
        )
        assert not plan_no_changes.has_changes

    def test_plan_total_changes(self):
        """Test plan total_changes property."""
        plan = TerraformPlan(
            plan_file="/path/to/plan",
            changes_summary={"add": 3, "change": 2, "destroy": 1},
            resource_changes=[],
            plan_output="",
        )
        assert plan.total_changes == 6


class TestTerraformState:
    """Test Terraform state functionality."""

    def test_state_initialization(self):
        """Test Terraform state initialization."""
        resources = [
            {"type": "aws_instance", "name": "web"},
            {"type": "aws_s3_bucket", "name": "data"},
        ]
        outputs = {"instance_ip": "1.2.3.4"}

        state = TerraformState(
            version="4",
            terraform_version="1.5.0",
            resources=resources,
            outputs=outputs,
        )

        assert state.version == "4"
        assert state.terraform_version == "1.5.0"
        assert state.resources == resources
        assert state.outputs == outputs

    def test_state_resource_count(self):
        """Test state resource_count property."""
        resources = [
            {"type": "aws_instance", "name": "web"},
            {"type": "aws_s3_bucket", "name": "data"},
        ]

        state = TerraformState(
            version="4",
            terraform_version="1.5.0",
            resources=resources,
            outputs={},
        )

        assert state.resource_count == 2

    def test_state_get_resources_by_type(self):
        """Test getting resources by type."""
        resources = [
            {"type": "aws_instance", "name": "web1"},
            {"type": "aws_instance", "name": "web2"},
            {"type": "aws_s3_bucket", "name": "data"},
        ]

        state = TerraformState(
            version="4",
            terraform_version="1.5.0",
            resources=resources,
            outputs={},
        )

        instances = state.get_resources_by_type("aws_instance")
        assert len(instances) == 2
        assert all(r["type"] == "aws_instance" for r in instances)

        buckets = state.get_resources_by_type("aws_s3_bucket")
        assert len(buckets) == 1
        assert buckets[0]["type"] == "aws_s3_bucket"


class TestHCLGenerator:
    """Test HCL generation functionality."""

    def test_generate_provider_config(self):
        """Test provider configuration generation."""
        config = {
            "region": "us-west-2",
            "profile": "default",
            "assume_role": {"role_arn": "arn:aws:iam::123456789012:role/TerraformRole"},
        }

        hcl = HCLGenerator.generate_provider_config("aws", config)

        assert 'provider "aws" {' in hcl
        assert 'region = "us-west-2"' in hcl
        assert 'profile = "default"' in hcl
        assert "assume_role = {" in hcl
        assert "}" in hcl

    def test_generate_resource_config(self):
        """Test resource configuration generation."""
        config = {
            "ami": "ami-12345678",
            "instance_type": "t3.micro",
            "tags": {"Name": "test-instance"},
        }

        hcl = HCLGenerator.generate_resource_config("aws_instance", "web", config)

        assert 'resource "aws_instance" "web" {' in hcl
        assert 'ami = "ami-12345678"' in hcl
        assert 'instance_type = "t3.micro"' in hcl
        assert "tags = {" in hcl
        assert "}" in hcl

    def test_generate_data_source_config(self):
        """Test data source configuration generation."""
        config = {"most_recent": True, "owners": ["amazon"]}

        hcl = HCLGenerator.generate_data_source_config("aws_ami", "ubuntu", config)

        assert 'data "aws_ami" "ubuntu" {' in hcl
        assert "most_recent = true" in hcl
        assert 'owners = ["amazon"]' in hcl
        assert "}" in hcl

    def test_generate_output_config(self):
        """Test output configuration generation."""
        config = {
            "value": "${aws_instance.web.public_ip}",
            "description": "Web server IP",
        }

        hcl = HCLGenerator.generate_output_config("web_ip", config)

        assert 'output "web_ip" {' in hcl
        assert 'value = "${aws_instance.web.public_ip}"' in hcl
        assert 'description = "Web server IP"' in hcl
        assert "}" in hcl

    def test_format_hcl_value_primitives(self):
        """Test HCL value formatting for primitive types."""
        assert HCLGenerator._format_hcl_value("string") == '"string"'
        assert HCLGenerator._format_hcl_value(True) == "true"
        assert HCLGenerator._format_hcl_value(False) == "false"
        assert HCLGenerator._format_hcl_value(42) == "42"
        assert HCLGenerator._format_hcl_value(3.14) == "3.14"

    def test_format_hcl_value_collections(self):
        """Test HCL value formatting for collections."""
        # Test list
        list_result = HCLGenerator._format_hcl_value(["a", "b", "c"])
        assert list_result == '["a", "b", "c"]'

        # Test dict
        dict_result = HCLGenerator._format_hcl_value({"key": "value", "num": 42})
        assert 'key = "value"' in dict_result
        assert "num = 42" in dict_result
        assert dict_result.startswith("{")
        assert dict_result.endswith("  }")


class TestTerraformTool:
    """Test Terraform tool functionality."""

    @pytest.fixture
    def tool_config(self):
        """Create test tool configuration."""
        return ToolConfig(
            name="terraform",
            version="1.0.0",
            environment={
                "working_dir": "/tmp/terraform_test",
                "terraform_version": "1.5.0",
                "backend_config": {"bucket": "test-bucket"},
                "var_files": ["test.tfvars"],
                "variables": {"env": "test"},
                "parallelism": 5,
                "auto_approve": False,
            },
        )

    @pytest.fixture
    def terraform_tool(self, tool_config):
        """Create Terraform tool instance."""
        return TerraformTool(tool_config)

    def test_tool_initialization(self, terraform_tool, tool_config):
        """Test Terraform tool initialization."""
        assert terraform_tool.config == tool_config
        assert terraform_tool.terraform_config.working_dir == "/tmp/terraform_test"
        assert terraform_tool.terraform_config.terraform_version == "1.5.0"
        assert terraform_tool.terraform_config.backend_config == {
            "bucket": "test-bucket"
        }
        assert terraform_tool.hcl_generator is not None
        assert terraform_tool._terraform_binary is None

    def test_create_terraform_config(self, terraform_tool):
        """Test Terraform configuration creation."""
        config = terraform_tool._create_terraform_config()

        assert config.working_dir == "/tmp/terraform_test"
        assert config.terraform_version == "1.5.0"
        assert config.backend_config == {"bucket": "test-bucket"}
        assert config.var_files == ["test.tfvars"]
        assert config.variables == {"env": "test"}
        assert config.parallelism == 5
        assert config.auto_approve is False

    @pytest.mark.asyncio
    @patch("os.makedirs")
    async def test_initialize_success(self, mock_makedirs, terraform_tool):
        """Test successful tool initialization."""
        with patch.object(terraform_tool, "_ensure_terraform_binary") as mock_ensure:
            mock_ensure.return_value = None

            await terraform_tool.initialize()

            mock_makedirs.assert_called_once_with(
                terraform_tool.terraform_config.working_dir, exist_ok=True
            )
            mock_ensure.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_terraform_binary_success(self, terraform_tool):
        """Test successful Terraform binary detection."""
        mock_result = {"success": True, "output": "Terraform v1.5.0"}

        with patch.object(terraform_tool, "_run_command", return_value=mock_result):
            await terraform_tool._ensure_terraform_binary()

            assert terraform_tool._terraform_binary == "terraform"

    @pytest.mark.asyncio
    async def test_ensure_terraform_binary_failure(self, terraform_tool):
        """Test Terraform binary detection failure."""
        mock_result = {"success": False, "output": "command not found"}

        with patch.object(terraform_tool, "_run_command", return_value=mock_result):
            with pytest.raises(ToolError, match="Terraform binary not found"):
                await terraform_tool._ensure_terraform_binary()

    @pytest.mark.asyncio
    async def test_get_schema(self, terraform_tool):
        """Test tool schema retrieval."""
        schema = await terraform_tool.get_schema()

        assert schema.name == "terraform"
        assert "Infrastructure as Code" in schema.description

        expected_actions = [
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

        for action in expected_actions:
            assert action in schema.actions

    @pytest.mark.asyncio
    async def test_init_action(self, terraform_tool):
        """Test init action execution."""
        mock_result = {
            "success": True,
            "output": "Terraform has been successfully initialized!",
        }

        with patch.object(
            terraform_tool, "_run_terraform_command", return_value=mock_result
        ):
            result = await terraform_tool._init_action(
                {"backend_config": {"bucket": "test"}, "upgrade": True}
            )

            assert result["success"]
            assert "initialized successfully" in result["message"]
            assert result["output"] == mock_result["output"]

    @pytest.mark.asyncio
    async def test_plan_action_success(self, terraform_tool):
        """Test plan action execution with successful plan."""
        mock_result = {
            "success": True,
            "output": '{"type": "resource_drift", "change": {"actions": ["create"], "resource": {"addr": "aws_instance.test"}}}',
        }

        with patch.object(
            terraform_tool, "_run_terraform_command", return_value=mock_result
        ):
            with patch.object(terraform_tool, "_parse_plan_output") as mock_parse:
                mock_plan = TerraformPlan(
                    plan_file="/tmp/plan",
                    changes_summary={"add": 1, "change": 0, "destroy": 0},
                    resource_changes=[],
                    plan_output=mock_result["output"],
                )
                mock_parse.return_value = mock_plan

                result = await terraform_tool._plan_action(
                    {
                        "variables": {"env": "test"},
                        "var_file": "test.tfvars",
                        "target": ["aws_instance.test"],
                        "destroy": False,
                        "out": "plan.out",
                    }
                )

                assert result["success"]
                assert result["has_changes"]
                assert result["changes_summary"] == {
                    "add": 1,
                    "change": 0,
                    "destroy": 0,
                }

    @pytest.mark.asyncio
    async def test_apply_action_with_plan_file(self, terraform_tool):
        """Test apply action with plan file."""
        mock_result = {"success": True, "output": "Apply complete!"}

        with patch.object(
            terraform_tool, "_run_terraform_command", return_value=mock_result
        ):
            result = await terraform_tool._apply_action(
                {"plan_file": "plan.out", "auto_approve": True}
            )

            assert result["success"]
            assert "completed successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_validate_action_success(self, terraform_tool):
        """Test validate action with valid configuration."""
        mock_result = {
            "success": True,
            "output": '{"valid": true, "error_count": 0, "warning_count": 0, "diagnostics": []}',
        }

        with patch.object(
            terraform_tool, "_run_terraform_command", return_value=mock_result
        ):
            result = await terraform_tool._validate_action({})

            assert result["success"]
            assert result["valid"]
            assert result["error_count"] == 0
            assert result["warning_count"] == 0

    @pytest.mark.asyncio
    async def test_validate_action_invalid(self, terraform_tool):
        """Test validate action with invalid configuration."""
        mock_result = {
            "success": True,
            "output": '{"valid": false, "error_count": 2, "warning_count": 1, "diagnostics": [{"summary": "Error message"}]}',
        }

        with patch.object(
            terraform_tool, "_run_terraform_command", return_value=mock_result
        ):
            result = await terraform_tool._validate_action({})

            assert result["success"]
            assert not result["valid"]
            assert result["error_count"] == 2
            assert result["warning_count"] == 1
            assert len(result["diagnostics"]) == 1

    @pytest.mark.asyncio
    async def test_show_action_json(self, terraform_tool):
        """Test show action with JSON output."""
        mock_result = {
            "success": True,
            "output": '{"format_version": "1.0", "terraform_version": "1.5.0"}',
        }

        with patch.object(
            terraform_tool, "_run_terraform_command", return_value=mock_result
        ):
            result = await terraform_tool._show_action({"json": True})

            assert result["success"]
            assert result["format"] == "json"
            assert result["data"]["format_version"] == "1.0"

    @pytest.mark.asyncio
    async def test_generate_hcl_action(self, terraform_tool):
        """Test HCL generation action."""
        with tempfile.TemporaryDirectory() as temp_dir:
            terraform_tool.terraform_config.working_dir = temp_dir

            params = {
                "providers": {"aws": {"region": "us-west-2"}},
                "resources": [
                    {
                        "type": "aws_instance",
                        "name": "web",
                        "config": {"ami": "ami-12345", "instance_type": "t3.micro"},
                    }
                ],
                "outputs": {"instance_ip": {"value": "${aws_instance.web.public_ip}"}},
            }

            result = await terraform_tool._generate_hcl_action(params)

            assert result["success"]
            assert "aws_instance" in result["hcl_configuration"]
            assert "provider" in result["hcl_configuration"]
            assert "output" in result["hcl_configuration"]
            assert result["config_file"].endswith("main.tf")
            assert os.path.exists(result["config_file"])

    @pytest.mark.asyncio
    async def test_workspace_operations(self, terraform_tool):
        """Test workspace management operations."""
        # Test workspace list
        mock_list_result = {
            "success": True,
            "output": "  default\n* development\n  production\n",
        }

        with patch.object(
            terraform_tool, "_run_terraform_command", return_value=mock_list_result
        ):
            result = await terraform_tool._workspace_list_action({})

            assert result["success"]
            assert result["current_workspace"] == "development"
            assert "default" in result["workspaces"]
            assert "production" in result["workspaces"]
            assert result["count"] == 3

        # Test workspace select
        mock_select_result = {
            "success": True,
            "output": "Switched to workspace 'production'",
        }

        with patch.object(
            terraform_tool, "_run_terraform_command", return_value=mock_select_result
        ):
            result = await terraform_tool._workspace_select_action(
                {"workspace": "production"}
            )

            assert result["success"]
            assert result["workspace"] == "production"
            assert "Switched" in result["message"]

    @pytest.mark.asyncio
    async def test_parse_plan_output(self, terraform_tool):
        """Test plan output parsing."""
        plan_output = """{"type": "resource_drift", "change": {"actions": ["create"], "resource": {"addr": "aws_instance.test1"}}}
{"type": "resource_drift", "change": {"actions": ["update"], "resource": {"addr": "aws_instance.test2"}}}"""

        plan = terraform_tool._parse_plan_output(plan_output)

        assert plan.changes_summary["add"] == 1
        assert plan.changes_summary["change"] == 1
        assert plan.changes_summary["destroy"] == 0
        assert len(plan.resource_changes) == 2
        assert plan.has_changes
        assert plan.total_changes == 2

    @pytest.mark.asyncio
    async def test_estimate_cost(self, terraform_tool):
        """Test cost estimation for different operations."""
        # Read-only operations
        plan_cost = await terraform_tool.estimate_cost("plan", {})
        assert plan_cost.estimated_cost == 0.0

        validate_cost = await terraform_tool.estimate_cost("validate", {})
        assert validate_cost.estimated_cost == 0.0

        # Light setup operations
        init_cost = await terraform_tool.estimate_cost("init", {})
        assert init_cost.estimated_cost == 0.001

        # Infrastructure operations
        apply_cost = await terraform_tool.estimate_cost("apply", {})
        assert apply_cost.estimated_cost == 0.10

        # Operations with targets
        targeted_apply_cost = await terraform_tool.estimate_cost(
            "apply", {"target": ["aws_instance.web1", "aws_instance.web2"]}
        )
        assert targeted_apply_cost.estimated_cost == 0.04  # 0.02 * 2

        # HCL generation
        hcl_cost = await terraform_tool.estimate_cost(
            "generate_hcl",
            {"resources": [{"type": "aws_instance"}, {"type": "aws_s3_bucket"}]},
        )
        assert hcl_cost.estimated_cost == 0.002  # 0.001 * 2

    @pytest.mark.asyncio
    async def test_get_supported_actions(self, terraform_tool):
        """Test supported actions list."""
        actions = await terraform_tool._get_supported_actions()

        expected_actions = [
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

        for action in expected_actions:
            assert action in actions

    @pytest.mark.asyncio
    async def test_execute_rollback(self, terraform_tool):
        """Test rollback execution (placeholder)."""
        result = await terraform_tool._execute_rollback("test_execution_id")

        assert "manual state management" in result["message"]
        assert "terraform plan -destroy" in result["recommendation"]

    @pytest.mark.asyncio
    async def test_run_command_success(self, terraform_tool):
        """Test successful command execution."""
        with patch("asyncio.create_subprocess_exec") as mock_create:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"Success output", b"")
            mock_process.returncode = 0
            mock_create.return_value = mock_process

            result = await terraform_tool._run_command(["terraform", "version"])

            assert result["success"]
            assert result["return_code"] == 0
            assert result["output"] == "Success output"
            assert result["error"] == ""

    @pytest.mark.asyncio
    async def test_run_command_failure(self, terraform_tool):
        """Test failed command execution."""
        with patch("asyncio.create_subprocess_exec") as mock_create:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"Error output")
            mock_process.returncode = 1
            mock_create.return_value = mock_process

            result = await terraform_tool._run_command(["terraform", "invalid"])

            assert not result["success"]
            assert result["return_code"] == 1
            assert result["output"] == ""
            assert result["error"] == "Error output"

    @pytest.mark.asyncio
    async def test_run_command_exception(self, terraform_tool):
        """Test command execution with exception."""
        with patch("asyncio.create_subprocess_exec") as mock_create:
            mock_create.side_effect = Exception("Process creation failed")

            result = await terraform_tool._run_command(["terraform", "version"])

            assert not result["success"]
            assert "Process creation failed" in result["error"]
            assert result["output"] == ""
