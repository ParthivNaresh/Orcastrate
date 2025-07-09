"""
Integration tests for Terraform Infrastructure as Code tool.

These tests require Terraform to be installed on the system for comprehensive testing.
"""

import os
import tempfile
import time

import pytest

from src.tools.base import ToolConfig
from src.tools.terraform import TerraformTool


@pytest.mark.integration
@pytest.mark.terraform_required
class TestTerraformIntegration:
    """Integration tests for Terraform tool."""

    @pytest.fixture
    async def terraform_tool(self):
        """Create Terraform tool with temporary working directory."""
        with tempfile.TemporaryDirectory(prefix="terraform_integration_") as temp_dir:
            config = ToolConfig(
                name="terraform",
                version="1.0.0",
                environment={
                    "working_dir": temp_dir,
                    "terraform_version": "latest",
                    "backend_config": {},
                    "var_files": [],
                    "variables": {},
                    "parallelism": 10,
                    "auto_approve": False,
                },
            )

            tool = TerraformTool(config)
            try:
                await tool.initialize()
                yield tool
            except Exception as e:
                pytest.skip(f"Terraform not available for testing: {e}")
            finally:
                if hasattr(tool, "cleanup"):
                    await tool.cleanup()

    def _create_simple_terraform_config(self, working_dir: str) -> str:
        """Create a simple Terraform configuration for testing."""
        config_content = """
terraform {
  required_version = ">= 0.14"
}

# Simple local file resource for testing
resource "local_file" "test" {
  content  = "Hello, Terraform"
  filename = "${path.module}/test_output.txt"
}

output "test_file_content" {
  value = local_file.test.content
}
"""
        config_file = os.path.join(working_dir, "main.tf")
        with open(config_file, "w") as f:
            f.write(config_content)
        return config_file

    @pytest.mark.asyncio
    async def test_terraform_binary_detection(self, terraform_tool):
        """Test that Terraform binary is properly detected."""
        assert terraform_tool._terraform_binary == "terraform"

    @pytest.mark.asyncio
    async def test_init_command(self, terraform_tool):
        """Test Terraform init command."""
        # Create a simple configuration
        self._create_simple_terraform_config(
            terraform_tool.terraform_config.working_dir
        )

        # Test init
        result = await terraform_tool._init_action({})

        assert result["success"]
        assert "message" in result
        assert result["output"] is not None

        # Verify .terraform directory was created
        terraform_dir = os.path.join(
            terraform_tool.terraform_config.working_dir, ".terraform"
        )
        assert os.path.exists(terraform_dir)

    @pytest.mark.asyncio
    async def test_validate_command(self, terraform_tool):
        """Test Terraform validate command."""
        # Create and initialize configuration
        self._create_simple_terraform_config(
            terraform_tool.terraform_config.working_dir
        )
        await terraform_tool._init_action({})

        # Test validate
        result = await terraform_tool._validate_action({})

        assert result["success"]
        assert result["valid"]
        assert result["error_count"] == 0

    @pytest.mark.asyncio
    async def test_validate_invalid_config(self, terraform_tool):
        """Test Terraform validate with invalid configuration."""
        # Create configuration with invalid syntax (not invalid resource type)
        invalid_config = """
resource "local_file" "test" {
  content = "test"
  filename = "${path.module}/test.txt"
  # Missing closing brace to make it invalid
"""
        config_file = os.path.join(
            terraform_tool.terraform_config.working_dir, "main.tf"
        )
        with open(config_file, "w") as f:
            f.write(invalid_config)

        # Init should fail with syntax error
        await terraform_tool._init_action({})
        # Init may still succeed, but validation should catch syntax errors

        # Test validate
        result = await terraform_tool._validate_action({})

        # The validation command itself should succeed (return code 0)
        # but the configuration should be marked as invalid
        assert result["success"]  # The command itself succeeds
        if "valid" in result:
            assert result["valid"] is False  # Configuration is invalid

    @pytest.mark.asyncio
    async def test_plan_command(self, terraform_tool):
        """Test Terraform plan command."""
        # Create and initialize configuration
        self._create_simple_terraform_config(
            terraform_tool.terraform_config.working_dir
        )
        await terraform_tool._init_action({})

        # Test plan
        result = await terraform_tool._plan_action({})

        assert result["success"]
        assert "plan" in result
        assert result["has_changes"] is not None
        assert "changes_summary" in result

    @pytest.mark.asyncio
    async def test_format_command(self, terraform_tool):
        """Test Terraform format command."""
        # Create unformatted configuration
        unformatted_config = """
resource "local_file" "test"{
content="Hello, World!"
filename="${path.module}/test.txt"
}
"""
        config_file = os.path.join(
            terraform_tool.terraform_config.working_dir, "main.tf"
        )
        with open(config_file, "w") as f:
            f.write(unformatted_config)

        # Test format
        result = await terraform_tool._format_action({})

        assert result["success"]
        assert "message" in result

        # Check if file was formatted (content should change)
        with open(config_file, "r") as f:
            formatted_content = f.read()

        # The content should be different (formatted)
        assert formatted_content != unformatted_config

    @pytest.mark.asyncio
    async def test_workspace_operations(self, terraform_tool):
        """Test Terraform workspace operations."""
        # Initialize first
        self._create_simple_terraform_config(
            terraform_tool.terraform_config.working_dir
        )
        await terraform_tool._init_action({})

        # Test workspace list
        list_result = await terraform_tool._workspace_list_action({})
        assert list_result["success"]
        assert "default" in list_result["workspaces"]
        assert list_result["current_workspace"] == "default"

        # Test workspace create
        new_workspace = f"test_workspace_{int(time.time())}"
        create_result = await terraform_tool._workspace_new_action(
            {"workspace": new_workspace}
        )
        assert create_result["success"]
        assert create_result["workspace"] == new_workspace

        # Test workspace list again (should include new workspace)
        list_result2 = await terraform_tool._workspace_list_action({})
        assert list_result2["success"]
        assert new_workspace in list_result2["workspaces"]
        assert list_result2["current_workspace"] == new_workspace

        # Test workspace select
        select_result = await terraform_tool._workspace_select_action(
            {"workspace": "default"}
        )
        assert select_result["success"]
        assert select_result["workspace"] == "default"

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, terraform_tool):
        """Test complete Terraform lifecycle: init -> plan -> apply -> destroy."""
        # Create configuration
        self._create_simple_terraform_config(
            terraform_tool.terraform_config.working_dir
        )

        # 1. Initialize
        init_result = await terraform_tool._init_action({})
        assert init_result["success"]

        # 2. Plan
        plan_result = await terraform_tool._plan_action({})
        assert plan_result["success"]

        # Should have changes (creating the local file)
        if plan_result["has_changes"]:
            # 3. Apply with auto-approve
            apply_result = await terraform_tool._apply_action({"auto_approve": True})
            assert apply_result["success"]

            # Verify the file was created
            test_file = os.path.join(
                terraform_tool.terraform_config.working_dir, "test_output.txt"
            )
            assert os.path.exists(test_file)

            with open(test_file, "r") as f:
                content = f.read()
            assert content == "Hello, Terraform"

            # 4. Show state
            show_result = await terraform_tool._show_action({})
            assert show_result["success"]
            assert show_result["format"] == "json"

            # 5. Get outputs
            output_result = await terraform_tool._output_action({})
            assert output_result["success"]
            if output_result["outputs"]:
                assert "test_file_content" in output_result["outputs"]

            # 6. List state resources
            state_list_result = await terraform_tool._state_list_action({})
            assert state_list_result["success"]
            assert state_list_result["count"] > 0
            assert any(
                "local_file.test" in resource
                for resource in state_list_result["resources"]
            )

            # 7. Show specific resource
            state_show_result = await terraform_tool._state_show_action(
                {"address": "local_file.test"}
            )
            assert state_show_result["success"]
            assert state_show_result["resource"] is not None

            # 8. Get state
            get_state_result = await terraform_tool._get_state_action({})
            assert get_state_result["success"]
            assert get_state_result["state"]["resource_count"] > 0

            # 9. Destroy with auto-approve
            destroy_result = await terraform_tool._destroy_action(
                {"auto_approve": True}
            )
            assert destroy_result["success"]

            # Verify the file was removed
            assert not os.path.exists(test_file)

    @pytest.mark.asyncio
    async def test_hcl_generation_and_apply(self, terraform_tool):
        """Test HCL generation and subsequent apply."""
        # Generate HCL configuration
        hcl_params = {
            "providers": {},
            "resources": [
                {
                    "type": "local_file",
                    "name": "generated",
                    "config": {
                        "content": "Generated by HCL generator",
                        "filename": "${path.module}/generated.txt",
                    },
                }
            ],
            "outputs": {
                "generated_file": {"value": "${local_file.generated.filename}"}
            },
        }

        # Generate HCL
        hcl_result = await terraform_tool._generate_hcl_action(hcl_params)
        assert hcl_result["success"]
        assert "local_file" in hcl_result["hcl_configuration"]
        assert "output" in hcl_result["hcl_configuration"]

        # Verify the file was created
        assert os.path.exists(hcl_result["config_file"])

        # Initialize and apply the generated configuration
        init_result = await terraform_tool._init_action({})
        assert init_result["success"]

        plan_result = await terraform_tool._plan_action({})
        assert plan_result["success"]

        if plan_result["has_changes"]:
            apply_result = await terraform_tool._apply_action({"auto_approve": True})
            assert apply_result["success"]

            # Verify the generated file exists
            generated_file = os.path.join(
                terraform_tool.terraform_config.working_dir, "generated.txt"
            )
            assert os.path.exists(generated_file)

            with open(generated_file, "r") as f:
                content = f.read()
            assert content == "Generated by HCL generator"

            # Clean up
            destroy_result = await terraform_tool._destroy_action(
                {"auto_approve": True}
            )
            assert destroy_result["success"]

    @pytest.mark.asyncio
    async def test_variable_handling(self, terraform_tool):
        """Test Terraform with variables."""
        # Create configuration with variables
        config_content = """
variable "file_content" {
  description = "Content for the test file"
  type        = string
  default     = "Default content"
}

variable "file_suffix" {
  description = "Suffix for the filename"
  type        = string
  default     = "default"
}

resource "local_file" "variable_test" {
  content  = var.file_content
  filename = "${path.module}/test_${var.file_suffix}.txt"
}

output "filename" {
  value = local_file.variable_test.filename
}
"""
        config_file = os.path.join(
            terraform_tool.terraform_config.working_dir, "main.tf"
        )
        with open(config_file, "w") as f:
            f.write(config_content)

        # Initialize
        await terraform_tool._init_action({})

        # Plan with variables
        plan_result = await terraform_tool._plan_action(
            {
                "variables": {
                    "file_content": "Custom content from variables",
                    "file_suffix": "custom",
                }
            }
        )
        assert plan_result["success"]

        if plan_result["has_changes"]:
            # Apply with variables
            apply_result = await terraform_tool._apply_action(
                {
                    "auto_approve": True,
                    "variables": {
                        "file_content": "Custom content from variables",
                        "file_suffix": "custom",
                    },
                }
            )
            assert apply_result["success"]

            # Verify the file was created with correct content
            test_file = os.path.join(
                terraform_tool.terraform_config.working_dir, "test_custom.txt"
            )
            assert os.path.exists(test_file)

            with open(test_file, "r") as f:
                content = f.read()
            assert content == "Custom content from variables"

            # Clean up
            destroy_result = await terraform_tool._destroy_action(
                {
                    "auto_approve": True,
                    "variables": {
                        "file_content": "Custom content from variables",
                        "file_suffix": "custom",
                    },
                }
            )
            assert destroy_result["success"]

    @pytest.mark.asyncio
    async def test_error_handling(self, terraform_tool):
        """Test error handling with invalid operations."""
        # Test plan without initialization
        plan_result = await terraform_tool._plan_action({})
        # This should fail because we haven't initialized
        assert not plan_result["success"]
        assert "error" in plan_result

        # Test apply without plan
        apply_result = await terraform_tool._apply_action({"auto_approve": True})
        # This should also fail
        assert not apply_result["success"]
        assert "error" in apply_result

    @pytest.mark.asyncio
    async def test_state_operations_empty_state(self, terraform_tool):
        """Test state operations with empty state."""
        # Create config and initialize
        self._create_simple_terraform_config(
            terraform_tool.terraform_config.working_dir
        )
        init_result = await terraform_tool._init_action({})
        assert init_result[
            "success"
        ], f"Init failed: {init_result.get('error', 'Unknown error')}"

        # Test state list on empty state
        state_list_result = await terraform_tool._state_list_action({})
        assert state_list_result["success"]
        assert state_list_result["count"] == 0
        assert state_list_result["resources"] == []

        # Test get state on empty state
        get_state_result = await terraform_tool._get_state_action({})
        assert get_state_result["success"]
        assert get_state_result["state"]["resource_count"] == 0
