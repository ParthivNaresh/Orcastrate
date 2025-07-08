"""
Tests for File System tool implementation.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.tools.base import ToolConfig, ToolError
from src.tools.filesystem import FileSystemTool


class TestFileSystemTool:
    """Test File System tool functionality."""

    @pytest.fixture
    def fs_config(self):
        """Create File System tool configuration."""
        return ToolConfig(
            name="filesystem",
            version="1.0.0",
            timeout=300,
            retry_count=3,
        )

    @pytest.fixture
    def fs_tool(self, fs_config):
        """Create File System tool instance."""
        return FileSystemTool(fs_config)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.asyncio
    async def test_tool_initialization(self, fs_tool):
        """Test File System tool initialization."""
        await fs_tool.initialize()

        # Should have created base path
        assert fs_tool._base_path is not None
        assert fs_tool._base_path.exists()

    @pytest.mark.asyncio
    async def test_get_schema(self, fs_tool):
        """Test getting tool schema."""
        schema = await fs_tool.get_schema()

        assert schema.name == "filesystem"
        assert (
            schema.description == "File system operations tool with security validation"
        )
        assert "create_directory" in schema.actions
        assert "write_file" in schema.actions
        assert "read_file" in schema.actions
        assert "filesystem" in schema.required_permissions

    @pytest.mark.asyncio
    async def test_estimate_cost_read_file(self, fs_tool):
        """Test cost estimation for reading files."""
        cost = await fs_tool.estimate_cost("read_file", {"max_size": 1048576})  # 1MB

        assert cost.estimated_cost > 0
        assert "read_operation" in cost.cost_breakdown
        assert cost.confidence == 0.9

    @pytest.mark.asyncio
    async def test_estimate_cost_write_file(self, fs_tool):
        """Test cost estimation for writing files."""
        cost = await fs_tool.estimate_cost("write_file", {"content": "test content"})

        assert cost.estimated_cost >= 0
        assert "write_operation" in cost.cost_breakdown

    @pytest.mark.asyncio
    async def test_validate_create_directory_params(self, fs_tool):
        """Test parameter validation for create_directory action."""
        validator = await fs_tool._create_validator()

        # Valid parameters
        result = validator.validate(
            "create_directory",
            {
                "path": "/tmp/orcastrate/test",
                "mode": "755",
            },
        )
        assert result.valid is True

        # Missing required parameters
        result = validator.validate("create_directory", {})
        assert result.valid is False
        assert "path is required for create_directory" in result.errors

    @pytest.mark.asyncio
    async def test_validate_write_file_params(self, fs_tool):
        """Test parameter validation for write_file action."""
        validator = await fs_tool._create_validator()

        # Valid parameters
        result = validator.validate(
            "write_file",
            {
                "path": "/tmp/orcastrate/test.txt",
                "content": "Hello World",
            },
        )
        assert result.valid is True

        # Missing content
        result = validator.validate(
            "write_file",
            {
                "path": "/tmp/orcastrate/test.txt",
            },
        )
        assert result.valid is False
        assert "content is required for write_file" in result.errors

        # Invalid JSON content
        result = validator.validate(
            "write_file",
            {
                "path": "/tmp/orcastrate/test.txt",
                "content": {"invalid": float("inf")},  # Not JSON serializable
            },
        )
        assert result.valid is False
        assert "JSON serializable" in str(result.errors)

    @pytest.mark.asyncio
    async def test_validate_path_security(self, fs_tool):
        """Test path validation for security."""
        validator = await fs_tool._create_validator()

        # Path traversal attempt
        result = validator.validate(
            "create_directory",
            {
                "path": "/tmp/orcastrate/../../../etc",
            },
        )
        assert result.valid is False
        assert "Path traversal not allowed" in result.errors

        # Dangerous absolute path
        result = validator.validate(
            "create_directory",
            {
                "path": "/etc/passwd",
            },
        )
        assert result.valid is False
        assert "not allowed" in str(result.errors)

    @pytest.mark.asyncio
    async def test_create_directory_success(self, fs_tool, temp_dir):
        """Test successful directory creation."""
        await fs_tool.initialize()

        test_path = temp_dir / "test_dir"

        with patch.object(fs_tool, "_base_path", temp_dir):
            result = await fs_tool._create_directory(
                {
                    "path": str(test_path),
                    "mode": "755",
                    "parents": True,
                }
            )

        assert result["success"] is True
        assert result["created"] is True
        assert test_path.exists()
        assert test_path.is_dir()

    @pytest.mark.asyncio
    async def test_write_file_success(self, fs_tool, temp_dir):
        """Test successful file writing."""
        await fs_tool.initialize()

        test_file = temp_dir / "test.txt"
        content = "Hello, World!"

        with patch.object(fs_tool, "_base_path", temp_dir):
            result = await fs_tool._write_file(
                {
                    "path": str(test_file),
                    "content": content,
                    "mode": "644",
                    "create_parents": True,
                }
            )

        assert result["success"] is True
        assert test_file.exists()
        assert test_file.read_text() == content

    @pytest.mark.asyncio
    async def test_write_file_json_content(self, fs_tool, temp_dir):
        """Test writing JSON content to file."""
        await fs_tool.initialize()

        test_file = temp_dir / "test.json"
        content = {"name": "test", "value": 123, "nested": {"key": "value"}}

        with patch.object(fs_tool, "_base_path", temp_dir):
            result = await fs_tool._write_file(
                {
                    "path": str(test_file),
                    "content": content,
                }
            )

        assert result["success"] is True
        assert test_file.exists()

        # Verify JSON content
        written_data = json.loads(test_file.read_text())
        assert written_data == content

    @pytest.mark.asyncio
    async def test_read_file_success(self, fs_tool, temp_dir):
        """Test successful file reading."""
        await fs_tool.initialize()

        test_file = temp_dir / "test.txt"
        content = "Hello, World!"
        test_file.write_text(content)

        with patch.object(fs_tool, "_base_path", temp_dir):
            result = await fs_tool._read_file(
                {
                    "path": str(test_file),
                    "encoding": "utf-8",
                }
            )

        assert result["success"] is True
        assert result["content"] == content
        assert result["size"] == len(content.encode())

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, fs_tool, temp_dir):
        """Test reading non-existent file."""
        await fs_tool.initialize()

        test_file = temp_dir / "nonexistent.txt"

        with patch.object(fs_tool, "_base_path", temp_dir):
            result = await fs_tool._read_file(
                {
                    "path": str(test_file),
                }
            )

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_read_file_size_limit(self, fs_tool, temp_dir):
        """Test file size limit enforcement."""
        await fs_tool.initialize()

        test_file = temp_dir / "large.txt"
        large_content = "x" * 1000  # Create content larger than limit
        test_file.write_text(large_content)

        with patch.object(fs_tool, "_base_path", temp_dir):
            result = await fs_tool._read_file(
                {
                    "path": str(test_file),
                    "max_size": 100,  # Small limit
                }
            )

        assert result["success"] is False
        assert "too large" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_list_directory_success(self, fs_tool, temp_dir):
        """Test successful directory listing."""
        await fs_tool.initialize()

        # Create test files
        (temp_dir / "file1.txt").write_text("test1")
        (temp_dir / "file2.txt").write_text("test2")
        (temp_dir / "subdir").mkdir()

        with patch.object(fs_tool, "_base_path", temp_dir):
            result = await fs_tool._list_directory(
                {
                    "path": str(temp_dir),
                    "recursive": False,
                    "include_hidden": False,
                }
            )

        assert result["success"] is True
        assert result["count"] == 3  # 2 files + 1 directory

        # Check entries
        entry_names = [entry["name"] for entry in result["entries"]]
        assert "file1.txt" in entry_names
        assert "file2.txt" in entry_names
        assert "subdir" in entry_names

    @pytest.mark.asyncio
    async def test_copy_file_success(self, fs_tool, temp_dir):
        """Test successful file copying."""
        await fs_tool.initialize()

        source_file = temp_dir / "source.txt"
        dest_file = temp_dir / "dest.txt"
        content = "Test content"
        source_file.write_text(content)

        with patch.object(fs_tool, "_base_path", temp_dir):
            result = await fs_tool._copy_file(
                {
                    "source": str(source_file),
                    "destination": str(dest_file),
                    "preserve_permissions": True,
                }
            )

        assert result["success"] is True
        assert dest_file.exists()
        assert dest_file.read_text() == content

    @pytest.mark.asyncio
    async def test_move_file_success(self, fs_tool, temp_dir):
        """Test successful file moving."""
        await fs_tool.initialize()

        source_file = temp_dir / "source.txt"
        dest_file = temp_dir / "dest.txt"
        content = "Test content"
        source_file.write_text(content)

        with patch.object(fs_tool, "_base_path", temp_dir):
            result = await fs_tool._move_file(
                {
                    "source": str(source_file),
                    "destination": str(dest_file),
                }
            )

        assert result["success"] is True
        assert not source_file.exists()
        assert dest_file.exists()
        assert dest_file.read_text() == content

    @pytest.mark.asyncio
    async def test_delete_file_success(self, fs_tool, temp_dir):
        """Test successful file deletion."""
        await fs_tool.initialize()

        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        with patch.object(fs_tool, "_base_path", temp_dir):
            result = await fs_tool._delete_file(
                {
                    "path": str(test_file),
                }
            )

        assert result["success"] is True
        assert result["deleted"] is True
        assert not test_file.exists()

    @pytest.mark.asyncio
    async def test_delete_file_already_gone(self, fs_tool, temp_dir):
        """Test deleting non-existent file."""
        await fs_tool.initialize()

        test_file = temp_dir / "nonexistent.txt"

        with patch.object(fs_tool, "_base_path", temp_dir):
            result = await fs_tool._delete_file(
                {
                    "path": str(test_file),
                }
            )

        assert result["success"] is True
        assert result["deleted"] is False
        assert "already does not exist" in result["message"]

    @pytest.mark.asyncio
    async def test_get_info_success(self, fs_tool, temp_dir):
        """Test getting file information."""
        await fs_tool.initialize()

        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        with patch.object(fs_tool, "_base_path", temp_dir):
            result = await fs_tool._get_info(
                {
                    "path": str(test_file),
                }
            )

        assert result["success"] is True
        info = result["info"]
        assert info["name"] == "test.txt"
        assert info["type"] == "file"
        assert info["size"] > 0
        assert "permissions" in info

    @pytest.mark.asyncio
    async def test_set_permissions_success(self, fs_tool, temp_dir):
        """Test setting file permissions."""
        await fs_tool.initialize()

        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        with patch.object(fs_tool, "_base_path", temp_dir):
            result = await fs_tool._set_permissions(
                {
                    "path": str(test_file),
                    "mode": "644",
                    "recursive": False,
                }
            )

        assert result["success"] is True
        assert result["mode"] == "644"

    @pytest.mark.asyncio
    async def test_execute_action_unknown(self, fs_tool):
        """Test executing unknown action."""
        await fs_tool.initialize()

        with pytest.raises(ToolError, match="Unknown action"):
            await fs_tool._execute_action("unknown_action", {})

    @pytest.mark.asyncio
    async def test_get_supported_actions(self, fs_tool):
        """Test getting supported actions."""
        actions = await fs_tool._get_supported_actions()

        expected_actions = [
            "create_directory",
            "list_directory",
            "remove_directory",
            "write_file",
            "read_file",
            "copy_file",
            "move_file",
            "delete_file",
            "get_info",
            "set_permissions",
        ]

        assert all(action in actions for action in expected_actions)

    @pytest.mark.asyncio
    async def test_full_workflow_integration(self, fs_tool, temp_dir):
        """Test a complete file system workflow."""
        await fs_tool.initialize()

        # Use paths within /tmp/orcastrate for this test
        project_dir = "/tmp/orcastrate/test_project"
        app_file = "/tmp/orcastrate/test_project/app.py"

        # Create directory
        dir_result = await fs_tool.execute("create_directory", {"path": project_dir})
        assert dir_result.success is True

        # Write file
        file_result = await fs_tool.execute(
            "write_file",
            {
                "path": app_file,
                "content": "print('Hello World')",
            },
        )
        assert file_result.success is True

        # Read file back
        read_result = await fs_tool.execute("read_file", {"path": app_file})
        assert read_result.success is True
        assert "Hello World" in read_result.output["content"]

        # List directory
        list_result = await fs_tool.execute("list_directory", {"path": project_dir})
        assert list_result.success is True
        assert len(list_result.output["entries"]) == 1
