"""
Tests for Git tool implementation.
"""

from unittest.mock import patch

import pytest

from src.tools.base import ToolError
from src.tools.git import GitTool


class TestGitTool:
    """Test Git tool functionality."""

    @pytest.fixture
    def git_tool(self, git_config):
        """Create Git tool instance."""
        return GitTool(git_config)

    @pytest.mark.asyncio
    async def test_tool_initialization(self, git_tool):
        """Test Git tool initialization."""
        with patch.object(git_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "git version 2.34.1",
                "stderr": "",
            }

            await git_tool.initialize()

            assert git_tool._git_available is True
            assert "git version" in git_tool._git_version

    @pytest.mark.asyncio
    async def test_initialization_git_not_available(self, git_tool):
        """Test initialization when Git is not available."""
        with patch.object(git_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 1,
                "stdout": "",
                "stderr": "Command not found",
            }

            with pytest.raises(ToolError, match="Git is not available"):
                await git_tool.initialize()

    @pytest.mark.asyncio
    async def test_get_schema(self, git_tool):
        """Test getting tool schema."""
        schema = await git_tool.get_schema()

        assert schema.name == "git"
        assert schema.description == "Git version control system tool"
        assert "init" in schema.actions
        assert "clone" in schema.actions
        assert "commit" in schema.actions
        assert "git" in schema.required_permissions
        assert "git" in schema.dependencies

    @pytest.mark.asyncio
    async def test_estimate_cost_clone(self, git_tool):
        """Test cost estimation for cloning repositories."""
        cost = await git_tool.estimate_cost("clone", {})

        assert cost.estimated_cost > 0
        assert "clone_operation" in cost.cost_breakdown
        assert cost.confidence == 0.8

    @pytest.mark.asyncio
    async def test_estimate_cost_push(self, git_tool):
        """Test cost estimation for pushing changes."""
        cost = await git_tool.estimate_cost("push", {})

        assert cost.estimated_cost > 0
        assert "push_operation" in cost.cost_breakdown

    @pytest.mark.asyncio
    async def test_validate_init_params(self, git_tool):
        """Test parameter validation for init action."""
        validator = await git_tool._create_validator()

        # Valid parameters
        result = validator.validate(
            "init",
            {
                "path": "/path/to/repo",
                "bare": False,
                "initial_branch": "main",
            },
        )
        assert result.valid is True

        # Missing required parameters
        result = validator.validate("init", {})
        assert result.valid is False
        assert "path is required for init" in result.errors

    @pytest.mark.asyncio
    async def test_validate_clone_params(self, git_tool):
        """Test parameter validation for clone action."""
        validator = await git_tool._create_validator()

        # Valid parameters
        result = validator.validate(
            "clone",
            {
                "url": "https://github.com/user/repo.git",
                "destination": "/path/to/destination",
            },
        )
        assert result.valid is True

        # Missing required parameters
        result = validator.validate("clone", {})
        assert result.valid is False
        assert "url is required for clone" in result.errors

        # Invalid URL format
        result = validator.validate(
            "clone",
            {
                "url": "invalid-url",
            },
        )
        assert result.valid is False
        assert "Invalid Git URL format: invalid-url" in result.errors

    @pytest.mark.asyncio
    async def test_validate_commit_params(self, git_tool):
        """Test parameter validation for commit action."""
        validator = await git_tool._create_validator()

        # Valid parameters
        result = validator.validate(
            "commit",
            {
                "path": "/path/to/repo",
                "message": "Add new feature",
            },
        )
        assert result.valid is True

        # Missing required parameters
        result = validator.validate(
            "commit",
            {
                "path": "/path/to/repo",
            },
        )
        assert result.valid is False
        assert "message is required for commit" in result.errors

        # Short commit message warning
        result = validator.validate(
            "commit",
            {
                "path": "/path/to/repo",
                "message": "x",  # Only 1 character
            },
        )
        assert result.valid is True
        assert "Commit message is very short" in result.warnings

    @pytest.mark.asyncio
    async def test_validate_branch_params(self, git_tool):
        """Test parameter validation for branch action."""
        validator = await git_tool._create_validator()

        # Valid parameters
        result = validator.validate(
            "branch",
            {
                "path": "/path/to/repo",
                "action": "create",
                "branch_name": "feature-branch",
            },
        )
        assert result.valid is True

        # Invalid action
        result = validator.validate(
            "branch",
            {
                "path": "/path/to/repo",
                "action": "invalid",
            },
        )
        assert result.valid is False
        assert (
            "Valid actions for branch: ['list', 'create', 'delete', 'switch']"
            in result.errors
        )

        # Missing branch name for operations
        result = validator.validate(
            "branch",
            {
                "path": "/path/to/repo",
                "action": "create",
            },
        )
        assert result.valid is False
        assert "branch_name is required for branch operations" in result.errors

    @pytest.mark.asyncio
    async def test_git_init_success(self, git_tool):
        """Test successful Git repository initialization."""
        with patch.object(git_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "Initialized empty Git repository",
                "stderr": "",
            }

            git_tool._git_available = True

            result = await git_tool._git_init(
                {
                    "path": "/path/to/repo",
                    "bare": False,
                    "initial_branch": "main",
                }
            )

            assert result["success"] is True
            assert result["path"] == "/path/to/repo"
            assert result["bare"] is False
            assert result["initial_branch"] == "main"

            # Verify command was constructed correctly
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "git" in call_args
            assert "init" in call_args
            assert "--initial-branch" in call_args
            assert "main" in call_args

    @pytest.mark.asyncio
    async def test_git_clone_success(self, git_tool):
        """Test successful Git repository cloning."""
        with patch.object(git_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "Cloning into 'repo'...\ndone.",
                "stderr": "",
            }

            git_tool._git_available = True

            result = await git_tool._git_clone(
                {
                    "url": "https://github.com/user/repo.git",
                    "destination": "/path/to/destination",
                    "branch": "develop",
                    "depth": 1,
                    "recursive": True,
                }
            )

            assert result["success"] is True
            assert result["url"] == "https://github.com/user/repo.git"
            assert result["destination"] == "/path/to/destination"
            assert result["branch"] == "develop"

            # Verify command was constructed correctly
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "git" in call_args
            assert "clone" in call_args
            assert "--branch" in call_args
            assert "develop" in call_args
            assert "--depth" in call_args
            assert "--recursive" in call_args

    @pytest.mark.asyncio
    async def test_git_status_success(self, git_tool):
        """Test successful Git status check."""
        with patch.object(git_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "M  file1.txt\n?? file2.txt",
                "stderr": "",
            }

            git_tool._git_available = True

            result = await git_tool._git_status(
                {
                    "path": "/path/to/repo",
                    "porcelain": True,
                    "untracked": True,
                }
            )

            assert result["success"] is True
            assert result["path"] == "/path/to/repo"
            assert result["clean"] is False  # Has changes
            assert "file1.txt" in result["status"]

    @pytest.mark.asyncio
    async def test_git_add_success(self, git_tool):
        """Test successful Git add operation."""
        with patch.object(git_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "",
                "stderr": "",
            }

            git_tool._git_available = True

            result = await git_tool._git_add(
                {
                    "path": "/path/to/repo",
                    "files": ["file1.txt", "file2.txt"],
                    "force": False,
                }
            )

            assert result["success"] is True
            assert result["path"] == "/path/to/repo"
            assert result["files"] == ["file1.txt", "file2.txt"]

            # Verify command was constructed correctly
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "git" in call_args
            assert "add" in call_args
            assert "file1.txt" in call_args
            assert "file2.txt" in call_args

    @pytest.mark.asyncio
    async def test_git_commit_success(self, git_tool):
        """Test successful Git commit operation."""
        with patch.object(git_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "[main abc123] Add new feature\n 2 files changed, 10 insertions(+)",
                "stderr": "",
            }

            git_tool._git_available = True

            result = await git_tool._git_commit(
                {
                    "path": "/path/to/repo",
                    "message": "Add new feature",
                    "author": "John Doe <john@example.com>",
                    "amend": False,
                }
            )

            assert result["success"] is True
            assert result["path"] == "/path/to/repo"
            assert result["message"] == "Add new feature"
            assert result["author"] == "John Doe <john@example.com>"

            # Verify command was constructed correctly
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "git" in call_args
            assert "commit" in call_args
            assert "-m" in call_args
            assert "Add new feature" in call_args
            assert "--author" in call_args

    @pytest.mark.asyncio
    async def test_git_branch_list_success(self, git_tool):
        """Test successful Git branch listing."""
        with patch.object(git_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "* main\n  feature-branch\n  develop",
                "stderr": "",
            }

            git_tool._git_available = True

            result = await git_tool._git_branch(
                {
                    "path": "/path/to/repo",
                    "action": "list",
                }
            )

            assert result["success"] is True
            assert result["action"] == "list"
            assert "main" in result["output"]
            assert "feature-branch" in result["output"]

    @pytest.mark.asyncio
    async def test_git_branch_create_success(self, git_tool):
        """Test successful Git branch creation."""
        with patch.object(git_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "",
                "stderr": "",
            }

            git_tool._git_available = True

            result = await git_tool._git_branch(
                {
                    "path": "/path/to/repo",
                    "action": "create",
                    "branch_name": "feature-branch",
                    "start_point": "main",
                }
            )

            assert result["success"] is True
            assert result["action"] == "create"
            assert result["branch_name"] == "feature-branch"

    @pytest.mark.asyncio
    async def test_git_pull_success(self, git_tool):
        """Test successful Git pull operation."""
        with patch.object(git_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "Already up to date.",
                "stderr": "",
            }

            git_tool._git_available = True

            result = await git_tool._git_pull(
                {
                    "path": "/path/to/repo",
                    "remote": "origin",
                    "branch": "main",
                    "rebase": True,
                }
            )

            assert result["success"] is True
            assert result["remote"] == "origin"
            assert result["branch"] == "main"

            # Verify command was constructed correctly
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "git" in call_args
            assert "pull" in call_args
            assert "--rebase" in call_args
            assert "origin" in call_args
            assert "main" in call_args

    @pytest.mark.asyncio
    async def test_git_push_success(self, git_tool):
        """Test successful Git push operation."""
        with patch.object(git_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "Everything up-to-date",
                "stderr": "",
            }

            git_tool._git_available = True

            result = await git_tool._git_push(
                {
                    "path": "/path/to/repo",
                    "remote": "origin",
                    "branch": "main",
                    "force": False,
                    "set_upstream": True,
                }
            )

            assert result["success"] is True
            assert result["remote"] == "origin"
            assert result["branch"] == "main"

            # Verify command was constructed correctly
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "git" in call_args
            assert "push" in call_args
            assert "--set-upstream" in call_args
            assert "origin" in call_args
            assert "main" in call_args

    @pytest.mark.asyncio
    async def test_git_log_success(self, git_tool):
        """Test successful Git log operation."""
        mock_log_output = """commit abc123 (HEAD -> main)
Author: John Doe <john@example.com>
Date:   Mon Jan 1 00:00:00 2023 +0000

    Add new feature

commit def456
Author: Jane Doe <jane@example.com>
Date:   Sun Dec 31 23:59:59 2022 +0000

    Initial commit"""

        with patch.object(git_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": mock_log_output,
                "stderr": "",
            }

            git_tool._git_available = True

            result = await git_tool._git_log(
                {
                    "path": "/path/to/repo",
                    "max_count": 5,
                    "oneline": False,
                    "graph": True,
                    "author": "John Doe",
                }
            )

            assert result["success"] is True
            assert "Add new feature" in result["log"]
            assert "John Doe" in result["log"]

    @pytest.mark.asyncio
    async def test_execute_action_unknown(self, git_tool):
        """Test executing unknown action."""
        git_tool._git_available = True

        with pytest.raises(ToolError, match="Unknown action"):
            await git_tool._execute_action("unknown_action", {})

    @pytest.mark.asyncio
    async def test_execute_action_git_not_available(self, git_tool):
        """Test executing action when Git is not available."""
        git_tool._git_available = False

        with pytest.raises(ToolError, match="Git is not available"):
            await git_tool._execute_action("init", {})

    @pytest.mark.asyncio
    async def test_git_remote_operations(self, git_tool):
        """Test Git remote management operations."""
        with patch.object(git_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "origin\thttps://github.com/user/repo.git (fetch)\norigin\thttps://github.com/user/repo.git (push)",
                "stderr": "",
            }

            git_tool._git_available = True

            # Test remote list
            result = await git_tool._git_remote(
                {
                    "path": "/path/to/repo",
                    "action": "list",
                }
            )

            assert result["success"] is True
            assert result["action"] == "list"
            assert "origin" in result["output"]

            # Test remote add
            mock_run.reset_mock()
            mock_run.return_value = {"returncode": 0, "stdout": "", "stderr": ""}

            result = await git_tool._git_remote(
                {
                    "path": "/path/to/repo",
                    "action": "add",
                    "remote_name": "upstream",
                    "url": "https://github.com/upstream/repo.git",
                }
            )

            assert result["success"] is True
            assert result["action"] == "add"
            assert result["remote_name"] == "upstream"

    @pytest.mark.asyncio
    async def test_git_tag_operations(self, git_tool):
        """Test Git tag management operations."""
        with patch.object(git_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "v1.0.0\nv1.1.0\nv2.0.0",
                "stderr": "",
            }

            git_tool._git_available = True

            # Test tag list
            result = await git_tool._git_tag(
                {
                    "path": "/path/to/repo",
                    "action": "list",
                }
            )

            assert result["success"] is True
            assert result["action"] == "list"
            assert "v1.0.0" in result["output"]

            # Test tag create
            mock_run.reset_mock()
            mock_run.return_value = {"returncode": 0, "stdout": "", "stderr": ""}

            result = await git_tool._git_tag(
                {
                    "path": "/path/to/repo",
                    "action": "create",
                    "tag_name": "v2.1.0",
                    "message": "Release version 2.1.0",
                }
            )

            assert result["success"] is True
            assert result["action"] == "create"
            assert result["tag_name"] == "v2.1.0"

    @pytest.mark.asyncio
    async def test_git_diff_operations(self, git_tool):
        """Test Git diff operations."""
        mock_diff_output = """diff --git a/file.txt b/file.txt
index abc123..def456 100644
--- a/file.txt
+++ b/file.txt
@@ -1,3 +1,4 @@
 line 1
 line 2
+new line
 line 3"""

        with patch.object(git_tool, "_run_command") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": mock_diff_output,
                "stderr": "",
            }

            git_tool._git_available = True

            result = await git_tool._git_diff(
                {
                    "path": "/path/to/repo",
                    "staged": False,
                    "files": ["file.txt"],
                    "name_only": False,
                }
            )

            assert result["success"] is True
            assert "new line" in result["diff"]
            assert "file.txt" in result["diff"]

    @pytest.mark.asyncio
    async def test_run_command_error_handling(self, git_tool):
        """Test command execution error handling."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_subprocess.side_effect = Exception("Process creation failed")

            result = await git_tool._run_command(["git", "--version"])

            assert result["returncode"] == -1
            assert "Process creation failed" in result["stderr"]

    @pytest.mark.asyncio
    async def test_get_supported_actions(self, git_tool):
        """Test getting supported actions."""
        actions = await git_tool._get_supported_actions()

        expected_actions = [
            "init",
            "clone",
            "status",
            "add",
            "commit",
            "branch",
            "merge",
            "pull",
            "push",
            "log",
            "tag",
            "remote",
            "diff",
        ]

        assert all(action in actions for action in expected_actions)

    @pytest.mark.asyncio
    async def test_full_workflow_integration(self, git_tool):
        """Test a complete Git workflow."""
        with patch.object(git_tool, "_run_command") as mock_run:
            # Mock successful responses for each step
            mock_run.side_effect = [
                # Git version check
                {"returncode": 0, "stdout": "git version 2.34.1", "stderr": ""},
                # Init repository
                {
                    "returncode": 0,
                    "stdout": "Initialized empty Git repository",
                    "stderr": "",
                },
                # Add files
                {"returncode": 0, "stdout": "", "stderr": ""},
                # Commit changes
                {
                    "returncode": 0,
                    "stdout": "[main abc123] Initial commit",
                    "stderr": "",
                },
                # Check status
                {
                    "returncode": 0,
                    "stdout": "nothing to commit, working tree clean",
                    "stderr": "",
                },
            ]

            # Initialize the tool
            await git_tool.initialize()
            assert git_tool._git_available is True

            # Initialize a repository
            init_result = await git_tool.execute("init", {"path": "/tmp/test-repo"})
            assert init_result.success is True

            # Add files
            add_result = await git_tool.execute(
                "add",
                {
                    "path": "/tmp/test-repo",
                    "files": ["README.md"],
                },
            )
            assert add_result.success is True

            # Commit changes
            commit_result = await git_tool.execute(
                "commit",
                {
                    "path": "/tmp/test-repo",
                    "message": "Initial commit",
                },
            )
            assert commit_result.success is True

            # Check status
            status_result = await git_tool.execute(
                "status",
                {
                    "path": "/tmp/test-repo",
                },
            )
            assert status_result.success is True
