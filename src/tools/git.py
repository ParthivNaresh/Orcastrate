"""
Git tool for repository management operations.

This module provides a concrete implementation of the Tool interface for Git
operations including repository management, branching, commits, and remotes.
"""

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    CostEstimate,
    Tool,
    ToolConfig,
    ToolError,
    ToolSchema,
    ValidationResult,
)


class GitTool(Tool):
    """
    Git tool for repository management operations.

    Provides functionality for:
    - Repository operations (init, clone, status, add, commit)
    - Branch management (create, switch, merge, delete)
    - Remote operations (fetch, pull, push)
    - Tag management (create, list, delete)
    - History and log operations
    """

    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self._git_available = False
        self._git_version: Optional[str] = None

    async def get_schema(self) -> ToolSchema:
        """Return the Git tool schema."""
        return ToolSchema(
            name="git",
            description="Git version control system tool",
            version=self.config.version,
            actions={
                "init": {
                    "description": "Initialize a new Git repository",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "bare": {"type": "boolean", "default": False},
                        "initial_branch": {"type": "string", "default": "main"},
                    },
                },
                "clone": {
                    "description": "Clone a remote repository",
                    "parameters": {
                        "url": {"type": "string", "required": True},
                        "destination": {"type": "string"},
                        "branch": {"type": "string"},
                        "depth": {"type": "integer"},
                        "recursive": {"type": "boolean", "default": False},
                    },
                },
                "status": {
                    "description": "Get repository status",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "porcelain": {"type": "boolean", "default": True},
                        "untracked": {"type": "boolean", "default": True},
                    },
                },
                "add": {
                    "description": "Add files to staging area",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "required": True,
                        },
                        "all": {"type": "boolean", "default": False},
                        "force": {"type": "boolean", "default": False},
                    },
                },
                "commit": {
                    "description": "Create a commit",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "message": {"type": "string", "required": True},
                        "author": {"type": "string"},
                        "amend": {"type": "boolean", "default": False},
                        "allow_empty": {"type": "boolean", "default": False},
                    },
                },
                "branch": {
                    "description": "Manage branches",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "action": {
                            "type": "string",
                            "enum": ["list", "create", "delete", "switch"],
                            "required": True,
                        },
                        "branch_name": {"type": "string"},
                        "start_point": {"type": "string"},
                        "force": {"type": "boolean", "default": False},
                    },
                },
                "merge": {
                    "description": "Merge branches",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "branch": {"type": "string", "required": True},
                        "no_ff": {"type": "boolean", "default": False},
                        "squash": {"type": "boolean", "default": False},
                        "strategy": {"type": "string"},
                    },
                },
                "pull": {
                    "description": "Pull changes from remote",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "remote": {"type": "string", "default": "origin"},
                        "branch": {"type": "string"},
                        "rebase": {"type": "boolean", "default": False},
                        "ff_only": {"type": "boolean", "default": False},
                    },
                },
                "push": {
                    "description": "Push changes to remote",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "remote": {"type": "string", "default": "origin"},
                        "branch": {"type": "string"},
                        "force": {"type": "boolean", "default": False},
                        "set_upstream": {"type": "boolean", "default": False},
                        "tags": {"type": "boolean", "default": False},
                    },
                },
                "log": {
                    "description": "Show commit history",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "max_count": {"type": "integer", "default": 10},
                        "oneline": {"type": "boolean", "default": False},
                        "graph": {"type": "boolean", "default": False},
                        "since": {"type": "string"},
                        "until": {"type": "string"},
                        "author": {"type": "string"},
                    },
                },
                "tag": {
                    "description": "Manage tags",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "action": {
                            "type": "string",
                            "enum": ["list", "create", "delete"],
                            "required": True,
                        },
                        "tag_name": {"type": "string"},
                        "message": {"type": "string"},
                        "commit": {"type": "string"},
                        "force": {"type": "boolean", "default": False},
                    },
                },
                "remote": {
                    "description": "Manage remotes",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "action": {
                            "type": "string",
                            "enum": ["list", "add", "remove", "set-url"],
                            "required": True,
                        },
                        "remote_name": {"type": "string"},
                        "url": {"type": "string"},
                    },
                },
                "diff": {
                    "description": "Show differences",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "staged": {"type": "boolean", "default": False},
                        "files": {"type": "array", "items": {"type": "string"}},
                        "commit1": {"type": "string"},
                        "commit2": {"type": "string"},
                        "name_only": {"type": "boolean", "default": False},
                    },
                },
            },
            required_permissions=["git"],
            dependencies=["git"],
            cost_model={
                "base_cost": 0.0,  # Git operations are typically free
                "clone_cost_per_mb": 0.001,
                "push_cost_per_mb": 0.001,
            },
        )

    async def estimate_cost(self, action: str, params: Dict[str, Any]) -> CostEstimate:
        """Estimate cost for Git operations."""
        base_cost = 0.0
        breakdown = {}

        if action == "clone":
            # Estimate based on potential repository size
            estimated_size_mb = 100  # Default estimate
            base_cost = estimated_size_mb * 0.001
            breakdown["clone_operation"] = base_cost

        elif action == "push":
            # Estimate based on potential data transfer
            estimated_size_mb = 10  # Default estimate for changes
            base_cost = estimated_size_mb * 0.001
            breakdown["push_operation"] = base_cost

        return CostEstimate(
            estimated_cost=base_cost,
            cost_breakdown=breakdown,
            confidence=0.8,
            factors=["Local operation", "Network bandwidth"],
        )

    async def _create_client(self) -> Any:
        """Create Git client (validate Git availability)."""
        try:
            # Check if Git is available
            result = await self._run_command(["git", "--version"])
            if result["returncode"] != 0:
                raise ToolError("Git is not available")

            self._git_version = result["stdout"].strip()
            self._git_available = True

            return {"git_available": True, "version": self._git_version}

        except Exception as e:
            raise ToolError(f"Failed to initialize Git client: {e}")

    async def _create_validator(self) -> Any:
        """Create parameter validator."""

        class GitValidator:
            def validate(self, action: str, params: Dict[str, Any]) -> ValidationResult:
                errors = []
                warnings = []
                normalized_params = params.copy()

                if action in [
                    "init",
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
                ]:
                    if not params.get("path"):
                        errors.append(f"path is required for {action}")
                    elif not self._is_valid_path(params["path"]):
                        errors.append(f"Invalid path format: {params['path']}")

                if action == "clone":
                    if not params.get("url"):
                        errors.append("url is required for clone")
                    elif not self._is_valid_git_url(params["url"]):
                        errors.append(f"Invalid Git URL format: {params['url']}")

                if action == "add":
                    if not params.get("files") and not params.get("all"):
                        errors.append(
                            "Either files list or all=True is required for add"
                        )

                if action == "commit":
                    if not params.get("message"):
                        errors.append("message is required for commit")
                    elif len(params["message"]) < 3:
                        warnings.append("Commit message is very short")

                if action in ["branch", "tag", "remote"]:
                    valid_actions = {
                        "branch": ["list", "create", "delete", "switch"],
                        "tag": ["list", "create", "delete"],
                        "remote": ["list", "add", "remove", "set-url"],
                    }
                    if (
                        not params.get("action")
                        or params["action"] not in valid_actions[action]
                    ):
                        errors.append(
                            f"Valid actions for {action}: {valid_actions[action]}"
                        )

                if action == "branch" and params.get("action") in [
                    "create",
                    "delete",
                    "switch",
                ]:
                    if not params.get("branch_name"):
                        errors.append("branch_name is required for branch operations")

                if action == "merge":
                    if not params.get("branch"):
                        errors.append("branch is required for merge")

                return ValidationResult(
                    valid=len(errors) == 0,
                    errors=errors,
                    warnings=warnings,
                    normalized_params=normalized_params,
                )

            def _is_valid_path(self, path: str) -> bool:
                """Validate path format."""
                try:
                    Path(path)
                    return True
                except (ValueError, OSError):
                    return False

            def _is_valid_git_url(self, url: str) -> bool:
                """Validate Git URL format."""
                # Check for common Git URL patterns
                git_patterns = [
                    r"^https?://.*\.git$",
                    r"^git@.*:.*\.git$",
                    r"^ssh://.*\.git$",
                    r"^https?://github\.com/.*",
                    r"^https?://gitlab\.com/.*",
                    r"^https?://bitbucket\.org/.*",
                ]

                return any(re.match(pattern, url) for pattern in git_patterns)

        return GitValidator()

    async def _execute_action(
        self, action: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Git action."""
        if not self._git_available:
            raise ToolError("Git is not available")

        if action == "init":
            return await self._git_init(params)
        elif action == "clone":
            return await self._git_clone(params)
        elif action == "status":
            return await self._git_status(params)
        elif action == "add":
            return await self._git_add(params)
        elif action == "commit":
            return await self._git_commit(params)
        elif action == "branch":
            return await self._git_branch(params)
        elif action == "merge":
            return await self._git_merge(params)
        elif action == "pull":
            return await self._git_pull(params)
        elif action == "push":
            return await self._git_push(params)
        elif action == "log":
            return await self._git_log(params)
        elif action == "tag":
            return await self._git_tag(params)
        elif action == "remote":
            return await self._git_remote(params)
        elif action == "diff":
            return await self._git_diff(params)
        else:
            raise ToolError(f"Unknown action: {action}")

    async def _git_init(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize a Git repository."""
        cmd = ["git", "init"]

        if params.get("bare"):
            cmd.append("--bare")

        if params.get("initial_branch"):
            cmd.extend(["--initial-branch", params["initial_branch"]])

        cmd.append(params["path"])

        result = await self._run_command(cmd)

        return {
            "path": params["path"],
            "bare": params.get("bare", False),
            "initial_branch": params.get("initial_branch", "main"),
            "success": result["returncode"] == 0,
            "output": result["stdout"],
        }

    async def _git_clone(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clone a Git repository."""
        cmd = ["git", "clone"]

        if params.get("branch"):
            cmd.extend(["--branch", params["branch"]])

        if params.get("depth"):
            cmd.extend(["--depth", str(params["depth"])])

        if params.get("recursive"):
            cmd.append("--recursive")

        cmd.append(params["url"])

        if params.get("destination"):
            cmd.append(params["destination"])

        result = await self._run_command(cmd)

        return {
            "url": params["url"],
            "destination": params.get("destination"),
            "branch": params.get("branch"),
            "success": result["returncode"] == 0,
            "output": result["stdout"],
        }

    async def _git_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get Git repository status."""
        cmd = ["git", "-C", params["path"], "status"]

        if params.get("porcelain"):
            cmd.append("--porcelain")

        if params.get("untracked"):
            cmd.append("--untracked-files=all")

        result = await self._run_command(cmd)

        return {
            "path": params["path"],
            "status": result["stdout"],
            "clean": result["returncode"] == 0 and not result["stdout"].strip(),
            "success": result["returncode"] == 0,
        }

    async def _git_add(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add files to Git staging area."""
        cmd = ["git", "-C", params["path"], "add"]

        if params.get("force"):
            cmd.append("--force")

        if params.get("all"):
            cmd.append("--all")
        else:
            cmd.extend(params["files"])

        result = await self._run_command(cmd)

        return {
            "path": params["path"],
            "files": params.get("files", []),
            "all": params.get("all", False),
            "success": result["returncode"] == 0,
            "output": result["stdout"],
        }

    async def _git_commit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Git commit."""
        cmd = ["git", "-C", params["path"], "commit", "-m", params["message"]]

        if params.get("author"):
            cmd.extend(["--author", params["author"]])

        if params.get("amend"):
            cmd.append("--amend")

        if params.get("allow_empty"):
            cmd.append("--allow-empty")

        result = await self._run_command(cmd)

        return {
            "path": params["path"],
            "message": params["message"],
            "author": params.get("author"),
            "success": result["returncode"] == 0,
            "output": result["stdout"],
        }

    async def _git_branch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Manage Git branches."""
        action = params["action"]
        cmd = ["git", "-C", params["path"], "branch"]

        if action == "list":
            cmd.append("--list")
        elif action == "create":
            cmd.append(params["branch_name"])
            if params.get("start_point"):
                cmd.append(params["start_point"])
        elif action == "delete":
            if params.get("force"):
                cmd.append("-D")
            else:
                cmd.append("-d")
            cmd.append(params["branch_name"])
        elif action == "switch":
            cmd = ["git", "-C", params["path"], "switch", params["branch_name"]]

        result = await self._run_command(cmd)

        return {
            "path": params["path"],
            "action": action,
            "branch_name": params.get("branch_name"),
            "success": result["returncode"] == 0,
            "output": result["stdout"],
        }

    async def _git_merge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge Git branches."""
        cmd = ["git", "-C", params["path"], "merge"]

        if params.get("no_ff"):
            cmd.append("--no-ff")

        if params.get("squash"):
            cmd.append("--squash")

        if params.get("strategy"):
            cmd.extend(["--strategy", params["strategy"]])

        cmd.append(params["branch"])

        result = await self._run_command(cmd)

        return {
            "path": params["path"],
            "branch": params["branch"],
            "success": result["returncode"] == 0,
            "output": result["stdout"],
        }

    async def _git_pull(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Pull changes from Git remote."""
        cmd = ["git", "-C", params["path"], "pull"]

        if params.get("rebase"):
            cmd.append("--rebase")

        if params.get("ff_only"):
            cmd.append("--ff-only")

        cmd.append(params.get("remote", "origin"))

        if params.get("branch"):
            cmd.append(params["branch"])

        result = await self._run_command(cmd)

        return {
            "path": params["path"],
            "remote": params.get("remote", "origin"),
            "branch": params.get("branch"),
            "success": result["returncode"] == 0,
            "output": result["stdout"],
        }

    async def _git_push(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Push changes to Git remote."""
        cmd = ["git", "-C", params["path"], "push"]

        if params.get("force"):
            cmd.append("--force")

        if params.get("set_upstream"):
            cmd.append("--set-upstream")

        if params.get("tags"):
            cmd.append("--tags")

        cmd.append(params.get("remote", "origin"))

        if params.get("branch"):
            cmd.append(params["branch"])

        result = await self._run_command(cmd)

        return {
            "path": params["path"],
            "remote": params.get("remote", "origin"),
            "branch": params.get("branch"),
            "success": result["returncode"] == 0,
            "output": result["stdout"],
        }

    async def _git_log(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get Git commit history."""
        cmd = ["git", "-C", params["path"], "log"]

        if params.get("max_count"):
            cmd.extend(["-n", str(params["max_count"])])

        if params.get("oneline"):
            cmd.append("--oneline")

        if params.get("graph"):
            cmd.append("--graph")

        if params.get("since"):
            cmd.extend(["--since", params["since"]])

        if params.get("until"):
            cmd.extend(["--until", params["until"]])

        if params.get("author"):
            cmd.extend(["--author", params["author"]])

        result = await self._run_command(cmd)

        return {
            "path": params["path"],
            "log": result["stdout"],
            "success": result["returncode"] == 0,
        }

    async def _git_tag(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Manage Git tags."""
        action = params["action"]
        cmd = ["git", "-C", params["path"], "tag"]

        if action == "list":
            cmd.append("--list")
        elif action == "create":
            if params.get("message"):
                cmd.extend(["-a", params["tag_name"], "-m", params["message"]])
            else:
                cmd.append(params["tag_name"])
            if params.get("commit"):
                cmd.append(params["commit"])
        elif action == "delete":
            cmd.extend(["-d", params["tag_name"]])

        result = await self._run_command(cmd)

        return {
            "path": params["path"],
            "action": action,
            "tag_name": params.get("tag_name"),
            "success": result["returncode"] == 0,
            "output": result["stdout"],
        }

    async def _git_remote(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Manage Git remotes."""
        action = params["action"]
        cmd = ["git", "-C", params["path"], "remote"]

        if action == "list":
            cmd.append("-v")
        elif action == "add":
            cmd.extend(["add", params["remote_name"], params["url"]])
        elif action == "remove":
            cmd.extend(["remove", params["remote_name"]])
        elif action == "set-url":
            cmd.extend(["set-url", params["remote_name"], params["url"]])

        result = await self._run_command(cmd)

        return {
            "path": params["path"],
            "action": action,
            "remote_name": params.get("remote_name"),
            "url": params.get("url"),
            "success": result["returncode"] == 0,
            "output": result["stdout"],
        }

    async def _git_diff(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Show Git differences."""
        cmd = ["git", "-C", params["path"], "diff"]

        if params.get("staged"):
            cmd.append("--staged")

        if params.get("name_only"):
            cmd.append("--name-only")

        if params.get("commit1") and params.get("commit2"):
            cmd.extend([params["commit1"], params["commit2"]])
        elif params.get("commit1"):
            cmd.append(params["commit1"])

        if params.get("files"):
            cmd.append("--")
            cmd.extend(params["files"])

        result = await self._run_command(cmd)

        return {
            "path": params["path"],
            "diff": result["stdout"],
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
        # For Git, rollback might involve reverting commits or resetting
        # This would depend on the specific operation that was performed
        return {
            "rollback_type": "git_operation",
            "execution_id": execution_id,
            "message": "Git rollback operations are action-specific",
        }

    async def _get_supported_actions(self) -> List[str]:
        """Get list of supported actions."""
        return [
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
