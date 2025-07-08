"""
File System tool for file and directory operations.

This module provides a concrete implementation of the Tool interface for
file system operations with security validation and path traversal protection.
"""

import json
import os
import shutil
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


class FileSystemTool(Tool):
    """
    File System tool for secure file and directory operations.

    Provides functionality for:
    - Directory operations (create, list, remove)
    - File operations (read, write, copy, move, delete)
    - Permission management
    - Security validation and path traversal protection
    """

    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self._base_path: Optional[Path] = None
        self._allowed_patterns: List[str] = []

    async def get_schema(self) -> ToolSchema:
        """Return the File System tool schema."""
        return ToolSchema(
            name="filesystem",
            description="File system operations tool with security validation",
            version=self.config.version,
            actions={
                "create_directory": {
                    "description": "Create a directory",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "mode": {"type": "string", "default": "755"},
                        "parents": {"type": "boolean", "default": True},
                    },
                },
                "list_directory": {
                    "description": "List directory contents",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "recursive": {"type": "boolean", "default": False},
                        "include_hidden": {"type": "boolean", "default": False},
                    },
                },
                "remove_directory": {
                    "description": "Remove a directory",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "recursive": {"type": "boolean", "default": False},
                        "force": {"type": "boolean", "default": False},
                    },
                },
                "write_file": {
                    "description": "Write content to a file",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "content": {"type": ["string", "object"], "required": True},
                        "mode": {"type": "string", "default": "644"},
                        "create_parents": {"type": "boolean", "default": True},
                        "encoding": {"type": "string", "default": "utf-8"},
                    },
                },
                "read_file": {
                    "description": "Read content from a file",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "encoding": {"type": "string", "default": "utf-8"},
                        "max_size": {"type": "integer", "default": 10485760},  # 10MB
                    },
                },
                "copy_file": {
                    "description": "Copy a file",
                    "parameters": {
                        "source": {"type": "string", "required": True},
                        "destination": {"type": "string", "required": True},
                        "preserve_permissions": {"type": "boolean", "default": True},
                        "create_parents": {"type": "boolean", "default": True},
                    },
                },
                "move_file": {
                    "description": "Move or rename a file",
                    "parameters": {
                        "source": {"type": "string", "required": True},
                        "destination": {"type": "string", "required": True},
                        "create_parents": {"type": "boolean", "default": True},
                    },
                },
                "delete_file": {
                    "description": "Delete a file",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "force": {"type": "boolean", "default": False},
                    },
                },
                "get_info": {
                    "description": "Get file or directory information",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                    },
                },
                "set_permissions": {
                    "description": "Set file or directory permissions",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "mode": {"type": "string", "required": True},
                        "recursive": {"type": "boolean", "default": False},
                    },
                },
            },
            required_permissions=["filesystem"],
            dependencies=[],
            cost_model={
                "base_cost": 0.0,  # File operations are typically free
                "read_cost_per_mb": 0.001,
                "write_cost_per_mb": 0.002,
            },
        )

    async def estimate_cost(self, action: str, params: Dict[str, Any]) -> CostEstimate:
        """Estimate cost for file system operations."""
        base_cost = 0.0
        breakdown = {}

        if action in ["read_file"]:
            # Estimate based on potential file size
            max_size_mb = params.get("max_size", 10485760) / (1024 * 1024)
            base_cost = max_size_mb * 0.001
            breakdown["read_operation"] = base_cost

        elif action in ["write_file", "copy_file"]:
            # Estimate based on content size
            content = params.get("content", "")
            if isinstance(content, str):
                size_mb = len(content.encode("utf-8")) / (1024 * 1024)
            else:
                # For objects, estimate JSON size
                size_mb = len(json.dumps(content).encode("utf-8")) / (1024 * 1024)

            base_cost = size_mb * 0.002
            breakdown["write_operation"] = base_cost

        return CostEstimate(
            estimated_cost=base_cost,
            cost_breakdown=breakdown,
            confidence=0.9,
            factors=["Local file system", "No network transfer"],
        )

    async def _create_client(self) -> Any:
        """Create file system client (validate permissions)."""
        try:
            # Set base path for operations (security boundary)
            self._base_path = Path("/tmp/orcastrate")

            # Create base directory if it doesn't exist
            self._base_path.mkdir(parents=True, exist_ok=True)

            # Test write permissions
            test_file = self._base_path / ".test_write"
            test_file.write_text("test")
            test_file.unlink()

            return {"filesystem_available": True, "base_path": str(self._base_path)}

        except Exception as e:
            raise ToolError(f"Failed to initialize file system client: {e}")

    async def _create_validator(self) -> Any:
        """Create parameter validator."""

        class FileSystemValidator:
            def __init__(self, tool_instance):
                self.tool = tool_instance

            def validate(self, action: str, params: Dict[str, Any]) -> ValidationResult:
                errors = []
                warnings = []
                normalized_params = params.copy()

                # Path validation for all actions
                if action in [
                    "create_directory",
                    "list_directory",
                    "remove_directory",
                    "write_file",
                    "read_file",
                    "delete_file",
                    "get_info",
                    "set_permissions",
                ]:
                    if not params.get("path"):
                        errors.append(f"path is required for {action}")
                    else:
                        path_errors = self._validate_path(params["path"])
                        errors.extend(path_errors)

                # Source/destination validation
                if action in ["copy_file", "move_file"]:
                    if not params.get("source"):
                        errors.append(f"source is required for {action}")
                    else:
                        source_errors = self._validate_path(params["source"])
                        errors.extend([f"source: {e}" for e in source_errors])

                    if not params.get("destination"):
                        errors.append(f"destination is required for {action}")
                    else:
                        dest_errors = self._validate_path(params["destination"])
                        errors.extend([f"destination: {e}" for e in dest_errors])

                # Content validation for write operations
                if action == "write_file":
                    if "content" not in params:
                        errors.append("content is required for write_file")
                    elif isinstance(params["content"], dict):
                        # Validate JSON serializable (strict mode)
                        try:
                            json.dumps(params["content"], allow_nan=False)
                        except (TypeError, ValueError):
                            errors.append("content object must be JSON serializable")

                # Mode validation
                if "mode" in params:
                    if not self._validate_mode(params["mode"]):
                        errors.append(f"Invalid mode format: {params['mode']}")

                # Size limits for read operations
                if (
                    action == "read_file" and params.get("max_size", 0) > 104857600
                ):  # 100MB
                    warnings.append("Large max_size may impact performance")

                return ValidationResult(
                    valid=len(errors) == 0,
                    errors=errors,
                    warnings=warnings,
                    normalized_params=normalized_params,
                )

            def _validate_path(self, path: str) -> List[str]:
                """Validate path for security."""
                errors = []

                try:
                    path_obj = Path(path).resolve()

                    # Check for path traversal attempts
                    if ".." in path:
                        errors.append("Path traversal not allowed")

                    # Check for absolute paths outside allowed areas
                    if path_obj.is_absolute():
                        # Allow paths under /tmp/orcastrate (handle symlinks)
                        allowed_base = Path("/tmp/orcastrate").resolve()
                        if not str(path_obj).startswith(str(allowed_base)):
                            errors.append(
                                "Absolute paths outside /tmp/orcastrate not allowed"
                            )

                    # Check for dangerous paths
                    dangerous_paths = ["/etc", "/usr", "/var", "/sys", "/proc", "/dev"]
                    for dangerous in dangerous_paths:
                        if str(path_obj).startswith(dangerous):
                            errors.append(f"Access to {dangerous} not allowed")

                except (ValueError, OSError) as e:
                    errors.append(f"Invalid path: {e}")

                return errors

            def _validate_mode(self, mode: str) -> bool:
                """Validate file mode format."""
                try:
                    # Support both octal string and symbolic notation
                    if mode.isdigit() and len(mode) == 3:
                        int(mode, 8)  # Validate octal
                        return True
                    # Could add symbolic mode validation here
                    return False
                except ValueError:
                    return False

        return FileSystemValidator(self)

    async def _execute_action(
        self, action: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute file system action."""
        if action == "create_directory":
            return await self._create_directory(params)
        elif action == "list_directory":
            return await self._list_directory(params)
        # elif action == "remove_directory":
        #     return await self._remove_directory(params)
        elif action == "write_file":
            return await self._write_file(params)
        elif action == "read_file":
            return await self._read_file(params)
        elif action == "copy_file":
            return await self._copy_file(params)
        elif action == "move_file":
            return await self._move_file(params)
        elif action == "delete_file":
            return await self._delete_file(params)
        elif action == "get_info":
            return await self._get_info(params)
        elif action == "set_permissions":
            return await self._set_permissions(params)
        else:
            raise ToolError(f"Unknown action: {action}")

    async def _create_directory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a directory."""
        try:
            path = Path(params["path"])
            mode_str = params.get("mode", "755")
            parents = params.get("parents", True)

            # Create directory
            path.mkdir(mode=int(mode_str, 8), parents=parents, exist_ok=True)

            return {
                "path": str(path),
                "created": True,
                "mode": mode_str,
                "success": True,
            }

        except Exception as e:
            return {
                "path": params["path"],
                "created": False,
                "success": False,
                "error": str(e),
            }

    async def _list_directory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List directory contents."""
        try:
            path = Path(params["path"])
            recursive = params.get("recursive", False)
            include_hidden = params.get("include_hidden", False)

            if not path.exists():
                raise FileNotFoundError(f"Directory not found: {path}")

            if not path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {path}")

            entries = []

            if recursive:
                pattern = "**/*" if include_hidden else "**/[!.]*"
                for item in path.rglob(pattern):
                    entries.append(self._get_file_info(item))
            else:
                for item in path.iterdir():
                    if not include_hidden and item.name.startswith("."):
                        continue
                    entries.append(self._get_file_info(item))

            return {
                "path": str(path),
                "entries": entries,
                "count": len(entries),
                "success": True,
            }

        except Exception as e:
            return {
                "path": params["path"],
                "entries": [],
                "success": False,
                "error": str(e),
            }

    async def _write_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Write content to a file."""
        try:
            path = Path(params["path"])
            content = params["content"]
            mode_str = params.get("mode", "644")
            create_parents = params.get("create_parents", True)
            encoding = params.get("encoding", "utf-8")

            # Create parent directories if needed
            if create_parents and path.parent != path:
                path.parent.mkdir(parents=True, exist_ok=True)

            # Handle different content types
            if isinstance(content, dict) or isinstance(content, list):
                # JSON content
                content_str = json.dumps(content, indent=2, ensure_ascii=False)
            else:
                content_str = str(content)

            # Write file
            path.write_text(content_str, encoding=encoding)

            # Set permissions
            path.chmod(int(mode_str, 8))

            return {
                "path": str(path),
                "size": len(content_str.encode(encoding)),
                "mode": mode_str,
                "encoding": encoding,
                "success": True,
            }

        except Exception as e:
            return {
                "path": params["path"],
                "success": False,
                "error": str(e),
            }

    async def _read_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read content from a file."""
        try:
            path = Path(params["path"])
            encoding = params.get("encoding", "utf-8")
            max_size = params.get("max_size", 10485760)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            if not path.is_file():
                raise IsADirectoryError(f"Path is not a file: {path}")

            # Check file size
            file_size = path.stat().st_size
            if file_size > max_size:
                raise ToolError(f"File too large: {file_size} bytes (max: {max_size})")

            # Read file
            content = path.read_text(encoding=encoding)

            return {
                "path": str(path),
                "content": content,
                "size": file_size,
                "encoding": encoding,
                "success": True,
            }

        except Exception as e:
            return {
                "path": params["path"],
                "success": False,
                "error": str(e),
            }

    async def _copy_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Copy a file."""
        try:
            source = Path(params["source"])
            destination = Path(params["destination"])
            preserve_permissions = params.get("preserve_permissions", True)
            create_parents = params.get("create_parents", True)

            if not source.exists():
                raise FileNotFoundError(f"Source file not found: {source}")

            # Create parent directories if needed
            if create_parents and destination.parent != destination:
                destination.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            if preserve_permissions:
                shutil.copy2(source, destination)
            else:
                shutil.copy(source, destination)

            return {
                "source": str(source),
                "destination": str(destination),
                "size": destination.stat().st_size,
                "success": True,
            }

        except Exception as e:
            return {
                "source": params["source"],
                "destination": params["destination"],
                "success": False,
                "error": str(e),
            }

    async def _move_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Move or rename a file."""
        try:
            source = Path(params["source"])
            destination = Path(params["destination"])
            create_parents = params.get("create_parents", True)

            if not source.exists():
                raise FileNotFoundError(f"Source file not found: {source}")

            # Create parent directories if needed
            if create_parents and destination.parent != destination:
                destination.parent.mkdir(parents=True, exist_ok=True)

            # Move file
            shutil.move(str(source), str(destination))

            return {
                "source": str(source),
                "destination": str(destination),
                "success": True,
            }

        except Exception as e:
            return {
                "source": params["source"],
                "destination": params["destination"],
                "success": False,
                "error": str(e),
            }

    async def _delete_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a file."""
        try:
            path = Path(params["path"])

            if not path.exists():
                return {
                    "path": str(path),
                    "deleted": False,
                    "success": True,
                    "message": "File already does not exist",
                }

            if path.is_file():
                path.unlink()
            else:
                raise IsADirectoryError(f"Path is a directory, not a file: {path}")

            return {
                "path": str(path),
                "deleted": True,
                "success": True,
            }

        except Exception as e:
            return {
                "path": params["path"],
                "deleted": False,
                "success": False,
                "error": str(e),
            }

    async def _get_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get file or directory information."""
        try:
            path = Path(params["path"])

            if not path.exists():
                raise FileNotFoundError(f"Path not found: {path}")

            info = self._get_file_info(path)

            return {
                "path": str(path),
                "info": info,
                "success": True,
            }

        except Exception as e:
            return {
                "path": params["path"],
                "success": False,
                "error": str(e),
            }

    async def _set_permissions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set file or directory permissions."""
        try:
            path = Path(params["path"])
            mode_str = params["mode"]
            recursive = params.get("recursive", False)

            if not path.exists():
                raise FileNotFoundError(f"Path not found: {path}")

            mode = int(mode_str, 8)

            if recursive and path.is_dir():
                # Set permissions recursively
                for item in path.rglob("*"):
                    item.chmod(mode)

            path.chmod(mode)

            return {
                "path": str(path),
                "mode": mode_str,
                "recursive": recursive,
                "success": True,
            }

        except Exception as e:
            return {
                "path": params["path"],
                "success": False,
                "error": str(e),
            }

    def _get_file_info(self, path: Path) -> Dict[str, Any]:
        """Get detailed information about a file or directory."""
        try:
            stat_info = path.stat()

            return {
                "name": path.name,
                "path": str(path),
                "type": "directory" if path.is_dir() else "file",
                "size": stat_info.st_size,
                "mode": oct(stat_info.st_mode)[-3:],
                "created": stat_info.st_ctime,
                "modified": stat_info.st_mtime,
                "accessed": stat_info.st_atime,
                "permissions": {
                    "readable": os.access(path, os.R_OK),
                    "writable": os.access(path, os.W_OK),
                    "executable": os.access(path, os.X_OK),
                },
            }

        except Exception as e:
            return {
                "name": path.name,
                "path": str(path),
                "error": str(e),
            }

    async def _execute_rollback(self, execution_id: str) -> Dict[str, Any]:
        """Execute rollback operation."""
        # File system rollback would involve restoring from backup
        # This is a simplified implementation
        return {
            "rollback_type": "filesystem_operation",
            "execution_id": execution_id,
            "message": "File system rollback requires backup restoration",
        }

    async def _get_supported_actions(self) -> List[str]:
        """Get list of supported actions."""
        return [
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
