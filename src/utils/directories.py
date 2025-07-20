"""
Secure directory management utilities.

This module provides cross-platform, secure directory creation following
OS-specific standards and security best practices.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional


def get_secure_app_directory(
    app_name: str = "orcastrate",
    subdirectory: Optional[str] = None,
    temp_prefix: Optional[str] = None,
    temp_suffix: Optional[str] = None,
) -> Path:
    """
    Get secure application directory based on platform and user permissions.

    This function follows platform-specific conventions and security best practices:
    - Windows: Uses %LOCALAPPDATA%/app_name
    - Unix/Linux: Follows XDG Base Directory Specification
    - Fallback: Creates secure temporary directory with owner-only permissions

    Args:
        app_name: Name of the application (default: "orcastrate")
        subdirectory: Optional subdirectory within the app directory
        temp_prefix: Prefix for temporary directory name (default: f"{app_name}_")
        temp_suffix: Suffix for temporary directory name (default: "_data")

    Returns:
        Path: Secure, writable directory path

    Examples:
        >>> # Basic usage
        >>> log_dir = get_secure_app_directory("orcastrate", "logs")

        >>> # Custom temporary naming
        >>> cache_dir = get_secure_app_directory(
        ...     "myapp", "cache", temp_prefix="cache_", temp_suffix="_tmp"
        ... )

    Raises:
        OSError: If no suitable directory can be created (rare)

    Security Features:
        - Uses platform-appropriate user-specific directories
        - Tests write permissions before use
        - Falls back to secure temporary directories with 0o700 permissions
        - Follows principle of least privilege
    """
    # Set default temporary naming
    if temp_prefix is None:
        temp_prefix = f"{app_name}_"
    if temp_suffix is None:
        temp_suffix = "_data"

    # Determine base directory path
    base_dir = _get_platform_specific_directory(app_name)

    # Add subdirectory if specified
    if subdirectory:
        app_dir = base_dir / subdirectory
    else:
        app_dir = base_dir

    # Try to use the platform-specific directory
    try:
        app_dir.mkdir(parents=True, exist_ok=True)
        _test_directory_writable(app_dir)
        return app_dir

    except (OSError, PermissionError):
        # Fall back to secure temporary directory
        return _create_secure_temp_directory(temp_prefix, temp_suffix)


def _get_platform_specific_directory(app_name: str) -> Path:
    """Get platform-appropriate application directory."""
    if os.name == "nt":  # Windows
        # Use %LOCALAPPDATA% on Windows
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / app_name
        else:
            # Fallback to temp directory if LOCALAPPDATA not available
            return Path(tempfile.gettempdir()) / app_name

    else:  # Unix-like systems (Linux, macOS, etc.)
        # Follow XDG Base Directory Specification
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        if xdg_data_home:
            return Path(xdg_data_home) / app_name
        else:
            # XDG default: ~/.local/share
            return Path.home() / ".local" / "share" / app_name


def _test_directory_writable(directory: Path) -> None:
    """
    Test if directory is writable by creating and removing a test file.

    Args:
        directory: Directory to test

    Raises:
        OSError: If directory is not writable
        PermissionError: If insufficient permissions
    """
    test_file = directory / ".write_test"
    test_file.touch()
    test_file.unlink()


def _create_secure_temp_directory(prefix: str, suffix: str) -> Path:
    """
    Create a secure temporary directory with owner-only permissions.

    Args:
        prefix: Prefix for directory name
        suffix: Suffix for directory name

    Returns:
        Path: Secure temporary directory with 0o700 permissions
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix, suffix=suffix))
    os.chmod(temp_dir, 0o700)  # Owner read/write/execute only
    return temp_dir


def get_secure_cache_directory(app_name: str = "orcastrate") -> Path:
    """
    Get secure cache directory following platform conventions.

    Args:
        app_name: Name of the application

    Returns:
        Path: Platform-appropriate cache directory
    """
    # Note: This logic is here for documentation, but we use get_secure_app_directory for actual implementation

    # Use the secure directory function for consistent behavior
    return get_secure_app_directory(
        app_name,
        subdirectory="cache" if os.name == "nt" else None,
        temp_prefix=f"{app_name}_cache_",
        temp_suffix="_tmp",
    )


def get_secure_config_directory(app_name: str = "orcastrate") -> Path:
    """
    Get secure configuration directory following platform conventions.

    Args:
        app_name: Name of the application

    Returns:
        Path: Platform-appropriate config directory
    """
    if os.name == "nt":  # Windows
        # Use %APPDATA%/app_name on Windows for config
        app_data = os.environ.get("APPDATA")
        if app_data:
            config_dir = Path(app_data) / app_name
        else:
            # Fallback to LOCALAPPDATA
            local_app_data = os.environ.get("LOCALAPPDATA", tempfile.gettempdir())
            config_dir = Path(local_app_data) / app_name / "config"
    else:  # Unix-like systems
        # Follow XDG: $XDG_CONFIG_HOME or ~/.config
        xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config_home:
            config_dir = Path(xdg_config_home) / app_name
        else:
            config_dir = Path.home() / ".config" / app_name

    # Use the secure directory function for consistent behavior
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        _test_directory_writable(config_dir)
        return config_dir
    except (OSError, PermissionError):
        return _create_secure_temp_directory(f"{app_name}_config_", "_tmp")
