"""
Tests for directory utilities.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.directories import (
    _create_secure_temp_directory,
    _get_platform_specific_directory,
    _test_directory_writable,
    get_secure_app_directory,
    get_secure_cache_directory,
    get_secure_config_directory,
)


class TestGetSecureAppDirectory:
    """Test secure app directory functionality."""

    def test_basic_usage(self):
        """Test basic directory creation."""
        result = get_secure_app_directory("test_app")
        assert result.exists()
        assert result.is_dir()
        assert "test_app" in str(result)

    def test_with_subdirectory(self):
        """Test directory creation with subdirectory."""
        result = get_secure_app_directory("test_app", "logs")
        assert result.exists()
        assert result.is_dir()
        assert "test_app" in str(result)
        assert "logs" in str(result)

    def test_custom_temp_naming(self):
        """Test custom temporary directory naming."""
        # Force fallback to temp directory by using invalid app name
        with patch(
            "src.utils.directories._get_platform_specific_directory"
        ) as mock_platform:
            mock_platform.return_value = Path("/invalid/path/that/cannot/be/created")

            result = get_secure_app_directory(
                "test_app", temp_prefix="custom_", temp_suffix="_test"
            )

            assert result.exists()
            assert result.is_dir()
            assert "custom_" in result.name
            assert "_test" in result.name

    @patch("os.name", "nt")
    @patch.dict(os.environ, {"LOCALAPPDATA": "/test/localappdata"})
    def test_windows_directory(self):
        """Test Windows directory logic."""
        result = _get_platform_specific_directory("test_app")
        assert result == Path("/test/localappdata/test_app")

    @patch("os.name", "posix")
    @patch.dict(os.environ, {"XDG_DATA_HOME": "/test/xdg"})
    def test_unix_xdg_directory(self):
        """Test Unix XDG directory logic."""
        result = _get_platform_specific_directory("test_app")
        assert result == Path("/test/xdg/test_app")

    @patch("os.name", "posix")
    @patch.dict(os.environ, {}, clear=True)
    def test_unix_default_directory(self):
        """Test Unix default directory logic."""
        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = Path("/home/testuser")
            result = _get_platform_specific_directory("test_app")
            assert result == Path("/home/testuser/.local/share/test_app")


class TestDirectoryWritable:
    """Test directory write permission testing."""

    def test_writable_directory(self, temp_dir):
        """Test with writable directory."""
        # Should not raise an exception
        _test_directory_writable(temp_dir)

    def test_non_writable_directory(self):
        """Test with non-writable directory."""
        with patch("pathlib.Path.touch") as mock_touch:
            mock_touch.side_effect = PermissionError("Permission denied")

            with pytest.raises(PermissionError):
                _test_directory_writable(Path("/some/path"))


class TestCreateSecureTempDirectory:
    """Test secure temporary directory creation."""

    def test_temp_directory_creation(self):
        """Test temporary directory is created with correct permissions."""
        result = _create_secure_temp_directory("test_", "_suffix")

        assert result.exists()
        assert result.is_dir()
        assert "test_" in result.name
        assert "_suffix" in result.name

        # Check permissions (Unix-like systems only)
        if os.name != "nt":
            stat_info = result.stat()
            permissions = stat_info.st_mode & 0o777
            assert permissions == 0o700


class TestSecureCacheDirectory:
    """Test cache directory functionality."""

    def test_cache_directory_creation(self):
        """Test cache directory creation."""
        result = get_secure_cache_directory("test_app")
        assert result.exists()
        assert result.is_dir()
        assert "test_app" in str(result)

    @patch("os.name", "nt")
    @patch.dict(os.environ, {"LOCALAPPDATA": "/test/localappdata"})
    def test_windows_cache_directory(self):
        """Test Windows cache directory path."""
        with patch("src.utils.directories.get_secure_app_directory") as mock_get_dir:
            mock_get_dir.return_value = Path("/test/path")

            get_secure_cache_directory("test_app")

            mock_get_dir.assert_called_once_with(
                "test_app",
                subdirectory="cache",
                temp_prefix="test_app_cache_",
                temp_suffix="_tmp",
            )

    @patch("os.name", "posix")
    @patch.dict(os.environ, {"XDG_CACHE_HOME": "/test/cache"})
    def test_unix_cache_directory(self):
        """Test Unix cache directory with XDG."""
        with patch("src.utils.directories.get_secure_app_directory") as mock_get_dir:
            mock_get_dir.return_value = Path("/test/path")

            get_secure_cache_directory("test_app")

            mock_get_dir.assert_called_once_with(
                "test_app",
                subdirectory=None,
                temp_prefix="test_app_cache_",
                temp_suffix="_tmp",
            )


class TestSecureConfigDirectory:
    """Test config directory functionality."""

    def test_config_directory_creation(self):
        """Test config directory creation."""
        result = get_secure_config_directory("test_app")
        assert result.exists()
        assert result.is_dir()
        assert "test_app" in str(result)

    @patch("os.name", "nt")
    @patch.dict(os.environ, {"APPDATA": "/test/appdata"})
    def test_windows_config_directory(self):
        """Test Windows config directory path."""
        with (
            patch("src.utils.directories._test_directory_writable"),
            patch("pathlib.Path.mkdir"),
        ):

            result = get_secure_config_directory("test_app")
            # Should use APPDATA path construction logic
            assert "test_app" in str(result)

    @patch("os.name", "posix")
    @patch.dict(os.environ, {"XDG_CONFIG_HOME": "/test/config"})
    def test_unix_config_directory(self):
        """Test Unix config directory with XDG."""
        with (
            patch("src.utils.directories._test_directory_writable"),
            patch("pathlib.Path.mkdir"),
        ):

            result = get_secure_config_directory("test_app")
            assert "test_app" in str(result)

    def test_config_directory_fallback(self):
        """Test config directory fallback to temp."""
        with (
            patch("pathlib.Path.mkdir", side_effect=PermissionError),
            patch("src.utils.directories._create_secure_temp_directory") as mock_temp,
        ):
            mock_temp.return_value = Path("/temp/fallback")

            result = get_secure_config_directory("test_app")

            mock_temp.assert_called_once_with("test_app_config_", "_tmp")
            assert result == Path("/temp/fallback")


class TestIntegration:
    """Integration tests for directory utilities."""

    def test_log_manager_integration(self):
        """Test integration with LogManager."""
        from src.logging_utils.log_manager import LogManager

        # Should not raise any exceptions
        log_manager = LogManager()
        assert log_manager.log_dir.exists()
        assert log_manager.log_dir.is_dir()

    def test_cli_integration(self):
        """Test CLI LOG_DIR creation."""
        from src.utils.directories import get_secure_app_directory

        log_dir = get_secure_app_directory("orcastrate", "logs")
        assert log_dir.exists()
        assert log_dir.is_dir()
        assert "orcastrate" in str(log_dir)
        assert "logs" in str(log_dir)
