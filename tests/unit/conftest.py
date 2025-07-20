"""
Unit test specific fixtures and configurations.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

import pytest

from src.logging_utils.log_manager import LogManager
from src.logging_utils.progress_tracker import ProgressTracker


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_progress_tracker():
    """Create a mock progress tracker for testing."""
    mock_log_manager = Mock(spec=LogManager)
    mock_log_manager.emit_event = AsyncMock()

    progress_tracker = Mock(spec=ProgressTracker)
    progress_tracker.log_manager = mock_log_manager
    progress_tracker.update_step_progress = Mock()
    progress_tracker.add_step_message = Mock()
    progress_tracker.log_step_success = Mock()
    progress_tracker.log_step_failure = Mock()
    progress_tracker.log_step_conditional = Mock()
    progress_tracker.start_execution_progress = Mock()
    progress_tracker.complete_execution_progress = Mock()
    return progress_tracker


@pytest.fixture
def mock_database_connection():
    """Create a mock database connection for testing."""
    connection = AsyncMock()
    connection.execute = AsyncMock()
    connection.fetchone = AsyncMock()
    connection.fetchall = AsyncMock()
    connection.commit = AsyncMock()
    connection.rollback = AsyncMock()
    connection.close = AsyncMock()
    return connection


@pytest.fixture
def database_config() -> Dict[str, Any]:
    """Create database configuration for testing."""
    return {
        "host": "localhost",
        "port": 5432,
        "database": "test_db",
        "username": "test_user",
        "password": "test_pass",
        "connection_timeout": 30,
        "max_connections": 10,
    }


@pytest.fixture
def redis_config() -> Dict[str, Any]:
    """Create Redis configuration for testing."""
    return {
        "host": "localhost",
        "port": 6379,
        "database": "0",
        "username": "",
        "password": "",
        "connection_timeout": 30,
        "max_connections": 10,
        "database_number": 0,
    }


@pytest.fixture
def mongodb_config() -> Dict[str, Any]:
    """Create MongoDB configuration for testing."""
    return {
        "host": "localhost",
        "port": 27017,
        "database": "test_db",
        "username": "",
        "password": "",
        "connection_timeout": 30,
        "max_connections": 10,
    }


@pytest.fixture
def mysql_config() -> Dict[str, Any]:
    """Create MySQL configuration for testing."""
    return {
        "host": "localhost",
        "port": 3306,
        "database": "test_db",
        "username": "test_user",
        "password": "test_pass",
        "connection_timeout": 30,
        "max_connections": 10,
    }
