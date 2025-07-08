"""
Tests for security manager functionality.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.security.manager import (
    AccessControl,
    AccessDeniedError,
    InputValidator,
    SecurityError,
    SecurityLevel,
    SecurityManager,
    SecurityPolicy,
    SecurityResult,
    ValidationError,
)


class TestSecurityPolicy:
    """Test SecurityPolicy model."""

    def test_security_policy_creation(self):
        """Test security policy creation."""
        policy = SecurityPolicy(
            name="test_policy",
            description="Test security policy",
            level=SecurityLevel.HIGH,
            rules=[{"type": "blacklist", "value": "dangerous_action"}],
            exceptions=["admin_user"],
            enabled=True,
        )

        assert policy.name == "test_policy"
        assert policy.description == "Test security policy"
        assert policy.level == SecurityLevel.HIGH
        assert len(policy.rules) == 1
        assert policy.exceptions == ["admin_user"]
        assert policy.enabled is True
        assert isinstance(policy.created_at, datetime)
        assert policy.expires_at is None

    def test_security_policy_with_expiration(self):
        """Test security policy with expiration."""
        expires_at = datetime.utcnow() + timedelta(days=30)

        policy = SecurityPolicy(
            name="temp_policy",
            description="Temporary policy",
            level=SecurityLevel.MEDIUM,
            expires_at=expires_at,
        )

        assert policy.expires_at == expires_at


class TestSecurityResult:
    """Test SecurityResult model."""

    def test_valid_security_result(self):
        """Test valid security result."""
        result = SecurityResult(
            valid=True,
            level=SecurityLevel.LOW,
            warnings=["Using default configuration"],
            recommendations=["Enable encryption"],
            risk_score=0.2,
            metadata={"scan_time": 0.5},
        )

        assert result.valid is True
        assert result.level == SecurityLevel.LOW
        assert result.violations == []
        assert result.warnings == ["Using default configuration"]
        assert result.recommendations == ["Enable encryption"]
        assert result.risk_score == 0.2
        assert result.metadata == {"scan_time": 0.5}

    def test_invalid_security_result(self):
        """Test invalid security result."""
        result = SecurityResult(
            valid=False,
            level=SecurityLevel.CRITICAL,
            violations=["SQL injection attempt", "Unauthorized access"],
            warnings=["Unusual activity pattern"],
            risk_score=0.9,
        )

        assert result.valid is False
        assert result.level == SecurityLevel.CRITICAL
        assert len(result.violations) == 2
        assert len(result.warnings) == 1
        assert result.risk_score == 0.9


class TestAccessControl:
    """Test AccessControl class."""

    @pytest.fixture
    def access_control(self):
        """Create access control instance for testing."""
        return AccessControl()

    def test_add_permission(self, access_control):
        """Test adding permissions to resources."""
        access_control.add_permission("environment", "create")
        access_control.add_permission("environment", "read")
        access_control.add_permission("database", "read")

        assert "environment" in access_control.permissions
        assert "create" in access_control.permissions["environment"]
        assert "read" in access_control.permissions["environment"]
        assert "database" in access_control.permissions
        assert "read" in access_control.permissions["database"]

    def test_create_role(self, access_control):
        """Test creating roles with permissions."""
        permissions = ["environment:create", "environment:read", "database:read"]
        access_control.create_role("developer", permissions)

        assert "developer" in access_control.roles
        assert access_control.roles["developer"] == set(permissions)

    def test_assign_role(self, access_control):
        """Test assigning roles to users."""
        access_control.create_role("admin", ["*:*"])
        access_control.assign_role("user1", "admin")
        access_control.assign_role("user1", "developer")  # Multiple roles

        assert "user1" in access_control.user_roles
        assert "admin" in access_control.user_roles["user1"]
        assert "developer" in access_control.user_roles["user1"]

    def test_check_permission_with_role(self, access_control):
        """Test permission checking with roles."""
        # Setup
        access_control.create_role(
            "developer", ["environment:create", "environment:read"]
        )
        access_control.assign_role("user1", "developer")

        # Test permissions
        assert access_control.check_permission("user1", "environment", "create") is True
        assert access_control.check_permission("user1", "environment", "read") is True
        assert (
            access_control.check_permission("user1", "environment", "delete") is False
        )
        assert access_control.check_permission("user1", "database", "read") is False

    def test_check_permission_admin(self, access_control):
        """Test admin permissions (wildcard)."""
        access_control.create_role("admin", ["*:*"])
        access_control.assign_role("admin_user", "admin")

        # Admin should have all permissions
        assert (
            access_control.check_permission("admin_user", "environment", "create")
            is True
        )
        assert (
            access_control.check_permission("admin_user", "database", "delete") is True
        )
        assert (
            access_control.check_permission("admin_user", "anything", "everything")
            is True
        )

    def test_check_permission_no_user(self, access_control):
        """Test permission checking for non-existent user."""
        assert (
            access_control.check_permission("nonexistent", "environment", "read")
            is False
        )

    def test_generate_api_key(self, access_control):
        """Test API key generation."""
        key = access_control.generate_api_key(
            "user1", ["environment:read"], expires_in=3600
        )

        assert key is not None
        assert len(key) > 0
        assert key in access_control.api_keys

        key_info = access_control.api_keys[key]
        assert key_info["user"] == "user1"
        assert key_info["scopes"] == ["environment:read"]
        assert isinstance(key_info["expires_at"], datetime)
        assert isinstance(key_info["created_at"], datetime)

    def test_validate_api_key_valid(self, access_control):
        """Test valid API key validation."""
        key = access_control.generate_api_key(
            "user1", ["environment:read"], expires_in=3600
        )

        assert access_control.validate_api_key(key, "environment:read") is True
        assert access_control.validate_api_key(key, "environment:write") is False

    def test_validate_api_key_wildcard_scope(self, access_control):
        """Test API key with wildcard scope."""
        key = access_control.generate_api_key("admin", ["*"], expires_in=3600)

        assert access_control.validate_api_key(key, "environment:read") is True
        assert access_control.validate_api_key(key, "anything:anything") is True

    def test_validate_api_key_expired(self, access_control):
        """Test expired API key validation."""
        key = access_control.generate_api_key(
            "user1", ["environment:read"], expires_in=0
        )

        # Key should be expired immediately
        assert access_control.validate_api_key(key, "environment:read") is False

    def test_validate_api_key_nonexistent(self, access_control):
        """Test non-existent API key validation."""
        assert (
            access_control.validate_api_key("nonexistent_key", "environment:read")
            is False
        )


class TestInputValidator:
    """Test InputValidator class."""

    @pytest.fixture
    def input_validator(self):
        """Create input validator for testing."""
        return InputValidator()

    def test_validate_safe_string(self, input_validator):
        """Test validation of safe string input."""
        result = input_validator.validate_input("safe_string_123", "general")

        assert result.valid is True
        assert len(result.violations) == 0
        assert result.risk_score < 0.4

    def test_validate_dangerous_string(self, input_validator):
        """Test validation of dangerous string input."""
        dangerous_inputs = [
            "rm -rf /",  # Command injection
            "<script>alert('xss')</script>",  # XSS
            "'; DROP TABLE users; --",  # SQL injection
            "../../../etc/passwd",  # Path traversal
            "eval('malicious code')",  # Code execution
        ]

        for dangerous_input in dangerous_inputs:
            result = input_validator.validate_input(dangerous_input, "general")

            assert result.valid is False
            assert len(result.violations) > 0
            assert result.risk_score >= 0.8

    def test_validate_long_string(self, input_validator):
        """Test validation of excessively long string."""
        long_string = "a" * (input_validator.max_string_length + 1)
        result = input_validator.validate_input(long_string, "general")

        assert result.valid is False
        assert any("String too long" in violation for violation in result.violations)
        assert result.risk_score >= 0.3

    def test_validate_filename_context(self, input_validator):
        """Test validation with filename context."""
        # Valid filename
        result = input_validator.validate_input("document.txt", "filename")
        assert result.valid is True

        # Invalid filename characters
        result = input_validator.validate_input("file<>name.txt", "filename")
        assert result.valid is False
        assert any(
            "Invalid filename characters" in violation
            for violation in result.violations
        )

    def test_validate_command_context(self, input_validator):
        """Test validation with command context."""
        result = input_validator.validate_input("ls -la", "command")

        assert result.valid is False
        assert any(
            "Command execution not allowed" in violation
            for violation in result.violations
        )
        assert result.risk_score == 1.0

    def test_validate_dict_input(self, input_validator):
        """Test validation of dictionary input."""
        safe_dict = {"name": "test", "value": 123, "config": {"enabled": True}}

        result = input_validator.validate_input(safe_dict, "general")
        assert result.valid is True

    def test_validate_dict_with_dangerous_values(self, input_validator):
        """Test validation of dictionary with dangerous values."""
        dangerous_dict = {
            "command": "rm -rf /",
            "script": "<script>alert('xss')</script>",
        }

        result = input_validator.validate_input(dangerous_dict, "general")
        assert result.valid is False
        assert len(result.violations) >= 2  # Should catch both dangerous values

    def test_validate_deep_dict(self, input_validator):
        """Test validation of deeply nested dictionary."""
        # Create a dict deeper than max_dict_depth
        deep_dict = {"level1": {"level2": {"level3": {}}}}
        for i in range(input_validator.max_dict_depth):
            deep_dict = {"level": deep_dict}

        result = input_validator.validate_input(deep_dict, "general")
        assert result.valid is False
        assert any(
            "Dictionary too deep" in violation for violation in result.violations
        )

    def test_validate_list_input(self, input_validator):
        """Test validation of list input."""
        safe_list = ["item1", "item2", {"key": "value"}]

        result = input_validator.validate_input(safe_list, "general")
        assert result.valid is True

    def test_validate_long_list(self, input_validator):
        """Test validation of excessively long list."""
        long_list = ["item"] * (input_validator.max_list_length + 1)

        result = input_validator.validate_input(long_list, "general")
        assert result.valid is False
        assert any("List too long" in violation for violation in result.violations)

    def test_validate_list_with_dangerous_items(self, input_validator):
        """Test validation of list with dangerous items."""
        dangerous_list = [
            "safe_item",
            "rm -rf /",
            {"command": "<script>alert('xss')</script>"},
        ]

        result = input_validator.validate_input(dangerous_list, "general")
        assert result.valid is False
        assert len(result.violations) >= 2


class TestSecurityManager:
    """Test SecurityManager class."""

    def test_security_manager_initialization(self, security_config):
        """Test security manager initialization."""
        manager = SecurityManager(security_config)

        assert manager.config == security_config
        assert manager.enabled is True
        assert manager.strict_mode is False
        assert manager.audit_enabled is True
        assert isinstance(manager.access_control, AccessControl)
        assert isinstance(manager.input_validator, InputValidator)
        assert isinstance(manager.policies, dict)
        assert isinstance(manager.security_events, list)

    @pytest.mark.asyncio
    async def test_security_manager_initialize(self, security_manager):
        """Test security manager initialization process."""
        with patch.object(
            security_manager, "_load_security_policies"
        ) as mock_load, patch.object(
            security_manager, "_setup_default_access_control"
        ) as mock_setup:
            await security_manager.initialize()

            mock_load.assert_called_once()
            mock_setup.assert_called_once()

    @pytest.mark.asyncio
    async def test_security_manager_initialize_failure(self, security_config):
        """Test security manager initialization failure."""
        manager = SecurityManager(security_config)

        with patch.object(
            manager, "_load_security_policies", side_effect=Exception("Load failed")
        ):
            with pytest.raises(SecurityError, match="Initialization failed"):
                await manager.initialize()

    @pytest.mark.asyncio
    async def test_validate_plan_success(self, security_manager, sample_plan):
        """Test successful plan validation."""
        await security_manager.initialize()

        with patch.object(
            security_manager, "_validate_step"
        ) as mock_validate_step, patch.object(
            security_manager, "_apply_security_policies"
        ) as mock_apply_policies, patch.object(
            security_manager, "_log_security_event"
        ) as mock_log:
            # Mock all validations to pass
            mock_validate_step.return_value = SecurityResult(
                valid=True, level=SecurityLevel.LOW
            )
            mock_apply_policies.return_value = SecurityResult(
                valid=True, level=SecurityLevel.LOW
            )

            result = await security_manager.validate_plan(sample_plan)

            assert result is True
            mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_plan_disabled(self, security_config, sample_plan):
        """Test plan validation when security is disabled."""
        security_config["enabled"] = False
        manager = SecurityManager(security_config)

        result = await manager.validate_plan(sample_plan)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_plan_step_failure(self, security_manager, sample_plan):
        """Test plan validation with step failure."""
        await security_manager.initialize()

        with patch.object(
            security_manager, "_validate_step"
        ) as mock_validate_step, patch.object(
            security_manager, "_log_security_event"
        ) as mock_log:
            # Mock step validation to fail
            mock_validate_step.return_value = SecurityResult(
                valid=False,
                level=SecurityLevel.HIGH,
                violations=["Dangerous operation detected"],
            )

            result = await security_manager.validate_plan(sample_plan)

            # Should still return True in non-strict mode
            assert result is True
            mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_plan_strict_mode(self, security_config, sample_plan):
        """Test plan validation in strict mode."""
        security_config["strict_mode"] = True
        manager = SecurityManager(security_config)
        await manager.initialize()

        with patch.object(manager, "_validate_step") as mock_validate_step:
            # Mock step validation to fail
            mock_validate_step.return_value = SecurityResult(
                valid=False,
                level=SecurityLevel.HIGH,
                violations=["Dangerous operation detected"],
            )

            result = await manager.validate_plan(sample_plan)

            # Should return False in strict mode
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_operation_success(self, security_manager):
        """Test successful operation validation."""
        await security_manager.initialize()

        # Setup access control
        security_manager.access_control.create_role("user", ["test_op:execute"])
        security_manager.access_control.assign_role("test_user", "user")

        result = await security_manager.validate_operation(
            "test_op", {"param": "safe_value"}, "test_user"
        )

        assert result.valid is True
        assert result.level in [SecurityLevel.LOW, SecurityLevel.MEDIUM]

    @pytest.mark.asyncio
    async def test_validate_operation_access_denied(self, security_manager):
        """Test operation validation with access denied."""
        await security_manager.initialize()

        result = await security_manager.validate_operation(
            "restricted_op", {"param": "value"}, "unauthorized_user"
        )

        assert result.valid is False
        assert result.risk_score == 1.0
        assert any("Access denied" in violation for violation in result.violations)

    @pytest.mark.asyncio
    async def test_validate_operation_dangerous_params(self, security_manager):
        """Test operation validation with dangerous parameters."""
        await security_manager.initialize()

        # Setup access control to allow operation
        security_manager.access_control.create_role("user", ["test_op:execute"])
        security_manager.access_control.assign_role("test_user", "user")

        result = await security_manager.validate_operation(
            "test_op", {"command": "rm -rf /"}, "test_user"  # Dangerous parameter
        )

        assert result.valid is False
        assert result.risk_score >= 0.8

    @pytest.mark.asyncio
    async def test_validate_operation_disabled(self, security_config):
        """Test operation validation when security is disabled."""
        security_config["enabled"] = False
        manager = SecurityManager(security_config)

        result = await manager.validate_operation("any_op", {}, "any_user")

        assert result.valid is True
        assert result.level == SecurityLevel.LOW

    @pytest.mark.asyncio
    async def test_validate_step(self, security_manager):
        """Test individual step validation."""
        await security_manager.initialize()

        # Safe step
        safe_step = {
            "id": "step1",
            "tool": "safe_tool",
            "action": "safe_action",
            "parameters": {"param": "safe_value"},
        }

        result = await security_manager._validate_step(safe_step)
        assert result.valid is True

        # Dangerous step with blacklisted tool
        dangerous_step = {
            "id": "step2",
            "tool": "dangerous_tool",  # In blacklist
            "action": "safe_action",
            "parameters": {},
        }

        result = await security_manager._validate_step(dangerous_step)
        assert result.valid is False
        assert result.risk_score == 1.0

    @pytest.mark.asyncio
    async def test_validate_operation_specific(self, security_manager):
        """Test operation-specific validation."""
        await security_manager.initialize()

        # Cloud delete operation (should trigger warning)
        result = await security_manager._validate_operation_specific("cloud_delete", {})
        assert len(result.warnings) > 0
        assert result.risk_score >= 0.6

        # Database drop operation (should be blocked)
        result = await security_manager._validate_operation_specific("db_drop", {})
        assert result.valid is False
        assert result.risk_score == 1.0

        # File operation with unsafe path
        result = await security_manager._validate_operation_specific(
            "file_read", {"path": "../../../etc/passwd"}
        )
        assert result.valid is False
        assert result.risk_score == 0.8

    @pytest.mark.asyncio
    async def test_log_security_event(self, security_manager):
        """Test security event logging."""
        await security_manager.initialize()

        await security_manager._log_security_event("test_event", {"data": "test"})

        assert len(security_manager.security_events) == 1
        event = security_manager.security_events[0]
        assert event["type"] == "test_event"
        assert event["data"] == {"data": "test"}
        assert "timestamp" in event

    @pytest.mark.asyncio
    async def test_security_event_limit(self, security_manager):
        """Test security event storage limit."""
        await security_manager.initialize()

        # Add more than 10000 events
        for i in range(10005):
            await security_manager._log_security_event(f"event_{i}", {})

        # Should keep only the last 10000
        assert len(security_manager.security_events) == 10000
        assert security_manager.security_events[0]["type"] == "event_5"
        assert security_manager.security_events[-1]["type"] == "event_10004"

    @pytest.mark.asyncio
    async def test_setup_default_access_control(self, security_manager):
        """Test default access control setup."""
        await security_manager._setup_default_access_control()

        # Check default roles
        assert "admin" in security_manager.access_control.roles
        assert "user" in security_manager.access_control.roles
        assert "readonly" in security_manager.access_control.roles

        # Check admin permissions
        admin_permissions = security_manager.access_control.roles["admin"]
        assert "*:*" in admin_permissions


class TestSecurityExceptions:
    """Test security-related exceptions."""

    def test_security_error(self):
        """Test SecurityError exception."""
        error = SecurityError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_access_denied_error(self):
        """Test AccessDeniedError exception."""
        error = AccessDeniedError("Access denied")
        assert str(error) == "Access denied"
        assert isinstance(error, SecurityError)

    def test_validation_error(self):
        """Test ValidationError exception."""
        error = ValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, SecurityError)
