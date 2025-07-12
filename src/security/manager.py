"""
Security Manager - Centralized security validation and enforcement.

This module implements comprehensive security controls for the system,
including input validation, access control, and security policy enforcement.
"""

import logging
import re
import secrets
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from ..agent.base import Plan


class SecurityLevel(Enum):
    """Security levels for operations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityPolicy(BaseModel):
    """Security policy configuration."""

    name: str
    description: str
    level: SecurityLevel
    rules: List[Dict[str, Any]] = Field(default_factory=list)
    exceptions: List[str] = Field(default_factory=list)
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class SecurityResult(BaseModel):
    """Result of security validation."""

    valid: bool
    level: SecurityLevel
    violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    risk_score: float = 0.0  # 0-1 scale
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AccessControl:
    """Access control system."""

    def __init__(self):
        self.permissions: Dict[str, Set[str]] = {}
        self.roles: Dict[str, Set[str]] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}

    def add_permission(self, resource: str, permission: str) -> None:
        """Add a permission to a resource."""
        if resource not in self.permissions:
            self.permissions[resource] = set()
        self.permissions[resource].add(permission)

    def create_role(self, role: str, permissions: List[str]) -> None:
        """Create a role with permissions."""
        self.roles[role] = set(permissions)

    def assign_role(self, user: str, role: str) -> None:
        """Assign a role to a user."""
        if user not in self.user_roles:
            self.user_roles[user] = set()
        self.user_roles[user].add(role)

    def check_permission(self, user: str, resource: str, permission: str) -> bool:
        """Check if user has permission for resource."""
        user_permissions = self._get_user_permissions(user)
        required_permission = f"{resource}:{permission}"
        return required_permission in user_permissions or "*:*" in user_permissions

    def _get_user_permissions(self, user: str) -> Set[str]:
        """Get all permissions for a user."""
        permissions = set()
        user_roles = self.user_roles.get(user, set())

        for role in user_roles:
            role_permissions = self.roles.get(role, set())
            permissions.update(role_permissions)

        return permissions

    def generate_api_key(
        self, user: str, scopes: List[str], expires_in: int = 3600
    ) -> str:
        """Generate an API key for a user."""
        key = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

        self.api_keys[key] = {
            "user": user,
            "scopes": scopes,
            "expires_at": expires_at,
            "created_at": datetime.utcnow(),
        }

        return key

    def validate_api_key(self, key: str, required_scope: str) -> bool:
        """Validate an API key and check scope."""
        if key not in self.api_keys:
            return False

        key_info = self.api_keys[key]

        # Check expiration
        if key_info["expires_at"] < datetime.utcnow():
            return False

        # Check scope
        scopes = key_info["scopes"]
        return required_scope in scopes or "*" in scopes


class InputValidator:
    """Input validation and sanitization."""

    def __init__(self):
        # Dangerous patterns that should be blocked
        self.dangerous_patterns = [
            r"rm\s+.*-.*rf|[;&|`$\(\)]",  # Command injection
            r"<script|javascript:|data:",  # XSS
            r"union\s+select|drop\s+table|'\s*;.*drop",  # SQL injection
            r"\.\./",  # Path traversal
            r"eval\(|exec\(",  # Code execution
            r"'\s*;\s*(drop|delete|truncate|alter)\s+",  # SQL injection variations
            r"--\s*[^\r\n]*",  # SQL comments
            r"/\*.*?\*/",  # Multi-line SQL comments
            r"\bxp_cmdshell|sp_configure",  # SQL Server dangerous procedures
        ]

        # Resource limits
        self.max_string_length = 10000
        self.max_list_length = 1000
        self.max_dict_depth = 10

    def validate_input(self, data: Any, context: str = "general") -> SecurityResult:
        """Validate input data for security issues."""
        violations = []
        warnings = []
        risk_score = 0.0

        try:
            # Validate based on data type
            if isinstance(data, str):
                result = self._validate_string(data, context)
                violations.extend(result.violations)
                warnings.extend(result.warnings)
                risk_score = max(risk_score, result.risk_score)

            elif isinstance(data, dict):
                result = self._validate_dict(data, context)
                violations.extend(result.violations)
                warnings.extend(result.warnings)
                risk_score = max(risk_score, result.risk_score)

            elif isinstance(data, list):
                result = self._validate_list(data, context)
                violations.extend(result.violations)
                warnings.extend(result.warnings)
                risk_score = max(risk_score, result.risk_score)

            # Determine security level
            if risk_score >= 0.8:
                level = SecurityLevel.CRITICAL
            elif risk_score >= 0.6:
                level = SecurityLevel.HIGH
            elif risk_score >= 0.4:
                level = SecurityLevel.MEDIUM
            else:
                level = SecurityLevel.LOW

            return SecurityResult(
                valid=len(violations) == 0,
                level=level,
                violations=violations,
                warnings=warnings,
                risk_score=risk_score,
            )

        except Exception as e:
            return SecurityResult(
                valid=False,
                level=SecurityLevel.CRITICAL,
                violations=[f"Validation error: {str(e)}"],
                risk_score=1.0,
            )

    def _validate_string(self, value: str, context: str) -> SecurityResult:
        """Validate string input."""
        violations: List[str] = []
        warnings: List[str] = []
        risk_score = 0.0

        # Check length
        if len(value) > self.max_string_length:
            violations.append(
                f"String too long: {len(value)} > {self.max_string_length}"
            )
            risk_score = max(risk_score, 0.3)

        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                violations.append(f"Dangerous pattern detected: {pattern}")
                risk_score = max(risk_score, 0.8)

        # Context-specific validation
        if context == "filename":
            if re.search(r'[<>:"|?*]', value):
                violations.append("Invalid filename characters")
                risk_score = max(risk_score, 0.5)

        elif context == "command":
            violations.append("Command execution not allowed")
            risk_score = 1.0

        # Determine security level
        if risk_score >= 0.8:
            level = SecurityLevel.CRITICAL
        elif risk_score >= 0.6:
            level = SecurityLevel.HIGH
        elif risk_score >= 0.4:
            level = SecurityLevel.MEDIUM
        else:
            level = SecurityLevel.LOW

        return SecurityResult(
            valid=len(violations) == 0,
            level=level,
            violations=violations,
            warnings=warnings,
            risk_score=risk_score,
        )

    def _validate_dict(
        self, value: dict, context: str, depth: int = 0
    ) -> SecurityResult:
        """Validate dictionary input."""
        violations = []
        warnings = []
        risk_score = 0.0

        # Check depth
        if depth > self.max_dict_depth:
            violations.append(f"Dictionary too deep: {depth} > {self.max_dict_depth}")
            risk_score = max(risk_score, 0.4)
            return SecurityResult(
                valid=False,
                level=SecurityLevel.MEDIUM,
                violations=violations,
                risk_score=risk_score,
            )

        # Validate keys and values
        for key, val in value.items():
            # Validate key
            if isinstance(key, str):
                key_result = self._validate_string(key, "dict_key")
                violations.extend(key_result.violations)
                warnings.extend(key_result.warnings)
                risk_score = max(risk_score, key_result.risk_score)

            # Validate value recursively
            if isinstance(val, dict):
                val_result = self._validate_dict(val, context, depth + 1)
            elif isinstance(val, list):
                val_result = self._validate_list(val, context, depth + 1)
            elif isinstance(val, str):
                val_result = self._validate_string(val, context)
            else:
                continue

            violations.extend(val_result.violations)
            warnings.extend(val_result.warnings)
            risk_score = max(risk_score, val_result.risk_score)

        # Determine security level
        if risk_score >= 0.8:
            level = SecurityLevel.CRITICAL
        elif risk_score >= 0.6:
            level = SecurityLevel.HIGH
        elif risk_score >= 0.4:
            level = SecurityLevel.MEDIUM
        else:
            level = SecurityLevel.LOW

        return SecurityResult(
            valid=len(violations) == 0,
            level=level,
            violations=violations,
            warnings=warnings,
            risk_score=risk_score,
        )

    def _validate_list(
        self, value: list, context: str, depth: int = 0
    ) -> SecurityResult:
        """Validate list input."""
        violations = []
        warnings = []
        risk_score = 0.0

        # Check length
        if len(value) > self.max_list_length:
            violations.append(f"List too long: {len(value)} > {self.max_list_length}")
            risk_score = max(risk_score, 0.3)

        # Validate items
        for item in value:
            if isinstance(item, dict):
                item_result = self._validate_dict(item, context, depth)
            elif isinstance(item, list):
                item_result = self._validate_list(item, context, depth + 1)
            elif isinstance(item, str):
                item_result = self._validate_string(item, context)
            else:
                continue

            violations.extend(item_result.violations)
            warnings.extend(item_result.warnings)
            risk_score = max(risk_score, item_result.risk_score)

        # Determine security level
        if risk_score >= 0.8:
            level = SecurityLevel.CRITICAL
        elif risk_score >= 0.6:
            level = SecurityLevel.HIGH
        elif risk_score >= 0.4:
            level = SecurityLevel.MEDIUM
        else:
            level = SecurityLevel.LOW

        return SecurityResult(
            valid=len(violations) == 0,
            level=level,
            violations=violations,
            warnings=warnings,
            risk_score=risk_score,
        )


class SecurityManager:
    """
    Centralized security management system.

    This class provides comprehensive security services including
    input validation, access control, and security policy enforcement.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.access_control = AccessControl()
        self.input_validator = InputValidator()
        self.policies: Dict[str, SecurityPolicy] = {}
        self.security_events: List[Dict[str, Any]] = []

        # Security settings
        self.enabled = config.get("enabled", True)
        self.strict_mode = config.get("strict_mode", False)
        self.audit_enabled = config.get("audit_enabled", True)

    async def initialize(self) -> None:
        """Initialize the security manager."""
        self.logger.info("Initializing Security Manager")

        try:
            await self._load_security_policies()
            await self._setup_default_access_control()

            self.logger.info("Security Manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Security Manager: {e}")
            raise SecurityError(f"Initialization failed: {e}")

    async def validate_plan(self, plan: Plan) -> bool:
        """Validate a plan for security compliance."""
        if not self.enabled:
            return True

        try:
            self.logger.info(f"Validating plan security: {plan.id}")

            # Validate plan structure
            plan_result = self.input_validator.validate_input(plan.model_dump(), "plan")
            if not plan_result.valid:
                self.logger.warning(f"Plan validation failed: {plan_result.violations}")
                if self.strict_mode:
                    return False

            # Validate individual steps
            for step in plan.steps:
                step_result = await self._validate_step(step)
                if not step_result.valid:
                    self.logger.warning(
                        f"Step validation failed: {step_result.violations}"
                    )
                    if self.strict_mode:
                        return False

            # Apply security policies
            policy_result = await self._apply_security_policies(plan)
            if not policy_result.valid:
                self.logger.warning(
                    f"Policy validation failed: {policy_result.violations}"
                )
                if self.strict_mode:
                    return False

            # Log security event
            if self.audit_enabled:
                await self._log_security_event(
                    "plan_validation",
                    {
                        "plan_id": plan.id,
                        "result": "passed",
                        "violations": plan_result.violations,
                        "warnings": plan_result.warnings,
                    },
                )

            return True

        except Exception as e:
            self.logger.error(f"Plan validation error: {e}")

            if self.audit_enabled:
                await self._log_security_event(
                    "plan_validation",
                    {"plan_id": plan.id, "result": "error", "error": str(e)},
                )

            return False

    async def validate_operation(
        self, operation: str, params: Dict[str, Any], user: str
    ) -> SecurityResult:
        """Validate an operation for security compliance."""
        if not self.enabled:
            return SecurityResult(valid=True, level=SecurityLevel.LOW)

        try:
            violations = []
            warnings = []
            risk_score = 0.0

            # Check access control
            if not self.access_control.check_permission(user, operation, "execute"):
                violations.append(f"Access denied for operation: {operation}")
                risk_score = 1.0

            # Validate parameters
            param_result = self.input_validator.validate_input(params, "operation")
            violations.extend(param_result.violations)
            warnings.extend(param_result.warnings)
            risk_score = max(risk_score, param_result.risk_score)

            # Check operation-specific security
            op_result = await self._validate_operation_specific(operation, params)
            violations.extend(op_result.violations)
            warnings.extend(op_result.warnings)
            risk_score = max(risk_score, op_result.risk_score)

            # Determine security level
            if risk_score >= 0.8:
                level = SecurityLevel.CRITICAL
            elif risk_score >= 0.6:
                level = SecurityLevel.HIGH
            elif risk_score >= 0.4:
                level = SecurityLevel.MEDIUM
            else:
                level = SecurityLevel.LOW

            result = SecurityResult(
                valid=len(violations) == 0,
                level=level,
                violations=violations,
                warnings=warnings,
                risk_score=risk_score,
            )

            # Log security event
            if self.audit_enabled:
                await self._log_security_event(
                    "operation_validation",
                    {
                        "operation": operation,
                        "user": user,
                        "result": "passed" if result.valid else "failed",
                        "violations": violations,
                        "warnings": warnings,
                        "risk_score": risk_score,
                    },
                )

            return result

        except Exception as e:
            self.logger.error(f"Operation validation error: {e}")

            if self.audit_enabled:
                await self._log_security_event(
                    "operation_validation",
                    {
                        "operation": operation,
                        "user": user,
                        "result": "error",
                        "error": str(e),
                    },
                )

            return SecurityResult(
                valid=False,
                level=SecurityLevel.CRITICAL,
                violations=[f"Validation error: {str(e)}"],
                risk_score=1.0,
            )

    async def _validate_step(self, step: Dict[str, Any]) -> SecurityResult:
        """Validate a single step."""
        violations = []
        warnings = []
        risk_score = 0.0

        # Validate step parameters
        param_result = self.input_validator.validate_input(
            step.get("parameters", {}), "step"
        )
        violations.extend(param_result.violations)
        warnings.extend(param_result.warnings)
        risk_score = max(risk_score, param_result.risk_score)

        # Check tool security
        tool_name = step.get("tool", "")
        if tool_name in self.config.get("blacklisted_tools", []):
            violations.append(f"Blacklisted tool: {tool_name}")
            risk_score = 1.0

        # Check action security
        action = step.get("action", "")
        if action in self.config.get("blacklisted_actions", []):
            violations.append(f"Blacklisted action: {action}")
            risk_score = 1.0

        # Determine security level
        if risk_score >= 0.8:
            level = SecurityLevel.CRITICAL
        elif risk_score >= 0.6:
            level = SecurityLevel.HIGH
        elif risk_score >= 0.4:
            level = SecurityLevel.MEDIUM
        else:
            level = SecurityLevel.LOW

        return SecurityResult(
            valid=len(violations) == 0,
            level=level,
            violations=violations,
            warnings=warnings,
            risk_score=risk_score,
        )

    async def _apply_security_policies(self, plan: Plan) -> SecurityResult:
        """Apply security policies to a plan."""
        violations = []
        warnings = []
        risk_score = 0.0

        for policy_name, policy in self.policies.items():
            if not policy.enabled:
                continue

            # Check policy expiration
            if policy.expires_at and policy.expires_at < datetime.utcnow():
                continue

            # Apply policy rules
            for rule in policy.rules:
                result = await self._apply_policy_rule(rule, plan)
                if not result.valid:
                    violations.extend(result.violations)
                    warnings.extend(result.warnings)
                    risk_score = max(risk_score, result.risk_score)

        # Determine security level
        if risk_score >= 0.8:
            level = SecurityLevel.CRITICAL
        elif risk_score >= 0.6:
            level = SecurityLevel.HIGH
        elif risk_score >= 0.4:
            level = SecurityLevel.MEDIUM
        else:
            level = SecurityLevel.LOW

        return SecurityResult(
            valid=len(violations) == 0,
            level=level,
            violations=violations,
            warnings=warnings,
            risk_score=risk_score,
        )

    async def _apply_policy_rule(
        self, rule: Dict[str, Any], plan: Plan
    ) -> SecurityResult:
        """Apply a single policy rule."""
        # TODO: Implement policy rule application
        return SecurityResult(valid=True, level=SecurityLevel.LOW)

    async def _validate_operation_specific(
        self, operation: str, params: Dict[str, Any]
    ) -> SecurityResult:
        """Validate operation-specific security requirements."""
        violations = []
        warnings = []
        risk_score = 0.0

        # Cloud operations
        if operation.startswith("cloud_"):
            # Check for sensitive cloud operations
            if operation in ["cloud_delete", "cloud_destroy"]:
                warnings.append("Destructive cloud operation detected")
                risk_score = max(risk_score, 0.6)

        # Database operations
        elif operation.startswith("db_") or operation in [
            "postgresql",
            "mysql",
            "mongodb",
            "redis",
        ]:
            db_result = await self._validate_database_operation(operation, params)
            violations.extend(db_result.violations)
            warnings.extend(db_result.warnings)
            risk_score = max(risk_score, db_result.risk_score)

        # File operations
        elif operation.startswith("file_"):
            # Check file paths
            file_path = params.get("path", "")
            if ".." in file_path or file_path.startswith("/"):
                violations.append("Unsafe file path detected")
                risk_score = 0.8

        # Determine security level
        if risk_score >= 0.8:
            level = SecurityLevel.CRITICAL
        elif risk_score >= 0.6:
            level = SecurityLevel.HIGH
        elif risk_score >= 0.4:
            level = SecurityLevel.MEDIUM
        else:
            level = SecurityLevel.LOW

        return SecurityResult(
            valid=len(violations) == 0,
            level=level,
            violations=violations,
            warnings=warnings,
            risk_score=risk_score,
        )

    async def _validate_database_operation(
        self, operation: str, params: Dict[str, Any]
    ) -> SecurityResult:
        """Comprehensive database security validation."""
        violations: List[str] = []
        warnings: List[str] = []
        risk_score = 0.0

        # Database connection security
        connection_result = self._validate_database_connection(params)
        violations.extend(connection_result.violations)
        warnings.extend(connection_result.warnings)
        risk_score = max(risk_score, connection_result.risk_score)

        # SQL injection and query security
        query_result = self._validate_database_queries(params)
        violations.extend(query_result.violations)
        warnings.extend(query_result.warnings)
        risk_score = max(risk_score, query_result.risk_score)

        # Database action security
        action_result = self._validate_database_actions(operation, params)
        violations.extend(action_result.violations)
        warnings.extend(action_result.warnings)
        risk_score = max(risk_score, action_result.risk_score)

        # Credential security
        credential_result = self._validate_database_credentials(params)
        violations.extend(credential_result.violations)
        warnings.extend(credential_result.warnings)
        risk_score = max(risk_score, credential_result.risk_score)

        # Determine security level
        if risk_score >= 0.8:
            level = SecurityLevel.CRITICAL
        elif risk_score >= 0.6:
            level = SecurityLevel.HIGH
        elif risk_score >= 0.4:
            level = SecurityLevel.MEDIUM
        else:
            level = SecurityLevel.LOW

        return SecurityResult(
            valid=len(violations) == 0,
            level=level,
            violations=violations,
            warnings=warnings,
            risk_score=risk_score,
        )

    def _validate_database_connection(self, params: Dict[str, Any]) -> SecurityResult:
        """Validate database connection parameters."""
        violations: List[str] = []
        warnings: List[str] = []
        risk_score = 0.0

        # Check for localhost/private connections only
        host = params.get("host", "")
        if host and not any(
            allowed in host.lower()
            for allowed in ["localhost", "127.0.0.1", "::1", "test-", "dev-", "local-"]
        ):
            warnings.append(f"External database host detected: {host}")
            risk_score = max(risk_score, 0.3)

        # Check for default ports (security through obscurity)
        port = params.get("port", 0)
        default_ports = {
            5432: "PostgreSQL",
            3306: "MySQL",
            27017: "MongoDB",
            6379: "Redis",
        }
        if port in default_ports:
            warnings.append(f"Using default {default_ports[port]} port {port}")
            risk_score = max(risk_score, 0.2)

        # Check SSL/TLS configuration
        ssl_enabled = params.get("ssl_enabled", False)
        if not ssl_enabled and host not in ["localhost", "127.0.0.1"]:
            warnings.append("SSL/TLS not enabled for external database connection")
            risk_score = max(risk_score, 0.4)

        return SecurityResult(
            valid=len(violations) == 0,
            level=SecurityLevel.LOW,
            violations=violations,
            warnings=warnings,
            risk_score=risk_score,
        )

    def _validate_database_queries(self, params: Dict[str, Any]) -> SecurityResult:
        """Validate database queries for SQL injection and dangerous operations."""
        violations: List[str] = []
        warnings: List[str] = []
        risk_score = 0.0

        # SQL injection patterns
        sql_injection_patterns = [
            r"'\s*;\s*(drop|delete|truncate|alter|update)\s+",  # SQL injection
            r"union\s+select",  # Union-based injection
            r"'\s*or\s+'1'\s*=\s*'1",  # Basic injection
            r"--\s*[^\r\n]*",  # SQL comments
            r"/\*.*?\*/",  # Multi-line comments
            r"exec\s*\(",  # Stored procedure execution
            r"sp_\w+",  # SQL Server stored procedures
            r"\bxp_\w+",  # Extended stored procedures
        ]

        # Check query parameter
        query = params.get("query", "")
        if query:
            for pattern in sql_injection_patterns:
                if re.search(pattern, query, re.IGNORECASE | re.DOTALL):
                    violations.append(
                        f"Potential SQL injection pattern detected: {pattern}"
                    )
                    risk_score = max(risk_score, 0.9)

        # Check for dangerous SQL operations
        dangerous_operations = [
            r"\bdrop\s+(table|database|schema|index)",
            r"\btruncate\s+table",
            r"\bdelete\s+from\s+\w+\s*;?\s*$",  # Delete without WHERE
            r"\bupdate\s+\w+\s+set\s+.*\s*;?\s*$",  # Update without WHERE
            r"\balter\s+(table|database)",
            r"\bgrant\s+",
            r"\brevoke\s+",
            r"\bcreate\s+user",
            r"\bdrop\s+user",
        ]

        for pattern in dangerous_operations:
            if re.search(pattern, query, re.IGNORECASE):
                violations.append(f"Dangerous SQL operation detected: {pattern}")
                risk_score = max(risk_score, 0.8)

        # Check for bulk operations without limits
        bulk_patterns = [
            r"\bselect\s+.*\bfrom\s+\w+\s*;?\s*$",  # Select without LIMIT
            r"\binsert\s+into\s+.*\bvalues\s*\(",  # Bulk inserts
        ]

        for pattern in bulk_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                if "limit" not in query.lower() and "top" not in query.lower():
                    warnings.append("Query without LIMIT clause detected")
                    risk_score = max(risk_score, 0.3)

        return SecurityResult(
            valid=len(violations) == 0,
            level=SecurityLevel.MEDIUM if violations else SecurityLevel.LOW,
            violations=violations,
            warnings=warnings,
            risk_score=risk_score,
        )

    def _validate_database_actions(
        self, operation: str, params: Dict[str, Any]
    ) -> SecurityResult:
        """Validate specific database actions."""
        violations: List[str] = []
        warnings: List[str] = []
        risk_score = 0.0

        # Dangerous database actions
        dangerous_actions = [
            "db_drop",  # Legacy format
            "drop_database",
            "drop_table",
            "truncate_table",
            "delete_all",
            "drop_user",
            "grant_superuser",
            "backup_database",
            "restore_database",
        ]

        action = params.get("action", operation)
        if action in dangerous_actions:
            violations.append(f"Dangerous database action blocked: {action}")
            risk_score = 1.0

        # Privileged operations that need warnings
        privileged_actions = [
            "create_user",
            "grant_permissions",
            "create_database",
            "alter_table",
            "create_index",
            "analyze_performance",
        ]

        if action in privileged_actions:
            warnings.append(f"Privileged database operation: {action}")
            risk_score = max(risk_score, 0.5)

        # Check for mass operations
        if "many" in action or "bulk" in action:
            warnings.append("Bulk database operation detected")
            risk_score = max(risk_score, 0.3)

        # Check table/collection names for suspicious patterns
        table_name = params.get("table_name") or params.get("collection")
        if table_name:
            suspicious_patterns = [
                r"^(admin|root|system|config)$",  # System tables
                r"^(user|password|secret|key)s?$",  # Sensitive data tables
                r"[^a-zA-Z0-9_]",  # Non-standard characters
            ]

            for pattern in suspicious_patterns:
                if re.search(pattern, table_name, re.IGNORECASE):
                    warnings.append(
                        f"Potentially sensitive table/collection: {table_name}"
                    )
                    risk_score = max(risk_score, 0.4)

        return SecurityResult(
            valid=len(violations) == 0,
            level=SecurityLevel.HIGH if violations else SecurityLevel.LOW,
            violations=violations,
            warnings=warnings,
            risk_score=risk_score,
        )

    def _validate_database_credentials(self, params: Dict[str, Any]) -> SecurityResult:
        """Validate database credentials and authentication."""
        violations: List[str] = []
        warnings: List[str] = []
        risk_score = 0.0

        # Check for weak credentials
        username = params.get("username", "")
        password = params.get("password", "")

        # Dangerous usernames
        dangerous_usernames = ["root", "admin", "sa", "postgres", "mysql", "oracle"]
        if username.lower() in dangerous_usernames:
            warnings.append(f"Using privileged username: {username}")
            risk_score = max(risk_score, 0.5)

        # Empty or weak passwords
        if password:
            if len(password) < 8:
                violations.append("Password too short (minimum 8 characters)")
                risk_score = max(risk_score, 0.7)
            elif password.lower() in ["password", "123456", "admin", username.lower()]:
                violations.append("Weak or common password detected")
                risk_score = max(risk_score, 0.8)
        elif username and not password:
            warnings.append("No password specified for database connection")
            risk_score = max(risk_score, 0.6)

        # Check for hardcoded credentials in parameters
        for key, value in params.items():
            if isinstance(value, str) and any(
                cred_key in key.lower()
                for cred_key in ["password", "pass", "secret", "key", "token"]
            ):
                if len(value) > 0:
                    warnings.append(f"Credentials found in parameters: {key}")
                    risk_score = max(risk_score, 0.4)

        return SecurityResult(
            valid=len(violations) == 0,
            level=SecurityLevel.MEDIUM if violations else SecurityLevel.LOW,
            violations=violations,
            warnings=warnings,
            risk_score=risk_score,
        )

    async def _load_security_policies(self) -> None:
        """Load security policies from configuration."""
        # TODO: Load from configuration files

    async def _setup_default_access_control(self) -> None:
        """Set up default access control rules."""
        # Create default roles
        self.access_control.create_role("admin", ["*:*"])
        self.access_control.create_role(
            "user", ["environment:create", "environment:read"]
        )
        self.access_control.create_role("readonly", ["environment:read"])

    async def _log_security_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a security event."""
        event = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }

        self.security_events.append(event)

        # Keep only last 10000 events
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-10000:]

        # Log to external system if configured
        # TODO: Implement external logging


class SecurityError(Exception):
    """Base exception for security-related errors."""


class AccessDeniedError(SecurityError):
    """Exception raised when access is denied."""


class ValidationError(SecurityError):
    """Exception raised during security validation."""
