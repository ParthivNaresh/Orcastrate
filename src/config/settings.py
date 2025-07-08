"""
Central configuration management for Orcastrate.

This module provides type-safe configuration management using Pydantic,
supporting multiple environments and validation of settings.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

# Type ignore for env parameter used throughout this module
# mypy: disable-error-code=call-arg


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    # PostgreSQL
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="orcastrate", env="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="", env="POSTGRES_PASSWORD")

    # MongoDB
    mongodb_url: str = Field(default="mongodb://localhost:27017", env="MONGODB_URL")
    mongodb_db: str = Field(default="orcastrate", env="MONGODB_DB")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_db: int = Field(default=0, env="REDIS_DB")

    # MySQL
    mysql_host: str = Field(default="localhost", env="MYSQL_HOST")
    mysql_port: int = Field(default=3306, env="MYSQL_PORT")
    mysql_db: str = Field(default="orcastrate", env="MYSQL_DB")
    mysql_user: str = Field(default="root", env="MYSQL_USER")
    mysql_password: str = Field(default="", env="MYSQL_PASSWORD")

    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def mysql_url(self) -> str:
        """Get MySQL connection URL."""
        return f"mysql://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}"

    class Config:
        env_prefix = ""
        case_sensitive = False


class CloudSettings(BaseSettings):
    """Cloud provider configuration."""

    # AWS
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(
        default=None, env="AWS_SECRET_ACCESS_KEY"
    )
    aws_region: str = Field(default="us-west-2", env="AWS_DEFAULT_REGION")
    aws_session_token: Optional[str] = Field(default=None, env="AWS_SESSION_TOKEN")

    # GCP (future)
    gcp_project_id: Optional[str] = Field(default=None, env="GCP_PROJECT_ID")
    gcp_credentials_path: Optional[str] = Field(
        default=None, env="GOOGLE_APPLICATION_CREDENTIALS"
    )
    gcp_region: str = Field(default="us-central1", env="GCP_REGION")

    # Azure (future)
    azure_subscription_id: Optional[str] = Field(
        default=None, env="AZURE_SUBSCRIPTION_ID"
    )
    azure_client_id: Optional[str] = Field(default=None, env="AZURE_CLIENT_ID")
    azure_client_secret: Optional[str] = Field(default=None, env="AZURE_CLIENT_SECRET")
    azure_tenant_id: Optional[str] = Field(default=None, env="AZURE_TENANT_ID")

    model_config = {"env_prefix": "", "case_sensitive": False}


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_default_model: str = Field(default="gpt-4", env="OPENAI_DEFAULT_MODEL")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=4000, env="OPENAI_MAX_TOKENS")
    openai_timeout: int = Field(default=30, env="OPENAI_TIMEOUT")

    # Anthropic
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_default_model: str = Field(
        default="claude-3-sonnet-20240229", env="ANTHROPIC_DEFAULT_MODEL"
    )
    anthropic_temperature: float = Field(default=0.7, env="ANTHROPIC_TEMPERATURE")
    anthropic_max_tokens: int = Field(default=4000, env="ANTHROPIC_MAX_TOKENS")
    anthropic_timeout: int = Field(default=30, env="ANTHROPIC_TIMEOUT")

    # General LLM settings
    llm_retry_attempts: int = Field(default=3, env="LLM_RETRY_ATTEMPTS")
    llm_retry_delay: float = Field(default=1.0, env="LLM_RETRY_DELAY")
    llm_rate_limit_requests_per_minute: int = Field(
        default=60, env="LLM_RATE_LIMIT_RPM"
    )

    @field_validator("openai_temperature", "anthropic_temperature")
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature is between 0 and 2."""
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v

    class Config:
        env_prefix = ""
        case_sensitive = False


class SecuritySettings(BaseSettings):
    """Security configuration."""

    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS",
        json_schema_extra={"env_parse": False},
    )
    max_request_size: int = Field(
        default=16 * 1024 * 1024, env="MAX_REQUEST_SIZE"
    )  # 16MB
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        elif isinstance(v, list):
            return v
        else:
            return []

    model_config = {
        "env_prefix": "",
        "case_sensitive": False,
        "env_parse_none_str": "null",
        "str_strip_whitespace": True,
    }


class TestingSettings(BaseSettings):
    """Testing configuration."""

    # Test database settings
    test_postgres_host: str = Field(default="localhost", env="TEST_POSTGRES_HOST")
    test_postgres_port: int = Field(default=5433, env="TEST_POSTGRES_PORT")
    test_postgres_db: str = Field(default="orcastrate_test", env="TEST_POSTGRES_DB")
    test_postgres_user: str = Field(default="postgres", env="TEST_POSTGRES_USER")
    test_postgres_password: str = Field(default="", env="TEST_POSTGRES_PASSWORD")

    # Live testing
    enable_live_testing: bool = Field(default=False, env="ENABLE_LIVE_TESTING")
    live_test_timeout: int = Field(default=300, env="LIVE_TEST_TIMEOUT")  # seconds

    # LocalStack for AWS testing
    localstack_endpoint: str = Field(
        default="http://localhost:4566", env="LOCALSTACK_ENDPOINT"
    )
    use_localstack: bool = Field(default=True, env="USE_LOCALSTACK")

    @property
    def test_postgres_url(self) -> str:
        """Get test PostgreSQL connection URL."""
        return f"postgresql://{self.test_postgres_user}:{self.test_postgres_password}@{self.test_postgres_host}:{self.test_postgres_port}/{self.test_postgres_db}"

    class Config:
        env_prefix = ""
        case_sensitive = False


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")

    # Metrics
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")

    # Tracing
    enable_tracing: bool = Field(default=False, env="ENABLE_TRACING")
    tracing_endpoint: Optional[str] = Field(default=None, env="TRACING_ENDPOINT")

    # Health checks
    health_check_timeout: int = Field(default=30, env="HEALTH_CHECK_TIMEOUT")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v):
        """Validate log format."""
        if v not in ["json", "text"]:
            raise ValueError("Log format must be 'json' or 'text'")
        return v

    class Config:
        env_prefix = ""
        case_sensitive = False


class AppSettings(BaseSettings):
    """Main application settings."""

    # Core application
    app_name: str = Field(default="Orcastrate", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")

    # Feature flags
    enable_llm_features: bool = Field(default=True, env="ENABLE_LLM_FEATURES")
    enable_multicloud: bool = Field(default=True, env="ENABLE_MULTICLOUD")
    enable_terraform: bool = Field(default=True, env="ENABLE_TERRAFORM")
    enable_kubernetes: bool = Field(default=True, env="ENABLE_KUBERNETES")

    # Planning settings
    max_plan_steps: int = Field(default=100, env="MAX_PLAN_STEPS")
    max_planning_time: int = Field(default=300, env="MAX_PLANNING_TIME")  # seconds
    default_planning_strategy: str = Field(
        default="hybrid_analysis", env="DEFAULT_PLANNING_STRATEGY"
    )

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    cloud: CloudSettings = Field(default_factory=CloudSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    testing: TestingSettings = Field(default_factory=TestingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment name."""
        valid_envs = ["development", "testing", "staging", "production", "docker", "ci"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment in ["testing", "ci"]

    def get_safe_dict(self) -> Dict[str, Any]:
        """Get configuration dict with sensitive values masked."""
        config = self.dict()

        def mask_sensitive(obj, path=""):
            """Recursively mask sensitive fields."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{path}.{key}" if path else key
                    if any(
                        sensitive in key.lower()
                        for sensitive in ["password", "secret", "key", "token"]
                    ):
                        if value and str(value).strip():
                            obj[key] = "***MASKED***"
                    elif isinstance(value, (dict, list)):
                        mask_sensitive(value, full_key)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        mask_sensitive(item, path)

        mask_sensitive(config)
        return config

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton pattern for settings
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Get application settings singleton."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings


def reload_settings() -> AppSettings:
    """Force reload of settings (useful for testing)."""
    global _settings
    _settings = None
    return get_settings()
