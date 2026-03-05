"""Centralized configuration for IdeaAgent using pydantic-settings.

Configuration Loading Mechanism:
================================
1. find_dotenv() searches for .env file from current directory upward
2. SettingsConfigDict reads values from the found .env file
3. Field aliases map .env keys (e.g., OPENAI_API_KEY) to Python attributes (openai_api_key)
4. If a key is missing in .env, the default value defined in Field() is used
5. System environment variables are completely ignored

Example .env file:
    OPENAI_API_KEY=sk-...
    DEFAULT_MODEL=gpt-4o
    MAX_TOKENS=32000

Usage:
    from ideaagent.config import settings, ensure_directories
    
    api_key = settings.openai_api_key
    model = settings.default_model
    ensure_directories()  # Create required directories when needed
"""

from pathlib import Path
from typing import Optional
from dotenv import find_dotenv

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from .env file only.

    Configuration is read exclusively from the .env file.
    Environment variables are ignored to ensure consistent behavior.
    
    All fields can be set in .env file using UPPERCASE names.
    Fields not present in .env will use the default values defined below.
    """

    # ── LLM Configuration ─────────────────────────────────────────────────────
    openai_api_key: Optional[str] = Field(
        default=None,
        alias="OPENAI_API_KEY",
        description="API key for LLM provider. Required for API calls.",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="OPENAI_BASE_URL",
        description="Base URL for the LLM API endpoint.",
    )
    default_model: str = Field(
        default="gpt-4o",
        alias="DEFAULT_MODEL",
        description="Model name to use for completions.",
    )
    max_tokens: int = Field(
        default=65536,
        alias="MAX_TOKENS",
        ge=1,
        le=131072,
        description="Maximum tokens in the response.",
    )
    temperature: float = Field(
        default=0.7,
        alias="TEMPERATURE",
        ge=0.0,
        le=2.0,
        description="Temperature for sampling (0.0 = deterministic, 2.0 = very random).",
    )

    # ── Agent Loop Configuration ──────────────────────────────────────────────
    max_agent_iterations: int = Field(
        default=20,
        alias="MAX_AGENT_ITERATIONS",
        ge=1,
        le=100,
        description="Maximum iterations the agent can perform in a single run.",
    )
    max_loop_count: int = Field(
        default=20,
        alias="MAX_LOOP_COUNT",
        ge=1,
        le=100,
        description="Maximum loop count before forcing termination (safety limit).",
    )
    execution_timeout: int = Field(
        default=50000,
        alias="EXECUTION_TIMEOUT",
        ge=1000,
        le=300000,
        description="Timeout for code execution in milliseconds.",
    )

    # ── Path Configuration ────────────────────────────────────────────────────
    workspace_root: Path = Field(
        default=Path.cwd() / "sandbox_workspaces",
        alias="WORKSPACE_ROOT",
        description="Root directory for sandbox workspaces.",
    )
    log_dir: Path = Field(
        default=Path.cwd() / "logs",
        alias="LOG_DIR",
        description="Directory for log files.",
    )
    db_path: Path = Field(
        default=Path.home() / ".ideaagent" / "tasks.db",
        alias="DB_PATH",
        description="Path to the SQLite database file.",
    )

    # ── MongoDB Configuration (Legacy) ────────────────────────────────────────
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        alias="MONGODB_URI",
        description="MongoDB connection URI (legacy, ignored when using SQLite).",
    )
    mongodb_database: str = Field(
        default="IdeaAgent",
        alias="MONGODB_DATABASE",
        description="MongoDB database name (legacy).",
    )

    # ── MCP Configuration ─────────────────────────────────────────────────────
    mcp_enabled: bool = Field(
        default=False,
        alias="MCP_ENABLED",
        description="Enable MCP integration.",
    )
    mcp_config_path: str = Field(
        default="mcp_config.json",
        alias="MCP_CONFIG_PATH",
        description="Path to MCP configuration file.",
    )

    # ── Logging Configuration ─────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    log_file: str = Field(
        default="ideaagent.log",
        alias="LOG_FILE",
        description="Log file name.",
    )

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
        env_ignore_empty=True,
    )

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within acceptable range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens", "execution_timeout")
    @classmethod
    def validate_positive_integers(cls, v: int) -> int:
        """Validate that integer values are positive."""
        if v <= 0:
            raise ValueError("value must be positive")
        return v

    def validate_paths(self) -> None:
        """Ensure required directories exist."""
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


# Singleton instance – import this everywhere instead of reading os.getenv() directly.
settings = Settings()


def ensure_directories() -> None:
    """Create required directories if they don't exist.
    
    Call this function explicitly when needed instead of auto-creating on import.
    """
    settings.validate_paths()
