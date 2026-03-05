"""Custom exceptions for IdeaAgent.

Provides a hierarchy of exceptions for clear error classification
and structured error handling throughout the application.
"""

from enum import Enum
from typing import Optional


# ── Error type classification ────────────────────────────────────────────────

class ErrorType(str, Enum):
    """Classification of execution errors for the agent loop."""
    MISSING_PACKAGE = "missing_package"
    MISSING_FILE = "missing_file"
    MISSING_DIRECTORY = "missing_directory"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


# ── Base exception ────────────────────────────────────────────────────────────

class IdeaAgentError(Exception):
    """Base exception for all IdeaAgent errors."""


# ── LLM exceptions ───────────────────────────────────────────────────────────

class LLMError(IdeaAgentError):
    """LLM interaction failed."""


class LLMNotConfiguredError(LLMError):
    """LLM client is not configured (missing API key, etc.)."""


class LLMResponseParseError(LLMError):
    """Failed to parse the LLM response."""


# ── Execution exceptions ─────────────────────────────────────────────────────

class ExecutionError(IdeaAgentError):
    """A step failed during sandbox execution.

    Attributes:
        step: Step number that failed.
        error: Raw error message from the sandbox.
        code: The code that was executed.
        error_type: Classified type of the error.
    """

    def __init__(
        self,
        step: int,
        error: str,
        code: str = "",
        error_type: ErrorType = ErrorType.UNKNOWN,
    ) -> None:
        self.step = step
        self.error = error
        self.code = code
        self.error_type = error_type
        super().__init__(f"Step {step} failed [{error_type.value}]: {error[:200]}")


class MaxRetriesExceededError(ExecutionError):
    """Agent loop exhausted all retry attempts without success."""

    def __init__(self, step: int, attempts: int, last_error: str) -> None:
        self.attempts = attempts
        super().__init__(
            step=step,
            error=f"Failed after {attempts} attempt(s): {last_error}",
            error_type=ErrorType.RUNTIME_ERROR,
        )


# ── Skill exceptions ──────────────────────────────────────────────────────────

class SkillNotFoundError(IdeaAgentError):
    """The requested skill does not exist."""

    def __init__(self, skill_name: str) -> None:
        self.skill_name = skill_name
        super().__init__(f"Skill not found: {skill_name!r}")


# ── Database exceptions ───────────────────────────────────────────────────────

class DatabaseError(IdeaAgentError):
    """Database operation failed."""


# ── Configuration exceptions ──────────────────────────────────────────────────

class ConfigurationError(IdeaAgentError):
    """Invalid or missing configuration."""


# ── Helper: error classifier ──────────────────────────────────────────────────

def classify_error(error: str) -> ErrorType:
    """Classify an error string into an :class:`ErrorType`.

    Args:
        error: The raw error/stderr string from sandbox execution.

    Returns:
        The most specific :class:`ErrorType` that matches, or
        :attr:`ErrorType.UNKNOWN` if nothing matches.
    """
    if not error or not error.strip():
        return ErrorType.UNKNOWN

    lower = error.lower()

    if "modulenotfounderror" in lower or "no module named" in lower:
        return ErrorType.MISSING_PACKAGE
    if "importerror" in lower and "cannot import" in lower:
        return ErrorType.MISSING_PACKAGE
    if "filenotfounderror" in lower:
        return ErrorType.MISSING_FILE
    if "notadirectoryerror" in lower or "no such file or directory" in lower:
        return ErrorType.MISSING_DIRECTORY
    if "syntaxerror" in lower or "indentationerror" in lower or "taberror" in lower:
        return ErrorType.SYNTAX_ERROR
    if "timeouterror" in lower or "timed out" in lower:
        return ErrorType.TIMEOUT

    return ErrorType.RUNTIME_ERROR
