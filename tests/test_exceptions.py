"""Unit tests for ideaagent.exceptions."""

import pytest
from ideaagent.exceptions import (
    ErrorType,
    ExecutionError,
    MaxRetriesExceededError,
    SkillNotFoundError,
    classify_error,
)


class TestClassifyError:
    def test_missing_package(self):
        assert classify_error("ModuleNotFoundError: No module named 'numpy'") == ErrorType.MISSING_PACKAGE

    def test_import_error(self):
        assert classify_error("ImportError: cannot import name 'foo'") == ErrorType.MISSING_PACKAGE

    def test_file_not_found(self):
        assert classify_error("FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'") == ErrorType.MISSING_FILE

    def test_syntax_error(self):
        assert classify_error("SyntaxError: invalid syntax") == ErrorType.SYNTAX_ERROR

    def test_indentation_error(self):
        assert classify_error("IndentationError: unexpected indent") == ErrorType.SYNTAX_ERROR

    def test_timeout(self):
        assert classify_error("TimeoutError: timed out after 300s") == ErrorType.TIMEOUT

    def test_generic_runtime(self):
        assert classify_error("ValueError: invalid literal for int()") == ErrorType.RUNTIME_ERROR

    def test_empty_string(self):
        assert classify_error("") == ErrorType.UNKNOWN

    def test_none_like_empty(self):
        assert classify_error("   ") == ErrorType.UNKNOWN


class TestExecutionError:
    def test_basic(self):
        exc = ExecutionError(step=2, error="something went wrong")
        assert exc.step == 2
        assert "something went wrong" in str(exc)

    def test_with_error_type(self):
        exc = ExecutionError(step=1, error="no module", error_type=ErrorType.MISSING_PACKAGE)
        assert exc.error_type == ErrorType.MISSING_PACKAGE

    def test_long_error_truncated_in_str(self):
        long_error = "x" * 500
        exc = ExecutionError(step=1, error=long_error)
        # __str__ truncates at 200 chars
        assert len(str(exc)) < 400


class TestMaxRetriesExceededError:
    def test_attributes(self):
        exc = MaxRetriesExceededError(step=3, attempts=5, last_error="timeout")
        assert exc.step == 3
        assert exc.attempts == 5
        assert "5" in str(exc)


class TestSkillNotFoundError:
    def test_message(self):
        exc = SkillNotFoundError("visualization")
        assert "visualization" in str(exc)
        assert exc.skill_name == "visualization"
