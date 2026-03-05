"""Unit tests for ideaagent.utils.code_parser."""

import pytest
from ideaagent.utils.code_parser import (
    extract_python_code,
    validate_python_code,
    extract_package_names,
)


# ── validate_python_code ──────────────────────────────────────────────────────

class TestValidatePythonCode:
    def test_valid_simple_script(self):
        code = "x = 1\nprint(x)\n"
        ok, msg = validate_python_code(code)
        assert ok is True
        assert msg == "Valid"

    def test_empty_string(self):
        ok, msg = validate_python_code("")
        assert ok is False

    def test_whitespace_only(self):
        ok, msg = validate_python_code("   \n  ")
        assert ok is False

    def test_syntax_error(self):
        code = "def broken(\n    pass"
        ok, msg = validate_python_code(code)
        assert ok is False
        assert "SyntaxError" in msg

    def test_indentation_error(self):
        code = "if True:\nprint('bad')"
        ok, msg = validate_python_code(code)
        assert ok is False


# ── extract_python_code ───────────────────────────────────────────────────────

class TestExtractPythonCode:
    def test_python_fenced_block(self):
        text = "Here is the code:\n```python\nprint('hello')\n```\n"
        result = extract_python_code(text)
        assert "print('hello')" in result

    def test_py_fenced_block(self):
        text = "```py\nx = 1 + 2\nprint(x)\n```"
        result = extract_python_code(text)
        assert "x = 1 + 2" in result

    def test_generic_fenced_fallback(self):
        text = "```\nimport os\nprint(os.getcwd())\n```"
        result = extract_python_code(text)
        assert "import os" in result

    def test_no_code_block(self):
        text = "This is just a plain text answer with no code."
        result = extract_python_code(text)
        assert result == ""

    def test_empty_input(self):
        assert extract_python_code("") == ""
        assert extract_python_code("   ") == ""

    def test_multiple_blocks_merged(self):
        text = (
            "Step 1:\n```python\nimport os\nprint('step1')\n```\n\n"
            "Step 2:\n```python\nimport os\nprint('step2')\n```"
        )
        result = extract_python_code(text)
        # import should appear only once (de-duplicated)
        assert result.count("import os") == 1
        assert "print('step1')" in result
        assert "print('step2')" in result

    def test_explanation_only_block_skipped(self):
        text = (
            "```python\n# This is just a comment\n```\n\n"
            "```python\nprint('actual code')\n```"
        )
        result = extract_python_code(text)
        assert "print('actual code')" in result

    def test_returns_valid_syntax(self):
        text = "```python\ndef main():\n    x = 1\n    print(x)\n\nmain()\n```"
        result = extract_python_code(text)
        ok, _ = validate_python_code(result)
        assert ok is True


# ── extract_package_names ─────────────────────────────────────────────────────

class TestExtractPackageNames:
    def test_single_package(self):
        error = "ModuleNotFoundError: No module named 'numpy'"
        pkgs = extract_package_names(error)
        assert "numpy" in pkgs

    def test_submodule_extracts_top_level(self):
        error = "ModuleNotFoundError: No module named 'sklearn.preprocessing'"
        pkgs = extract_package_names(error)
        assert "sklearn" in pkgs

    def test_no_match(self):
        error = "RuntimeError: division by zero"
        pkgs = extract_package_names(error)
        assert pkgs == []

    def test_no_duplicates(self):
        error = (
            "No module named 'pandas'\n"
            "No module named 'pandas'"
        )
        pkgs = extract_package_names(error)
        assert pkgs.count("pandas") == 1
