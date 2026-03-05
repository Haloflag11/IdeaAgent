"""Code extraction, action parsing, and validation utilities for IdeaAgent.

This module implements the core parsing layer of the Agentic Loop.
The LLM outputs structured action blocks; this module extracts and
classifies them so the executor can run each action and feed results
back into the conversation context.

Action types supported
----------------------
<bash>...</bash>            Shell command(s) to run via subprocess
<python>...</python>        Python code to execute in the workspace
<read_file>path</read_file> Read a file and return its content
<write_file path="...">content</write_file>  Write content to a file
<task_complete>msg</task_complete>  Signal that the step is done
"""

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Unicode → ASCII substitution table
# Covers the most common symbols that LLMs insert in print() statements and
# that cause UnicodeEncodeError on Windows terminals using GBK/CP936.
# ---------------------------------------------------------------------------
_UNICODE_REPLACEMENTS: dict[str, str] = {
    # Checkmarks / crosses
    "\u2713": "[OK]",    # ✓
    "\u2714": "[OK]",    # ✔
    "\u2717": "[FAIL]",  # ✗
    "\u2718": "[FAIL]",  # ✘
    "\u2716": "[FAIL]",  # ✖
    # Warning / info
    "\u26a0": "[WARN]",  # ⚠
    "\u2139": "[INFO]",  # ℹ
    # Arrows
    "\u2192": "->",      # →
    "\u2190": "<-",      # ←
    "\u2191": "^",       # ↑
    "\u2193": "v",       # ↓
    "\u21d2": "=>",      # ⇒
    # Stars / bullets
    "\u2605": "[*]",     # ★
    "\u2606": "[*]",     # ☆
    "\u25cf": "[o]",     # ●
    "\u25cb": "[o]",     # ○
    "\u25c6": "[>]",     # ◆
    "\u25c7": "[>]",     # ◇
    "\u2022": "-",       # •
    "\u25b6": ">",       # ▶
    "\u25aa": "-",       # ▪
    # Superscripts (common in R², m², etc.)
    "\u00b2": "2",       # ²
    "\u00b3": "3",       # ³
    "\u00b9": "1",       # ¹
    # Fractions / math
    "\u00bd": "1/2",     # ½
    "\u00bc": "1/4",     # ¼
    "\u00d7": "x",       # ×
    "\u00f7": "/",       # ÷
    "\u2212": "-",       # − (minus sign)
    "\u221e": "inf",     # ∞
    # Deco
    "\u2014": "--",      # — (em dash)
    "\u2013": "-",       # – (en dash)
    "\u2026": "...",     # …
    # Box-drawing / progress bars (common in tqdm / rich output)
    "\u2588": "#",       # █
    "\u2592": "#",       # ▒
    "\u2591": ".",       # ░
    "\u258c": "|",       # ▌
    "\u2590": "|",       # ▐
    # Tick/box symbols
    "\u2610": "[ ]",     # ☐
    "\u2611": "[x]",     # ☑
    "\u2612": "[x]",     # ☒
    # Miscellaneous
    "\u00e9": "e",       # é
    "\u2764": "<3",      # ❤
}


# ===========================================================================
# Action types
# ===========================================================================

class ActionType(Enum):
    BASH = "bash"
    PYTHON = "python"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    MKDIR = "mkdir"         # <mkdir path="..."></mkdir> - create directory
    TASK_COMPLETE = "task_complete"
    THINKING = "thinking"   # <thinking> blocks – log only, don't execute


@dataclass
class AgentAction:
    """A single action emitted by the LLM."""
    action_type: ActionType
    content: str                       # code / path / message
    params: dict[str, str] = field(default_factory=dict)  # extra attrs from tag


# ===========================================================================
# Action parser
# ===========================================================================

# Patterns for each action tag.  We use re.DOTALL so content can be multiline.
_ACTION_PATTERNS: list[tuple[ActionType, re.Pattern]] = [
    (ActionType.BASH,
     re.compile(r"<bash>(.*?)</bash>", re.DOTALL | re.IGNORECASE)),
    (ActionType.PYTHON,
     re.compile(r"<python>(.*?)</python>", re.DOTALL | re.IGNORECASE)),
    (ActionType.READ_FILE,
     re.compile(r"<read_file>(.*?)</read_file>", re.DOTALL | re.IGNORECASE)),
    (ActionType.WRITE_FILE,
     re.compile(
         r'<write_file\s+path=["\']([^"\']+)["\']>(.*?)</write_file>',
         re.DOTALL | re.IGNORECASE,
     )),
    (ActionType.MKDIR,
     re.compile(
         r'<mkdir\s+path=["\']([^"\']+)["\']>\s*</mkdir>',
         re.DOTALL | re.IGNORECASE,
     )),
    (ActionType.TASK_COMPLETE,
     re.compile(r"<task_complete>(.*?)</task_complete>", re.DOTALL | re.IGNORECASE)),
    (ActionType.THINKING,
     re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)),
]

# Also support legacy ```python ... ``` fenced blocks as PYTHON actions
_PYTHON_FENCE = re.compile(
    r"```(?:python|py)\s*\n(.*?)(?:\n```|```)",
    re.DOTALL | re.IGNORECASE,
)
_BASH_FENCE = re.compile(
    r"```(?:bash|sh|shell)\s*\n(.*?)(?:\n```|```)",
    re.DOTALL | re.IGNORECASE,
)
_GENERIC_FENCE = re.compile(
    r"```\s*\n(.*?)(?:\n```|```)",
    re.DOTALL,
)


def parse_agent_actions(text: str) -> list[AgentAction]:
    """Parse all agent actions from an LLM response.

    The parser recognises two formats:

    1. **XML-style action tags** (preferred, new format)::

        <bash>pip install pandas</bash>
        <python>import pandas as pd ...</python>
        <read_file>data/results.csv</read_file>
        <write_file path="results/summary.md">...</write_file>
        <task_complete>All done.</task_complete>

    2. **Markdown fenced code blocks** (fallback)::

        ```python
        import pandas as pd
        ```
        ```bash
        pip install pandas
        ```

    Priority rules:
    - If XML-style action tags are found, they take priority
    - If NO XML-style tags are found, fenced code blocks are parsed
    - If BOTH are found, XML-style tags are used (they indicate the LLM is using the new format)

    Args:
        text: Raw LLM response.

    Returns:
        Ordered list of :class:`AgentAction` objects.
    """
    if not text or not text.strip():
        return []

    # Collect (start_pos, action) pairs so we can sort by position
    found: list[tuple[int, AgentAction]] = []

    # --- XML-style tags ---
    for action_type, pattern in _ACTION_PATTERNS:
        if action_type == ActionType.WRITE_FILE:
            for m in pattern.finditer(text):
                path_val = m.group(1).strip()
                content = m.group(2)
                content = sanitize_unicode(content)
                found.append((
                    m.start(),
                    AgentAction(
                        action_type=action_type,
                        content=content,
                        params={"path": path_val},
                    ),
                ))
        else:
            for m in pattern.finditer(text):
                content = m.group(1).strip()
                if action_type == ActionType.PYTHON:
                    content = sanitize_unicode(content)
                found.append((
                    m.start(),
                    AgentAction(action_type=action_type, content=content),
                ))

    # --- Check if XML-style actions were found (excluding thinking) ---
    has_xml_actions = any(
        a.action_type not in (ActionType.THINKING,)
        for _, a in found
    )

    if has_xml_actions:
        # XML-style tags found - only return XML actions, sort and return
        found.sort(key=lambda x: x[0])
        return [action for _, action in found]

    # --- No XML-style actions found - fall back to fenced code blocks ---
    # This is the legacy / fallback mode for backward compatibility
    fence_found: list[tuple[int, AgentAction]] = []
    
    for m in _PYTHON_FENCE.finditer(text):
        content = sanitize_unicode(m.group(1).strip())
        fence_found.append((
            m.start(),
            AgentAction(action_type=ActionType.PYTHON, content=content),
        ))
    
    for m in _BASH_FENCE.finditer(text):
        fence_found.append((
            m.start(),
            AgentAction(action_type=ActionType.BASH, content=m.group(1).strip()),
        ))
    
    if not fence_found:
        # Try generic fenced blocks as last resort
        for m in _GENERIC_FENCE.finditer(text):
            content = sanitize_unicode(m.group(1).strip())
            # Heuristic: if it looks like Python, treat it as Python
            if _looks_like_python(content):
                fence_found.append((
                    m.start(),
                    AgentAction(action_type=ActionType.PYTHON, content=content),
                ))
    
    # Sort by position so execution order matches the text order
    fence_found.sort(key=lambda x: x[0])
    return [action for _, action in fence_found]


def _looks_like_python(code: str) -> bool:
    """Heuristic: does this code block look like Python?"""
    python_keywords = ("import ", "def ", "class ", "print(", "for ", "if ",
                       "return ", "from ", "with ", "try:", "except")
    return any(kw in code for kw in python_keywords)


# ===========================================================================
# Legacy helpers (kept for backward compatibility with existing callers)
# ===========================================================================

def _is_explanation_only(code: str) -> bool:
    """Return True if *code* contains no executable statements."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    executable = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.stmt)
        and not isinstance(node, (ast.Expr,))
    ]
    expr_nodes = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and not isinstance(node.value, ast.Constant)
    ]
    return len(executable) == 0 and len(expr_nodes) == 0


def _merge_blocks(blocks: list[str]) -> str:
    """Merge multiple code blocks into a single script (dedup imports)."""
    if not blocks:
        return ""
    if len(blocks) == 1:
        return blocks[0]

    seen_imports: set[str] = set()
    merged_parts: list[str] = []

    for block in blocks:
        lines = block.splitlines()
        filtered: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                if stripped not in seen_imports:
                    seen_imports.add(stripped)
                    filtered.append(line)
            else:
                filtered.append(line)
        merged_parts.append("\n".join(filtered))

    return "\n\n".join(merged_parts)


def sanitize_unicode(code: str) -> str:
    """Replace problematic Unicode characters with ASCII-safe equivalents.

    Windows terminals that use GBK / CP936 encoding cannot encode many
    Unicode codepoints (e.g. ✓ U+2713, ² U+00B2).  LLMs frequently emit
    these in ``print()`` statements, causing ``UnicodeEncodeError`` at
    runtime.  This function replaces the most common offenders with plain
    ASCII alternatives so the generated script runs without encoding issues.

    Args:
        code: Python source code that may contain Unicode characters.

    Returns:
        The same code with problematic Unicode characters replaced.
    """
    if not code:
        return code
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        if char in code:
            code = code.replace(char, replacement)
    return code


def validate_python_code(code: str) -> tuple[bool, str]:
    """Check whether *code* is syntactically valid Python.

    Args:
        code: Python source code string.

    Returns:
        ``(True, "Valid")`` on success, or ``(False, error_message)`` on failure.
    """
    if not code or not code.strip():
        return False, "Empty code"
    try:
        ast.parse(code)
        return True, "Valid"
    except SyntaxError as exc:
        return False, f"SyntaxError at line {exc.lineno}: {exc.msg}"
    except Exception as exc:
        return False, str(exc)


def extract_python_code(text: str) -> str:
    r"""Extract the best executable Python code from *text*.

    This is the legacy single-block extractor.  For the new Agentic Loop,
    use :func:`parse_agent_actions` instead.

    Strategy:
    1. Try ```python``` / ```py``` fenced blocks.
    2. Fall back to generic ``` fenced blocks.
    3. Filter out explanation-only blocks.
    4. Merge remaining blocks (deduplicating imports).
    5. Validate syntax.

    Args:
        text: Arbitrary text (LLM response, markdown, etc.).

    Returns:
        Extracted Python source code, or ``""`` if nothing executable found.
    """
    if not text or not text.strip():
        return ""

    blocks: list[str] = [m.strip() for m in _PYTHON_FENCE.findall(text)]

    if not blocks:
        blocks = [m.strip() for m in _GENERIC_FENCE.findall(text)]

    executable_blocks = [b for b in blocks if b and not _is_explanation_only(b)]

    if not executable_blocks:
        return ""

    merged = _merge_blocks(executable_blocks)
    merged = sanitize_unicode(merged)

    valid, _ = validate_python_code(merged)
    if valid:
        return merged

    for block in executable_blocks:
        block = sanitize_unicode(block)
        valid, _ = validate_python_code(block)
        if valid:
            return block

    longest = max(executable_blocks, key=len)
    return sanitize_unicode(longest)


def extract_package_names(error: str) -> list[str]:
    """Extract missing package names from a ModuleNotFoundError message.

    Args:
        error: Error output from the Python interpreter.

    Returns:
        List of package names to install (may be empty).
    """
    packages: list[str] = []
    pattern = re.compile(r"No module named ['\"]?([a-zA-Z0-9_\-\.]+)['\"]?")
    for match in pattern.finditer(error):
        pkg = match.group(1).split(".")[0]
        if pkg and pkg not in packages:
            packages.append(pkg)
    return packages
