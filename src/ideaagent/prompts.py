"""Prompt templates for IdeaAgent LLM calls.

All prompt string constants and builder functions live here so that
llm.py stays focused on API call logic only.

ENCODING NOTE: Do NOT put special Unicode symbols (checkmarks, arrows,
superscripts, bullets, etc.) anywhere in these prompts.  Doing so causes
the LLM to imitate them and triggers UnicodeEncodeError on Windows
GBK/CP936 consoles at runtime.  Use ASCII-only alternatives throughout.
"""

from __future__ import annotations

import json
from typing import Optional

from .models import ResearchType


# ===========================================================================
# AVAILABLE ACTIONS - System Prompt Section
# ===========================================================================

_AVAILABLE_ACTIONS_SECTION = """
## Available Actions

You are an autonomous AI Agent. You can perform the following actions:

### 1. `<bash>command</bash>` - Execute Shell Command
- Runs in the .venv virtual environment
- Supports pip, python, git, curl, and other shell commands
- Output is displayed in real-time
- Example: `<bash>pip install torch numpy</bash>`

### 2. `<python>code</python>` - Execute Python Code
- Run Python code snippets directly
- Executes in the .venv environment
- Output is displayed in real-time
- Example: `<python>import torch; print(torch.__version__)</python>`

### 3. `<write_file path="relative/path">content</write_file>` - Write File
- Create or overwrite files in the workspace
- Path must be relative (e.g., "src/model.py")
- Parent directories are created automatically if needed
- Example: `<write_file path="config.py">BATCH_SIZE = 128</write_file>`

### 4. `<read_file>relative/path</read_file>` - Read File
- Read existing files from the workspace
- Returns file content for your reference
- Example: `<read_file>data/config.json</read_file>`

### 5. `<mkdir path="relative/path"></mkdir>` - Create Directory
- Create directories in the workspace
- Supports recursive creation (like mkdir -p)
- Path must be relative
- Example: `<mkdir path="data/raw"></mkdir>`

### 6. `<thinking>thoughts</thinking>` - Thinking Process
- Display your reasoning process
- Does not execute any action
- Example: `<thinking>I need to install dependencies first</thinking>`

## Important Rules

1. All file paths must be **relative paths** (relative to workspace)
2. Do NOT use absolute paths
3. Ensure directories exist before writing files (use `<mkdir>` or auto-create)
4. Ensure dependencies are installed before executing Python scripts
5. Commands executed with `<bash>` run in the .venv environment
6. Use ASCII-only characters in print() statements (avoid Unicode symbols)

## Example Project Structure

You can autonomously create hierarchical structures like:

```
workspace/
+-- config.py           # Configuration and hyperparameters
+-- data/               # Data directory
|   +-- raw/
+-- src/
|   +-- __init__.py
|   +-- data_loader.py  # Data loading
|   +-- model.py        # Model definitions
|   +-- train.py        # Training logic
+-- results/            # Output results
|   +-- metrics.json
|   +-- plots/
+-- logs/               # Log files
```
"""


# ---------------------------------------------------------------------------
# Plan generation prompts
# ---------------------------------------------------------------------------

def get_plan_system_prompt(research_type: ResearchType, available_skills: str = "") -> str:
    """System prompt for experiment plan generation."""
    base = (
        f"You are an expert research assistant specializing in "
        f"{research_type.value.replace('-', ' ')}.\n"
        "Your task is to help researchers validate their experimental ideas by "
        "creating detailed, actionable plans.\n\n"
        "When generating plans:\n"
        "1. Break down the research idea into clear, sequential steps\n"
        "2. Each step should be specific and executable\n"
        "3. Estimate realistic time requirements\n"
        "4. Identify which skills (if any) are needed for each step\n"
        "5. Consider potential challenges and edge cases\n"
        f"6. Ensure the plan follows best practices for "
        f"{research_type.value.replace('-', ' ')} research\n"
    )
    if available_skills:
        base += f"\n\nAvailable skills that can be used:\n{available_skills}"
    return base


def get_plan_user_prompt(idea_description: str) -> str:
    """User prompt for experiment plan generation."""
    return (
        "Please create a detailed experiment plan for the following research idea:\n\n"
        f"{idea_description}\n\n"
        "Respond with a JSON object in this exact format:\n"
        "{\n"
        '    "title": "Clear, descriptive title for the experiment",\n'
        '    "description": "Brief overview of what this experiment aims to achieve",\n'
        '    "estimated_total_time": 60,\n'
        '    "skills_needed": ["skill-name-1", "skill-name-2"],\n'
        '    "steps": [\n'
        "        {\n"
        '            "step_number": 1,\n'
        '            "description": "Clear description of what to do in this step",\n'
        '            "skill_required": "skill-name",\n'
        '            "estimated_duration": 15\n'
        "        }\n"
        "    ]\n"
        "}\n\n"
        "Ensure the plan is practical, follows scientific methodology, and can be "
        "executed step by step."
    )


def get_plan_regeneration_prompt(
    idea_description: str,
    previous_plan: str,
    feedback: str,
) -> str:
    """User prompt for plan regeneration after user feedback."""
    return (
        "Please revise the experiment plan based on the following feedback.\n\n"
        f"Original research idea:\n{idea_description}\n\n"
        f"Previous plan:\n{previous_plan}\n\n"
        f"User feedback:\n{feedback}\n\n"
        "Create an improved plan that addresses the feedback while maintaining "
        "scientific rigor.\nUse the same JSON format as before."
    )


# ---------------------------------------------------------------------------
# Step execution prompts
# ---------------------------------------------------------------------------

# The execution system prompt is designed for the action-based Agent.
# The Agent uses XML-style action tags to interact with the environment.
# NOTE: Use a plain str (not r-string) - all backslashes here are intentional
#       double-escaped for Windows path examples shown to the LLM.

_EXECUTION_SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert {domain} researcher and Python programmer.\n"
    "You operate as an AUTONOMOUS AGENT that can directly interact with the\n"
    "workspace using action tags.\n"
    "\n"
    "You are working on Windows. Always use\n"
    "    if __name__ == '__main__':\n"
    "in Python scripts that you intend to run as standalone scripts.\n"
    "\n"
    "=== AVAILABLE ACTIONS ===\n"
    "\n"
    "You can use the following action tags in your response:\n"
    "\n"
    "1. <bash>command</bash> - Execute a shell command in the .venv environment\n"
    "   Example: <bash>pip install torch numpy pandas</bash>\n"
    "\n"
    "2. <python>code</python> - Execute Python code directly\n"
    "   Example: <python>import torch; print(torch.__version__)</python>\n"
    "\n"
    '3. <write_file path="relative/path">content</write_file> - Write content to a file\n'
    '   Example: <write_file path="config.py">BATCH_SIZE = 128</write_file>\n'
    "\n"
    "4. <read_file>relative/path</read_file> - Read a file's content\n"
    "   Example: <read_file>data/config.json</read_file>\n"
    "\n"
    '5. <mkdir path="relative/path"></mkdir> - Create a directory\n'
    '   Example: <mkdir path="data/raw"></mkdir>\n'
    "\n"
    "6. <thinking>thoughts</thinking> - Show your reasoning (not executed)\n"
    "   Example: <thinking>I need to install dependencies first</thinking>\n"
    "\n"
    "7. <listing_files>tree</listing_files> - List files in a directory\n"
    "   Example: <listing_files>tree</listing_files>\n"
    "\n"
    "=== MULTI-FILE PROJECT STRUCTURE ===\n"
    "\n"
    "You should create a well-structured project with multiple module files:\n"
    "\n"
    "Typical structure:\n"
    "  config.py       -- hyperparameters and constants\n"
    "  data_loader.py  -- data loading, preprocessing, dataset classes\n"
    "  model.py        -- model / algorithm definitions\n"
    "  train.py        -- training loop, optimizer, scheduler\n"
    "  evaluate.py     -- evaluation, metrics, reports\n"
    "  visualize.py    -- plots and figures\n"
    "  utils.py        -- shared helpers, seed setting\n"
    "\n"
    "=== RECOMMENDED WORKFLOW ===\n"
    "\n"
    "1. SEARCH BEFORE YOU READ (essential for large codebases):\n"
    "   NEVER blindly read_file a large file. First locate what you need:\n"
    "\n"
    "    a)list files in the current directory:\n"
    "      <listing_files>tree</listing_files>\n"
    "\n"
    "   b) Find files by name pattern:\n"
    "      <bash>find . -name '*.py' | grep -i keyword</bash>\n"
    "      <bash>find . -type f -name '*.yaml' | head -20</bash>\n"
    "\n"
    "   c) Search for a class, function, or variable definition:\n"
    "      <bash>grep -rn 'class MyClass' . --include='*.py'</bash>\n"
    "      <bash>grep -rn 'def target_function' . --include='*.py'</bash>\n"
    "      <bash>grep -rn 'VARIABLE_NAME' . --include='*.py' | head -20</bash>\n"
    "\n"
    "   d) Get line numbers in a specific file, then read only those lines:\n"
    "      <bash>grep -n 'class ' path/to/file.py | head -30</bash>\n"
    "      <bash>sed -n '50,120p' path/to/file.py</bash>\n"
    "\n"
    "   Only use <read_file path> once you have confirmed the file is relevant\n"
    "   and reasonably sized. For files > 500 lines, prefer sed/grep to extract\n"
    "   only the relevant section.\n"
    "\n"
    "2. Create directories as needed:\n"
    '   <mkdir path="results"></mkdir>\n'
    "\n"
    "3. Install dependencies:\n"
    "   <bash>pip install torch numpy pandas matplotlib</bash>\n"
    "\n"
    "4. Write or modify files:\n"
    '   <write_file path="config.py">\n'
    "BATCH_SIZE = 128\n"
    "LEARNING_RATE = 0.001\n"
    "EPOCHS = 10\n"
    "</write_file>\n"
    "\n"
    "5. Run the code:\n"
    "   <bash>python train.py</bash>\n"
    "\n"
    "=== FILE SYSTEM RULES ===\n"
    "\n"
    "- The WORKSPACE PATH is provided in the context under 'WORKSPACE (YOUR WORKING DIRECTORY)'\n"
    "- All <write_file> and <read_file> paths are RELATIVE to that workspace path\n"
    "- Do NOT use absolute paths in action tags\n"
    "- Use mkdir to create directories before writing files (or let write_file auto-create parents)\n"
    "- Use encoding='utf-8' for all text I/O\n"
    "- Save intermediates (.csv, .npy, .json, .pkl) for later steps\n"
    "\n"
    "=== ENCODING SAFETY (CRITICAL for Windows / GBK terminals) ===\n"
    "\n"
    "NEVER use non-ASCII symbols inside print() or f-string literals.\n"
    "Forbidden: checkmarks, crosses, arrows, stars, bullets, box-drawing, superscripts.\n"
    "Use ONLY: [OK]  [FAIL]  [WARN]  ->  <-  ^  v  *  -\n"
    "\n"
    "=== GENERAL RULES ===\n"
    "\n"
    "- Use action tags to perform tasks\n"
    "- You can use multiple actions in a single response\n"
    "- Actions are executed in the order they appear\n"
    "- Set random seeds (numpy, random, torch) for reproducibility\n"
    "- Use try/except around I/O and training; print [FAIL] on error\n"
    "- No wandb / tensorboard / mlflow unless explicitly required\n"
    "- No Unicode special symbols anywhere in generated code\n"
    "- Print progress messages using ASCII-only characters\n"
)


def get_execution_system_prompt(research_type: ResearchType) -> str:
    """System prompt for step code generation (shared by execute and fix)."""
    domain = research_type.value.replace("-", " ")
    return _EXECUTION_SYSTEM_PROMPT_TEMPLATE.format(domain=domain)


def get_execution_system_prompt_with_tools(
    research_type: ResearchType,
    available_tools: list,
) -> str:
    """System prompt augmented with an MCP tool list."""
    base = get_execution_system_prompt(research_type)
    tools_json = json.dumps(available_tools, indent=2, ensure_ascii=True)
    return base + f"\n\n=== AVAILABLE TOOLS (reference) ===\n```json\n{tools_json}\n```"


def get_execution_user_prompt(
    step_description: str,
    context: str,
    skill_instructions: Optional[str] = None,
) -> str:
    """User prompt asking the LLM to generate actions for executing a step."""
    lines = [
        "Execute the following step using action tags.",
        "",
        "== STEP ==",
        step_description,
        "",
        "== CONTEXT (outputs from previous steps) ==",
        context,
        "",
        "== REQUIREMENTS ==",
        "1. Use action tags to perform tasks: <bash>, <python>, <write_file>, <read_file>, <mkdir>, <thinking>",
        "2. Create a well-structured project with multiple module files (config.py, data_loader.py, model.py, train.py, etc.)",
        "3. Use <mkdir> to create directories before writing files",
        "4. Use <bash>pip install ...</bash> to install dependencies",
        '5. Use <write_file path="...">...</write_file> to create module files',
        "6. Use <bash>python xxx.py</bash> or <python>...</python> to run code",
        "7. Print progress using ASCII-only characters: [OK] [FAIL] [WARN]",
        "8. Save all outputs to files so subsequent steps can load them",
        "9. Include try/except around I/O and external calls",
        "10. Set random seeds for reproducibility",
    ]

    if skill_instructions:
        lines += ["", "== SKILL INSTRUCTIONS ==", skill_instructions]

    lines += ["", "Generate actions now:"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Error analysis / fix prompt
# ---------------------------------------------------------------------------

def get_fix_user_prompt(
    step_description: str,
    failed_code: str,
    error_message: str,
    context: str,
    attempt: int,
    max_attempts: int,
    skill_instructions: Optional[str] = None,
) -> str:
    """User prompt asking the LLM to diagnose and fix failed actions."""
    lines = [
        f"The following actions FAILED during execution "
        f"(attempt {attempt}/{max_attempts}).",
        "",
        "== STEP ==",
        step_description,
        "",
        "== ACTIONS THAT WERE EXECUTED ==",
        failed_code,
        "",
        "== ERROR OUTPUT ==",
        error_message,
        "",
        "== CONTEXT ==",
        context,
        "",
        "== YOUR TASK ==",
        "1. Diagnose the root cause of the error.",
        "2. Produce CORRECTED actions using action tags.",
        "3. Use action tags: <bash>, <python>, <write_file>, <read_file>, <mkdir>, <thinking>",
        "",
        "COMMON FIXES:",
        "- ModuleNotFoundError: use <bash>pip install package-name</bash> to install missing package.",
        "- UnicodeEncodeError (GBK/CP936): replace ALL non-ASCII symbols in print()/f-strings",
        "  with ASCII alternatives: [OK] [FAIL] [WARN] -> <- ^ v * -",
        "- FileNotFoundError: check Path(...).exists() before reading.",
        '- PermissionError: use <mkdir path="..."></mkdir> first.',
        "- Other: wrap failing section in try/except and print [FAIL] on error.",
    ]

    if skill_instructions:
        lines += ["", "== SKILL INSTRUCTIONS ==", skill_instructions]

    lines += ["", "Generate the FIXED actions now:"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judge-and-fix prompt
# ---------------------------------------------------------------------------

def get_judge_user_prompt(
    step_description: str,
    code: str,
    stdout: str,
    stderr: str,
    returncode: int,
    context: str,
    attempt: int,
    max_attempts: int,
    skill_instructions: Optional[str] = None,
) -> str:
    """User prompt asking the LLM to judge execution and return fix or success."""
    stdout_trunc = (stdout or "")[:4000]
    stderr_trunc = (stderr or "")[:2000]

    lines = [
        f"You are acting as an autonomous agent executor "
        f"(attempt {attempt}/{max_attempts}).",
        "",
        "You just executed the following actions for this step:",
        "",
        "== STEP ==",
        step_description,
        "",
        "== ACTIONS THAT WERE EXECUTED ==",
        code,
        "",
        "== EXECUTION RESULT ==",
        f"Return code: {returncode}",
        "",
        "--- stdout ---",
        stdout_trunc,
        "",
        "--- stderr ---",
        stderr_trunc,
        "",
        "== CONTEXT (previous steps) ==",
        context,
        "",
        "== YOUR JOB ==",
        "Read ALL of the output above carefully.",
        "",
        "If the step completed successfully (ran without errors and produced expected outputs):",
        '  Respond with ONLY this JSON (no other text):',
        '  {"status": "success"}',
        "",
        "If there are ANY errors, exceptions, logical failures, or wrong/incomplete output:",
        '  Diagnose the root cause, then respond with ONLY this JSON (no markdown):',
        '  {"status": "fix", "reason": "<brief diagnosis>", '
        '"code": "<complete corrected actions with action tags>"}',
        "",
        'The "code" field must contain the ENTIRE corrected response with action tags (not a diff).',
        "Use action tags: <bash>, <python>, <write_file>, <read_file>, <mkdir>, <thinking>",
        "All code MUST follow encoding rules (ASCII-only in print()).",
    ]

    if skill_instructions:
        lines += ["", "== SKILL INSTRUCTIONS ==", skill_instructions]

    return "\n".join(lines)
