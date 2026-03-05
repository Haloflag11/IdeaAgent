"""Workspace management utilities for IdeaAgent."""

import sys
from pathlib import Path
from typing import Optional


def get_workspace_structure(
    workspace_dir: Path,
    max_depth: int = 3,
) -> str:
    """Return a tree-style string representation of *workspace_dir*.

    Args:
        workspace_dir: Root directory to traverse.
        max_depth: Maximum depth of directory traversal.

    Returns:
        Multi-line string showing the directory tree, or an informational
        message if the directory does not exist yet.
    """
    if not workspace_dir.exists():
        return "Workspace directory does not exist yet."

    structure: list[str] = []

    def _traverse(path: Path, depth: int, prefix: str = "") -> None:
        if depth > max_depth:
            return
        try:
            items = sorted(
                path.iterdir(),
                key=lambda x: (x.is_file(), x.name.lower()),
            )
        except PermissionError:
            return

        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            structure.append(f"{prefix}{connector}{item.name}")
            if item.is_dir():
                extension = "    " if is_last else "│   "
                _traverse(item, depth + 1, prefix + extension)

    structure.append(f"📁 {workspace_dir.name}/")
    _traverse(workspace_dir, 0)
    return "\n".join(structure)


def build_execution_context(
    execution_context: list[dict],
    current_step: int,
    max_steps_in_context: int = 10,
) -> str:
    """Build a human-readable context string from previous step results.

    Includes step outputs, success/failure status, and any errors.
    Applies a sliding window so that very long histories are pruned to the
    most recent *max_steps_in_context* entries.

    Args:
        execution_context: List of step-result dicts (keys: step_number,
            description, success, output, error).
        current_step: The step number currently being executed (used only
            for the header).
        max_steps_in_context: Maximum number of previous steps to include.

    Returns:
        Formatted context string.
    """
    if not execution_context:
        return "No previous execution context."

    # Prune: keep only recent steps
    recent = execution_context[-max_steps_in_context:]

    parts: list[str] = []
    for ctx in recent:
        step_num = ctx.get("step_number", "?")
        desc = ctx.get("description", "")
        success = ctx.get("success", False)
        output = ctx.get("output") or ""
        error = ctx.get("error") or ""
        packages = ctx.get("packages_installed") or []
        files_created = ctx.get("files_created") or []

        block = [
            f"--- Step {step_num}: {desc} ---",
            f"  Status: {'SUCCESS' if success else 'FAILED'}",
        ]
        if output:
            truncated = output[:500] + "..." if len(output) > 500 else output
            block.append(f"  Output:\n{truncated}")
        if error:
            truncated_err = error[:300] + "..." if len(error) > 300 else error
            block.append(f"  Error:\n{truncated_err}")
        if packages:
            block.append(f"  Packages installed: {', '.join(packages)}")
        if files_created:
            block.append(f"  Files created: {', '.join(files_created)}")

        parts.append("\n".join(block))

    return "\n\n".join(parts)


def build_rich_context(
    workspace_dir: Path,
    execution_context: list[dict],
    current_step: int,
    installed_packages: Optional[list[str]] = None,
    user_workspace_path: Optional[Path] = None,
    constant_context: Optional[str] = None,
) -> str:
    """Build a comprehensive context dict string for the LLM.

    Combines:
    - Constant context (initial instruction + workspace, if provided)
    - User workspace RAG context (if a user workspace is specified)
    - Workspace directory structure
    - Previous step history (with outputs, errors, installed packages)
    - Environment information

    Args:
        workspace_dir: The task workspace directory.
        execution_context: History of previous step results.
        current_step: The current step number.
        installed_packages: Packages installed so far in this task.
        user_workspace_path: Optional path to the user-specified workspace
            directory.  When provided, all readable files in that directory
            are scanned and their content is injected as AgenticRAG context
            so the Agent can reference existing data, code, and notes.
        constant_context: Optional constant context string from ContextManager.
            This ensures initial instruction and workspace path are always
            included in every LLM call.

    Returns:
        Formatted multi-section context string.
    """
    sections: list[str] = []

    # 0. Constant context (always first - initial instruction + workspace)
    if constant_context:
        sections.append(constant_context)

    # 1. User workspace
    if user_workspace_path is not None:
        try:
            from .workspace_rag import build_workspace_rag_context
            rag_context = build_workspace_rag_context(user_workspace_path)
            if rag_context:
                sections.append(rag_context)
        except Exception as _rag_exc:
            import logging as _logging
            _logging.getLogger("IdeaAgent.workspace").warning(
                "Failed to build AgenticRAG context from %s: %s",
                user_workspace_path, _rag_exc,
            )

    # 1. Workspace layout guidelines (always first so the LLM sees it prominently)
    sections.append(
        "=== WORKSPACE LAYOUT (MANDATORY) ===\n"
        "The task workspace has the following pre-created subdirectories.\n"
        "You MUST save all outputs to the appropriate subdirectory – never write files\n"
        "directly into the workspace root.\n\n"
        "  data/     – raw and processed datasets (.csv, .json, .npy, …)\n"
        "  models/   – serialised model artefacts (.pkl, .pth, .h5, .cbm, …)\n"
        "  plots/    – all visualisation outputs (.png, .svg, .html, …)\n"
        "  results/  – metrics, evaluation reports, summaries (.json, .csv, …)\n"
        "  logs/     – training / run logs (.txt, .csv, …)\n\n"
        "Example paths (relative to the workspace root):\n"
        '  pd.DataFrame(results).to_csv("results/metrics.csv")\n'
        '  joblib.dump(model, "models/rf_model.pkl")\n'
        '  plt.savefig("plots/confusion_matrix.png")\n'
        '  json.dump(stats, open("results/summary.json", "w"))\n\n'
        "The subdirectories already exist – do NOT call mkdir() on them."
    )

    # 2. Workspace structure
    ws_struct = get_workspace_structure(workspace_dir)
    sections.append(f"=== WORKSPACE STRUCTURE ===\n{ws_struct}")

    # 3. Execution history
    ctx_str = build_execution_context(execution_context, current_step)
    sections.append(f"=== EXECUTION HISTORY (previous steps) ===\n{ctx_str}")

    # 4. Environment
    env_lines = [
        f"Python: {sys.version.split()[0]}",
        f"Working directory: {workspace_dir}",
        f"Platform: {sys.platform}",
    ]
    if installed_packages:
        env_lines.append(f"Packages installed in this task: {', '.join(installed_packages)}")
    sections.append("=== ENVIRONMENT ===\n" + "\n".join(env_lines))

    return "\n\n".join(sections)
