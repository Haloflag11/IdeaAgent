"""AgenticRAG – reads user workspace files and builds LLM-ready context.

This module is responsible for scanning a user-specified workspace directory,
reading supported file types, and producing a structured context string that
can be injected into the Agent's prompts (both plan generation and per-step
execution).

Design goals
------------
* **Zero heavy dependencies** – uses only the stdlib plus packages that are
  already required by IdeaAgent (numpy, pandas if available).
* **Graceful degradation** – if a file cannot be read, it is skipped with a
  warning rather than crashing.
* **Compact summaries** – large files are summarised (first N rows for CSV,
  shape/dtype for numpy arrays, etc.) so the context stays within token limits.
* **Read-only** – this module never writes to the user workspace.
"""

from __future__ import annotations

import json
import logging
import re
import traceback
from pathlib import Path
from typing import Optional

logger = logging.getLogger("IdeaAgent.workspace_rag")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum characters to include per file when injecting into the LLM context.
MAX_TEXT_CHARS_PER_FILE = 5_000
MAX_CODE_CHARS_PER_FILE = 8_000
MAX_CSV_ROWS = 50
# Maximum total context size (characters) produced by the RAG scan.
MAX_TOTAL_RAG_CHARS = 40_000
# Maximum directory depth to recurse into.
MAX_SCAN_DEPTH = 3

# File extensions grouped by reading strategy.
CODE_EXTENSIONS = {".py", ".ipynb", ".r", ".R", ".sh", ".bash", ".sql"}
TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".log", ".csv_schema"}
CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".env.example"}
CSV_EXTENSIONS = {".csv", ".tsv"}
NUMPY_EXTENSIONS = {".npy", ".npz"}
BINARY_SKIP_EXTENSIONS = {
    ".pkl", ".pth", ".h5", ".hdf5", ".parquet", ".feather",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico",
    ".zip", ".tar", ".gz", ".bz2", ".7z",
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".exe", ".dll", ".so", ".dylib",
    ".mp3", ".mp4", ".wav", ".avi",
    ".db", ".sqlite", ".sqlite3",
}


# ---------------------------------------------------------------------------
# Individual file readers
# ---------------------------------------------------------------------------

def _read_text_file(path: Path, max_chars: int = MAX_TEXT_CHARS_PER_FILE) -> str:
    """Read a plain-text file, truncating if needed."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n... [truncated – {len(content)} chars total]"
        return content
    except Exception as exc:
        return f"[Could not read file: {exc}]"


def _read_code_file(path: Path) -> str:
    """Read a source-code file (Python, R, SQL, …)."""
    return _read_text_file(path, max_chars=MAX_CODE_CHARS_PER_FILE)


def _read_csv_file(path: Path) -> str:
    """Read a CSV/TSV file and return a compact summary."""
    try:
        import pandas as pd
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, nrows=MAX_CSV_ROWS + 1,
                         encoding="utf-8", low_memory=False)
        truncated = len(df) > MAX_CSV_ROWS
        if truncated:
            df = df.head(MAX_CSV_ROWS)
        shape_line = f"Shape: {df.shape[0]}{'+'  if truncated else ''} rows x {len(df.columns)} columns"
        dtypes_line = "Dtypes: " + ", ".join(f"{c}({dt})" for c, dt in df.dtypes.items())
        preview = df.to_string(index=False, max_cols=20)
        parts = [shape_line, dtypes_line, "", "Preview:", preview]
        if truncated:
            parts.append(f"... [{MAX_CSV_ROWS} of more rows shown]")
        return "\n".join(parts)
    except ImportError:
        # pandas not available – fall back to plain text
        return _read_text_file(path, max_chars=MAX_TEXT_CHARS_PER_FILE)
    except Exception as exc:
        return f"[Could not parse CSV: {exc}]"


def _read_numpy_file(path: Path) -> str:
    """Read a .npy / .npz file and return metadata only."""
    try:
        import numpy as np
        if path.suffix == ".npy":
            arr = np.load(path, allow_pickle=False)
            return f"numpy array | shape={arr.shape} | dtype={arr.dtype}"
        else:
            data = np.load(path, allow_pickle=False)
            keys = list(data.keys())
            lines = [f"numpy archive with {len(keys)} arrays: {keys}"]
            for k in keys[:10]:
                arr = data[k]
                lines.append(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
            return "\n".join(lines)
    except ImportError:
        return "[numpy not available – skipping array metadata]"
    except Exception as exc:
        return f"[Could not read numpy file: {exc}]"


def _read_config_file(path: Path) -> str:
    """Read a JSON / YAML / TOML / INI config file."""
    try:
        if path.suffix == ".json":
            content = path.read_text(encoding="utf-8", errors="replace")
            # Pretty-print to validate and normalise
            try:
                obj = json.loads(content)
                content = json.dumps(obj, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                pass  # Return raw text if invalid JSON
        else:
            content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) > MAX_TEXT_CHARS_PER_FILE:
            content = content[:MAX_TEXT_CHARS_PER_FILE] + "\n... [truncated]"
        return content
    except Exception as exc:
        return f"[Could not read config: {exc}]"


def _read_pkl_metadata(path: Path) -> str:
    """Attempt to load a pickle and return type/shape metadata (safe mode)."""
    try:
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        type_name = type(obj).__name__
        extra = ""
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                extra = f", shape={obj.shape}, dtype={obj.dtype}"
        except ImportError:
            pass
        try:
            import pandas as pd
            if isinstance(obj, pd.DataFrame):
                extra = f", shape={obj.shape}, columns={list(obj.columns)}"
        except ImportError:
            pass
        return f"pickle object | type={type_name}{extra}"
    except Exception as exc:
        return f"[Could not load pickle metadata: {exc}]"


# ---------------------------------------------------------------------------
# Directory scanner
# ---------------------------------------------------------------------------

def _should_skip_dir(name: str) -> bool:
    """Return True for directories that should never be scanned."""
    skip_names = {
        ".git", ".venv", "venv", "__pycache__", ".mypy_cache",
        ".pytest_cache", "node_modules", ".tox", ".eggs", "dist", "build",
        ".idea", ".vscode",
    }
    return name in skip_names or name.startswith(".")


def scan_workspace(
    workspace_path: Path,
    max_depth: int = MAX_SCAN_DEPTH,
) -> list[dict]:
    """Recursively scan *workspace_path* and return a list of file records.

    Each record is a dict with keys:
        - ``rel_path``: str  – path relative to workspace_path
        - ``abs_path``: Path
        - ``ext``: str       – file extension (lower-case)
        - ``size_bytes``: int
        - ``content``: str   – extracted / summarised content

    Files larger than 10 MB are always skipped.

    Args:
        workspace_path: Root directory to scan.
        max_depth: Maximum recursion depth (0 = only the root level).

    Returns:
        List of file record dicts, sorted by relative path.
    """
    records: list[dict] = []

    def _recurse(current: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return

        for entry in entries:
            if entry.is_dir():
                if not _should_skip_dir(entry.name):
                    _recurse(entry, depth + 1)
            elif entry.is_file():
                ext = entry.suffix.lower()
                size = entry.stat().st_size

                # Skip very large files (> 10 MB)
                if size > 10 * 1024 * 1024:
                    logger.debug("Skipping large file: %s (%d bytes)", entry, size)
                    continue

                # Determine content based on extension
                if ext in BINARY_SKIP_EXTENSIONS and ext != ".pkl":
                    # Try pkl metadata, skip others silently
                    continue
                elif ext == ".pkl":
                    content = _read_pkl_metadata(entry)
                elif ext in CODE_EXTENSIONS:
                    content = _read_code_file(entry)
                elif ext in CSV_EXTENSIONS:
                    content = _read_csv_file(entry)
                elif ext in NUMPY_EXTENSIONS:
                    content = _read_numpy_file(entry)
                elif ext in CONFIG_EXTENSIONS:
                    content = _read_config_file(entry)
                elif ext in TEXT_EXTENSIONS or ext == "":
                    # Plain text or no extension
                    content = _read_text_file(entry)
                else:
                    # Unknown extension – try as text, skip on error
                    try:
                        raw = entry.read_bytes()
                        # Quick heuristic: if <30% non-ASCII bytes → treat as text
                        non_ascii = sum(1 for b in raw[:1024] if b > 127)
                        if non_ascii / max(len(raw[:1024]), 1) < 0.30:
                            content = _read_text_file(entry)
                        else:
                            continue  # Binary file, skip
                    except Exception:
                        continue

                rel_path = entry.relative_to(workspace_path)
                records.append(
                    {
                        "rel_path": str(rel_path),
                        "abs_path": entry,
                        "ext": ext,
                        "size_bytes": size,
                        "content": content,
                    }
                )

    _recurse(workspace_path, 0)
    return records


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_workspace_rag_context(
    workspace_path: Path,
    max_total_chars: int = MAX_TOTAL_RAG_CHARS,
) -> str:
    """Build a structured RAG context string from the user workspace.

    The context includes:
    1. A workspace directory tree.
    2. Per-file content blocks, ordered by file type priority.

    Args:
        workspace_path: Path to the user workspace directory.
        max_total_chars: Soft cap on total context size (characters).

    Returns:
        Multi-section string ready to inject into an LLM prompt.
        Returns an empty string if the directory is empty or does not exist.
    """
    if not workspace_path.exists() or not workspace_path.is_dir():
        return ""

    # Check if directory is effectively empty (only README.md)
    all_files = [
        p for p in workspace_path.rglob("*")
        if p.is_file() and p.name != "README.md"
        and not _should_skip_dir(p.parent.name)
    ]
    if not all_files:
        return ""

    records = scan_workspace(workspace_path)
    # Filter out README.md from the user workspace itself (it's our docs file)
    records = [
        r for r in records
        if not (r["rel_path"] == "README.md" and workspace_path.name == "user_workspace")
    ]

    if not records:
        return ""

    # Build directory tree summary
    tree_lines: list[str] = [f"📁 {workspace_path.name}/"]
    for r in records:
        parts = Path(r["rel_path"]).parts
        indent = "  " * (len(parts) - 1)
        tree_lines.append(f"{indent}  └── {parts[-1]}  ({r['size_bytes']:,} bytes)")
    tree_str = "\n".join(tree_lines)

    # Build per-file content blocks, respecting total char budget
    sections: list[str] = []
    total_chars = len(tree_str)

    # Prioritise: code > csv > config > text > numpy metadata
    priority_order = (
        CODE_EXTENSIONS
        | CSV_EXTENSIONS
        | CONFIG_EXTENSIONS
        | TEXT_EXTENSIONS
        | NUMPY_EXTENSIONS
    )

    def _priority(r: dict) -> int:
        ext = r["ext"]
        if ext in CODE_EXTENSIONS:
            return 0
        if ext in CSV_EXTENSIONS:
            return 1
        if ext in CONFIG_EXTENSIONS:
            return 2
        if ext in TEXT_EXTENSIONS:
            return 3
        return 4  # numpy / other

    sorted_records = sorted(records, key=_priority)

    for r in sorted_records:
        header = f"--- FILE: {r['rel_path']} ---"
        block = f"{header}\n{r['content']}\n"
        if total_chars + len(block) > max_total_chars:
            # Add a truncation notice and stop
            sections.append(
                f"--- [Further files omitted – context limit of "
                f"{max_total_chars:,} chars reached] ---"
            )
            break
        sections.append(block)
        total_chars += len(block)

    if not sections:
        return ""

    header_banner = (
        f"=== USER WORKSPACE FILES (AgenticRAG) ===\n"
        f"**USER-WORKSPACE PATH:** {workspace_path.resolve()}\n\n"
        f"The following files were found in the user-specified workspace.\n"
        f"Use their content as GROUND TRUTH for data paths, column names,\n"
        f"existing code, configuration values, and any domain knowledge.\n"
        f"IMPORTANT: All file references in this section are relative to the\n"
        f"user-workspace path above: [{workspace_path.resolve()}]\n\n"
        f"When writing code, prefer to load these files directly rather than\n"
        f"recreating the data from scratch.\n"
    )

    return (
        header_banner
        + "\n-- Workspace Directory Tree --\n"
        + tree_str
        + "\n\n-- File Contents --\n\n"
        + "\n".join(sections)
    )
