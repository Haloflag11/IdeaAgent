# User Workspace

This directory is the **default user workspace** for IdeaAgent.

Place any files you want the Agent to read and use as context here.
When you run IdeaAgent with `--workspace <path>` (or use this default folder),
the Agent will automatically scan all files in that directory and inject their
contents as **AgenticRAG context** into its planning and execution prompts.

## How It Works

1. Put your data files, notebooks, scripts, CSVs, JSON configs, text notes, etc.
   into this folder (or any folder you choose).
2. Run IdeaAgent and specify the workspace path:
   ```
   ideaagent --workspace ./user_workspace
   # or point to any other folder:
   ideaagent --workspace /path/to/my/project
   ```
3. The Agent will:
   - Scan all readable files in the workspace directory (recursively, up to 3 levels deep).
   - Read and summarise their content.
   - Inject that knowledge into the **plan generation** and **each execution step**
     so the Agent can refer to your existing data, code, and notes.

## Supported File Types

| Extension | How it is read |
|-----------|---------------|
| `.py`, `.ipynb`, `.r`, `.R` | Full source code |
| `.csv` | First 50 rows + column names + shape |
| `.json`, `.yaml`, `.yml`, `.toml`, `.cfg`, `.ini` | Full content |
| `.txt`, `.md`, `.rst`, `.log` | Full text (truncated at 5 000 chars per file) |
| `.npy`, `.npz`, `.pkl` | Metadata only (shape / dtype / keys) |
| Other binary / image files | Skipped |

## Tips

- Keep individual files reasonably small (< 1 MB) for best results.
- Use descriptive file names so the Agent can infer what each file contains.
- You can organise files in subdirectories – they will all be scanned.
- The workspace is **read-only** from the Agent's perspective; it never writes
  to your user workspace.
