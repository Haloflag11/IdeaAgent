
<div align="center">

![UI](./asset/cover.png)



<br/>
<br/>

![Github](https://img.shields.io/badge/github-Haloflag11-blue?logo=github&color=blue&link=https://github.com/Haloflag11) ![Windows](https://img.shields.io/badge/os-windows-orange?logo=windows) ![Last Update](https://img.shields.io/github/last-commit/Haloflag11/IdeaAgent?label=Last%20Update&style=flat-square) ![ML](https://img.shields.io/badge/machine%20learning-yellow?) ![DL](https://img.shields.io/badge/deep%20learning-purple) 


<strong> This is a very early edition，feel free to submit issues or contributions
</div>

# IdeaAgent

Experimental Agent for validating machine learning research ideas.



## Features

- Support for multiple research types: Deep Learning, Machine Learning, Agent
- AgentSkills-based skill system for extensible capabilities
- Real-time task status tracking and display
- Sandboxed execution using conda environments
- Loop detection to prevent infinite execution
- SQLlite for persistent storage
- MCP (Model Context Protocol) support
- CLI interface inspired by Claude Code

## Installation

```bash
# Clone or navigate to the project
cd IdeaAgent

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -e .

# Or using uv
uv sync
```

## Configuration

1. Copy the example environment file:
```bash
copy .env.example .env
```

2. Edit `.env` and set your API keys and configurations.

## Usage

```bash
# Start the agent
IdeaAgent

# Validate a skill
IdeaAgent validate ./skills/your-skill

# List available skills
IdeaAgent skills

#Individualy set workspace
IdeaAgent: /workspace ./user_workspace

# Run with specific idea
IdeaAgent: /run  "deep-learning"  "Your research idea here" --workspace ./user_workspace

Example： /run machine-learning Compare Linear Regression and Logistic Regression
```

## Project Structure

```
IdeaAgent/
├── src/ideaagent/          # Main source code
│   ├── __init__.py
│   ├── cli.py              # CLI interface
│   ├── models.py           # Data models
│   ├── database.py         # SQLite database integration
│   ├── llm.py              # LLM calling module
│   ├── config.py           # Configuration management
│   ├── prompts.py          # Prompt templates
│   ├── context.py          # Context management
│   ├── sandbox.py          # Sandboxed execution
│   ├── state.py            # Task state management
│   ├── loop_detector.py    # Loop detection
│   ├── mcp.py              # MCP (Model Context Protocol) support
│   ├── exceptions.py       # Custom exceptions
│   ├── skills/             # AgentSkills integration
│   │   ├── __init__.py
│   │   ├── manager.py      # Skill manager
│   │   └── errors.py       # Skill errors
│   └── utils/              # Utility modules
│       ├── __init__.py
│       ├── code_parser.py  # Code extraction & validation
│       ├── file_manager.py # File operations
│       ├── bash_executor.py# Bash command execution
│       ├── workspace.py    # Workspace management
│       ├── workspace_rag.py# AgenticRAG context builder
│       ├── stream_parser.py# Stream output parser
│       └── banner.py       # Banner utilities
├── skills/                 # Skill definitions
│   ├── deep-learning/      # Deep learning skill
│   ├── machine-learning/   # Machine learning skill
│   ├── agent/              # Agent skill
│   ├── data-preprocessing/ # Data preprocessing skill
│   ├── model-training/     # Model training skill
│   ├── experiment-tracking/# Experiment tracking skill
│   └── visualization/      # Visualization skill
├── tests/                  # Test files
├── user_workspace/         # User workspace (AgenticRAG)
├── .env.example            # Environment variables template
├── .gitignore
├── pyproject.toml
└── README.md
```

## Creating Skills

Skills follow the AgentSkills specification. See `skills/` directory for examples.

```bash
# Create a new skill directory
mkdir skills/my-skill

# Create SKILL.md with frontmatter
cat > skills/my-skill/SKILL.md << EOF
---
name: my-skill
description: What this skill does
---

# Skill Instructions

Detailed instructions here.
EOF

# Validate the skill
IdeaAgent validate ./skills/my-skill
```
## Star History

[![Star History Chart](https://api.star-history.com/image?repos=Haloflag11/IdeaAgent.git&type=date&legend=top-left)](https://www.star-history.com/?repos=Haloflag11%2FIdeaAgent.git&type=date&legend=top-left)

## License

Apache License 2.0
