"""CLI interface for IdeaAgent inspired by Claude Code."""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.text import Text
from rich.rule import Rule

from .config import settings, ensure_directories

# ── Logging setup ────────────────────────────────────────────────────────────
ensure_directories()
log_dir: Path = settings.log_dir

log_file = log_dir / f"ideaagent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("IdeaAgent")

console = Console()


# ── Main CLI class ────────────────────────────────────────────────────────────

class IdeaAgentCLI:
    """Main CLI application for IdeaAgent.

    Supports full dependency injection for all major components, which makes
    unit-testing straightforward – just pass mock objects at construction time.
    """

    def __init__(
        self,
        db=None,
        sandbox=None,
        llm=None,
        skill_manager=None,
        mcp_manager=None,
        loop_detector=None,
        user_workspace: Optional[Path] = None,
    ):
        """Initialise the CLI application.

        Args:
            db: Optional :class:`~ideaagent.database.Database` instance.
                Defaults to a freshly created ``Database()``.
            sandbox: Optional :class:`~ideaagent.sandbox.VenvSandbox` instance.
            llm: Optional :class:`~ideaagent.llm.LLMClient` instance.
            skill_manager: Optional :class:`~ideaagent.skills.manager.SkillManager`.
            mcp_manager: Optional :class:`~ideaagent.mcp.MCPManager`.
            loop_detector: Optional :class:`~ideaagent.loop_detector.LoopDetector`.
            user_workspace: Optional path to a user-specified workspace directory.
                When set, all readable files in the directory are scanned and
                injected as AgenticRAG context into every LLM call for this run.
                Only affects the current run – it is never persisted.
        """
        self.console = console
        self.width = self.console.width

        # ── User workspace (AgenticRAG) ───────────────────────────────────────
        self.user_workspace: Optional[Path] = None
        if user_workspace is not None:
            uw = Path(user_workspace).expanduser().resolve()
            if uw.exists() and uw.is_dir():
                self.user_workspace = uw
                self.console.print(
                    f"[bold green]User workspace:[/bold green] [dim]{uw}[/dim]"
                )
            else:
                self.console.print(
                    f"[yellow]Warning: --workspace path does not exist or is not a "
                    f"directory: {uw}[/yellow]"
                )

        # Lazy imports to avoid circular dependencies
        from .database import Database
        from .skills.manager import SkillManager
        from .llm import LLMClient
        from .sandbox import VenvSandbox
        from .mcp import MCPManager
        from .loop_detector import LoopDetector

        # Dependency injection – fall back to defaults when not provided
        self.db = db or Database()
        self.skills_root = Path.cwd() / "skills"
        self.skill_manager = skill_manager or SkillManager(self.skills_root)
        self.sandbox = sandbox or VenvSandbox(
            timeout=settings.execution_timeout,
            workspace=settings.workspace_root,
        )
        self.mcp_manager = mcp_manager or MCPManager()
        self.loop_detector = loop_detector or LoopDetector(
            max_loop_count=settings.max_loop_count
        )

        # LLM – may be None if API key is missing
        if llm is not None:
            self.llm = llm
        else:
            try:
                self.llm = LLMClient()
            except ValueError as exc:
                self.llm = None
                self.console.print(f"[yellow]Warning: {exc}[/yellow]")

    # ── Banner & welcome ──────────────────────────────────────────────────────

    def print_banner(self) -> None:
        """Print application banner."""
        banner = Text()
        banner.append("IdeaAgent ", style="bold magenta")
        banner.append("v0.1.0\n", style="dim")
        banner.append("Experimental Agent for ML Research\n", style="dim italic")
        banner.append("Type ", style="dim")
        banner.append("? ", style="bold cyan")
        banner.append("for shortcuts", style="dim")
        self.console.print(Panel(banner, border_style="magenta"))

    def print_welcome(self) -> None:
        """Print welcome message with status grid."""
        self.print_banner()
        status_info = self.get_status_info()

        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="cyan", justify="right")
        grid.add_column(style="white")

        for label, value in status_info.items():
            grid.add_row(f"{label}:", value)

        self.console.print(Panel(grid, title="Status", border_style="green"))
        self.console.print()

    def get_status_info(self) -> dict:
        """Return status information for the welcome display."""
        info: dict = {}

        # Database
        if self.db.is_connected():
            stats = self.db.get_task_statistics()
            info["Database"] = f"[green]Connected[/green] ({stats['total']} tasks)"
        else:
            info["Database"] = "[yellow]Disconnected (running in memory mode)[/yellow]"

        # Skills
        skills = self.skill_manager.discover_skills()
        info["Skills"] = f"{len(skills)} available"

        # LLM
        if self.llm:
            info["LLM"] = f"[green]Configured[/green] ({self.llm.model})"
        else:
            info["LLM"] = "[red]Not configured[/red] (set OPENAI_API_KEY)"

        # MCP
        mcp_status = self.mcp_manager.client.get_status()
        if mcp_status["enabled"] and mcp_status["servers_count"] > 0:
            info["MCP"] = f"[green]Enabled[/green] ({mcp_status['servers_count']} servers)"
        else:
            info["MCP"] = "[dim]Disabled[/dim]"

        # User workspace (AgenticRAG)
        if self.user_workspace is not None:
            try:
                file_count = sum(
                    1 for p in self.user_workspace.rglob("*") if p.is_file()
                )
                info["Workspace"] = (
                    f"[green]{self.user_workspace.name}[/green] "
                    f"[dim]({file_count} files, AgenticRAG active)[/dim]"
                )
            except Exception:
                info["Workspace"] = f"[green]{self.user_workspace}[/green]"
        else:
            info["Workspace"] = "[dim]None (use --workspace or /workspace)[/dim]"

        return info

    # ── Help / list / details ─────────────────────────────────────────────────

    def print_help(self) -> None:
        """Print help information."""
        help_text = """
# IdeaAgent Commands

## Main Commands
- `/help` - Show this help message
- `/status` - Show current status
- `/quit` or `/exit` - Exit the application

## Task Commands
- `/new` - Create a new task (interactive)
- `/run <type> <prompt>` - Run a task directly
  - Types: `deep-learning`, `machine-learning`, `agent`
- `/list` - List all tasks
- `/show <id>` - Show task details
- `/delete <id>` - Delete a task

## Workspace Commands (AgenticRAG)
- `/workspace` - Show current user workspace
- `/workspace <path>` - Set user workspace for this session
- `/workspace clear` - Clear the current user workspace

## Skill Commands
- `/skills` - List available skills
- `/validate <path>` - Validate a skill directory

## System Commands
- `/config` - Show configuration
- `/clear` - Clear screen

## Startup Flag
- `ideaagent --workspace <path>` - Launch with a user workspace pre-loaded
"""
        self.console.print(Markdown(help_text))

    def print_task_list(self) -> None:
        """Print list of all tasks."""
        tasks = self.db.get_all_tasks(limit=20)

        if not tasks:
            self.console.print("[dim]No tasks found.[/dim]")
            return

        table = Table(title="Tasks", border_style="blue")
        table.add_column("ID", style="cyan", width=8)
        table.add_column("Type", style="magenta")
        table.add_column("Description", style="white", max_width=50)
        table.add_column("Status", style="green")
        table.add_column("Created", style="dim", width=12)

        for task in tasks:
            status_style = {
                "pending": "yellow",
                "planning": "cyan",
                "running": "green",
                "completed": "bold green",
                "failed": "red",
                "stopped": "dim",
            }.get(task.status.value, "white")

            table.add_row(
                task.id[:8],
                task.research_type.value,
                task.idea_description[:50] + "..."
                if len(task.idea_description) > 50
                else task.idea_description,
                f"[{status_style}]{task.status.value}[/{status_style}]",
                task.created_at.strftime("%Y-%m-%d"),
            )

        self.console.print(table)

    def print_task_details(self, task_id: str) -> None:
        """Print detailed task information."""
        task = self.db.get_task(task_id)

        if not task:
            self.console.print(f"[red]Task not found: {task_id}[/red]")
            return

        self.console.print(
            Panel(f"Task: {task.id}", title="Task Details", border_style="blue")
        )

        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="cyan", justify="right")
        grid.add_column(style="white")

        grid.add_row("ID:", task.id)
        grid.add_row("Type:", task.research_type.value)
        grid.add_row("Status:", f"[green]{task.status.value}[/green]")
        grid.add_row("Description:", task.idea_description)
        grid.add_row("Loop Count:", str(task.loop_count))
        grid.add_row("Created:", task.created_at.strftime("%Y-%m-%d %H:%M:%S"))
        grid.add_row("Updated:", task.updated_at.strftime("%Y-%m-%d %H:%M:%S"))

        if task.plan:
            grid.add_row("\nPlan Title:", task.plan.title)
            grid.add_row("Plan Description:", task.plan.description)
            grid.add_row("Total Steps:", str(len(task.plan.steps)))
            grid.add_row("Progress:", f"{self._calculate_progress(task)}%")

        self.console.print(grid)

        if task.execution_results:
            self.console.print("\n[bold]Execution Results:[/bold]")
            for result in task.execution_results:
                status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
                self.console.print(
                    f"  {status} Step {result.step_number}: {result.output[:100]}..."
                )

    def _calculate_progress(self, task) -> float:
        """Calculate task progress percentage."""
        if not task.plan or len(task.plan.steps) == 0:
            return 0.0
        return round((len(task.execution_results) / len(task.plan.steps)) * 100, 1)

    # ── Skills ────────────────────────────────────────────────────────────────

    def print_skills(self) -> None:
        """Print available skills."""
        skills = self.skill_manager.discover_skills()

        if not skills:
            self.console.print("[dim]No skills found in ./skills directory.[/dim]")
            self.console.print("\n[dim]Create your first skill:[/dim]")
            self.console.print("  [cyan]mkdir skills/my-skill[/cyan]")
            self.console.print("  [cyan]cat > skills/my-skill/SKILL.md[/cyan]")
            return

        table = Table(title="Available Skills", border_style="magenta")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="white", max_width=60)
        table.add_column("Skills Needed", style="dim")

        for skill in skills:
            table.add_row(
                skill.name,
                skill.description,
                ", ".join(skill.metadata.get("skills", []))
                if isinstance(skill.metadata.get("skills"), list)
                else "-",
            )

        self.console.print(table)

    def validate_skill(self, skill_path: str) -> None:
        """Validate a skill directory."""
        path = Path(skill_path)

        if not path.exists():
            self.console.print(f"[red]Path does not exist: {path}[/red]")
            return

        errors = self.skill_manager.validate(path)

        if errors:
            self.console.print(f"[red]Validation failed for {path}:[/red]")
            for error in errors:
                self.console.print(f"  [red]- {error}[/red]")
        else:
            self.console.print(f"[green]Valid skill: {path}[/green]")
            props = self.skill_manager.read_properties(path)
            self.console.print(f"  Name: [cyan]{props.name}[/cyan]")
            self.console.print(f"  Description: {props.description}")

    # ── Context & code helpers ────────────────────────────────────────────────

    def _format_skills_as_xml(self, skills) -> str:
        """Format skills as XML string for LLM consumption.
        
        Args:
            skills: List of SkillProperties objects
            
        Returns:
            XML-formatted string of available skills
        """
        if not skills:
            return "<skills>No skills available</skills>"
        
        xml_parts = ["<skills>"]
        for skill in skills:
            xml_parts.append(f"  <skill>")
            xml_parts.append(f"    <name>{skill.name}</name>")
            xml_parts.append(f"    <description>{skill.description}</description>")
            if skill.metadata:
                for key, value in skill.metadata.items():
                    xml_parts.append(f"    <{key}>{value}</{key}>")
            xml_parts.append(f"  </skill>")
        xml_parts.append("</skills>")
        
        return "\n".join(xml_parts)
    
    def _plan_to_string(self, plan) -> str:
        """Convert an ExperimentPlan object to a string representation.
        
        Args:
            plan: ExperimentPlan object
            
        Returns:
            String representation of the plan for LLM consumption
        """
        lines = [
            f"Title: {plan.title}",
            f"Description: {plan.description}",
            f"Estimated Total Time: {plan.estimated_total_time} minutes",
            f"Skills Needed: {', '.join(plan.skills_needed) if plan.skills_needed else 'None'}",
            "",
            "Steps:",
        ]
        
        for step in plan.steps:
            skill_info = f" (Skill: {step.skill_required})" if step.skill_required else ""
            duration_info = f" ({step.estimated_duration}min)" if step.estimated_duration else ""
            lines.append(
                f"  {step.step_number}. {step.description}{skill_info}{duration_info}"
            )
        
        return "\n".join(lines)

    def _build_execution_context(
        self, execution_context: list, current_step: int
    ) -> str:
        """Build execution context string from previous steps (legacy wrapper).

        Delegates to the :mod:`ideaagent.utils.workspace` utility so that the
        richer version (with installed packages, files created, etc.) is used
        automatically when those fields are present in the context dicts.
        """
        from .utils.workspace import build_execution_context

        return build_execution_context(execution_context, current_step)

    def _get_workspace_structure(self, workspace_dir: Path, max_depth: int = 3) -> str:
        """Get the directory structure of the workspace.

        Args:
            workspace_dir: Path to the workspace directory.
            max_depth: Maximum depth to traverse.

        Returns:
            String representation of the directory structure.
        """
        from .utils.workspace import get_workspace_structure

        return get_workspace_structure(workspace_dir, max_depth)

    def _extract_python_code(self, text: str) -> str:
        """Extract Python code from markdown code blocks.

        Delegates to the robust :func:`~ideaagent.utils.code_parser.extract_python_code`
        implementation which handles multiple blocks, de-duplicates imports, and
        validates syntax.

        Args:
            text: Text containing markdown code blocks.

        Returns:
            Extracted Python code or empty string if none found.
        """
        from .utils.code_parser import extract_python_code

        return extract_python_code(text)

    def _parse_response_sections(self, text: str) -> list[tuple[str, str]]:
        """Parse response into alternating text and code sections.

        Args:
            text: Raw response text potentially containing ```python ... ``` blocks.

        Returns:
            List of (section_type, content) tuples where section_type is either
            'text' or 'code'.
        """
        import re
        sections: list[tuple[str, str]] = []
        
        # Pattern to match ```python ... ``` or ``` ... ``` blocks
        pattern = r'```(?:python|py)?\s*\n(.*?)```'
        
        last_end = 0
        for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            # Get text before the code block
            before_text = text[last_end:match.start()].strip()
            if before_text:
                sections.append(('text', before_text))
            
            # Get the code block content
            code_content = match.group(1).strip()
            if code_content:
                sections.append(('code', code_content))
            
            last_end = match.end()
        
        # Add any remaining text after the last code block
        remaining = text[last_end:].strip()
        if remaining:
            sections.append(('text', remaining))
        
        return sections

    def _display_response_sections(self, text: str, title: str = "Response", border_style: str = "green") -> None:
        """Display response with Python code blocks rendered as syntax-highlighted code.

        Args:
            text: Raw response text potentially containing ```python ... ``` blocks.
            title: Title for the panel/wrapper
            border_style: Color style for borders
        """
        if not text:
            self.console.print(f"[dim]{title}: (empty)[/dim]")
            return
        
        sections = self._parse_response_sections(text)
        
        if not sections:
            self.console.print(f"[dim]{title}: (empty)[/dim]")
            return
        
        # If only one text section, display as plain text panel
        if len(sections) == 1 and sections[0][0] == 'text':
            display_text = text[:2000] + ("..." if len(text) > 2000 else "")
            self.console.print(
                Panel(display_text, title=f"[bold]{title}[/bold]", border_style=border_style)
            )
            return
        
        # Multiple sections or contains code - display each section
        self.console.print(f"\n[bold cyan]{title}:[/bold cyan]")
        
        for i, (section_type, content) in enumerate(sections):
            if section_type == 'code':
                # Render with syntax highlighting
                # Truncate very long code blocks for display
                if len(content) > 8000:
                    content = content[:8000] + "\n... [truncated]"
                
                syntax = Syntax(
                    content,
                    lexer="python",
                    theme="monokai",
                    line_numbers=True,
                    word_wrap=True,
                )
                self.console.print(Panel(syntax, title="[bold green]Python Code[/bold green]", border_style="yellow"))
            else:
                # Plain text section - truncate if too long
                display_content = content[:2000] + ("..." if len(content) > 2000 else "")
                self.console.print(
                    Panel(display_content, title="[bold yellow]Text[/bold yellow]", border_style="blue", padding=(0, 1))
                )

    def _build_skill_system_prompt(self, skill) -> str:
        """Build a skill-aware system prompt injection string.

        Args:
            skill: :class:`~ideaagent.skills.manager.SkillProperties` object.

        Returns:
            Formatted string to prepend to system messages.
        """
        skill_path = self.skill_manager.get_skill_path(skill.name)
        if skill_path is None:
            return ""
        skill_md = self.skill_manager.find_skill_md(skill_path)
        if skill_md is None:
            return ""
        content = skill_md.read_text(encoding="utf-8")
        return (
            f"\n\n=== ACTIVE SKILL: {skill.name} ===\n"
            f"{content}\n"
            f"=== END SKILL ===\n"
        )

    def _thinking_callback_factory(self, step_num: int, iteration: int):
        """Create a callback function for streaming thinking output.

        Args:
            step_num: Current step number
            iteration: Current iteration attempt number

        Returns:
            Callback function for streaming thinking content
        """
        buffer = []
        
        def callback(chunk_type: str, content: str):
            if chunk_type == 'thinking':
                buffer.append(content)
                # Stream thinking output in real-time
                self.console.print(
                    f"[cyan]{content}[/cyan]",
                    end="",
                )
                self.console.file.flush()
        
        return callback

    # ── Agent loop (NEW ACTION-BASED EXECUTION) ──────────────────────────────

    async def _execute_step_with_agent_loop(
        self,
        step,
        full_context: str,
        skill_instructions: Optional[str],
        workspace_dir: Path,
        installed_packages: list[str],
    ) -> tuple[bool, str, str, str]:
        """Execute a single step using the new action-based Agentic Loop.

        The loop works as follows:
          1. Generate response with actions (first iteration only).
          2. Parse all actions from the response.
          3. Execute each action (BASH, PYTHON, WRITE_FILE, READ_FILE, MKDIR).
          4. Collect all outputs.
          5. Pass outputs to LLM for judgment.
          6. LLM decides: "success" or "fix".
          7. Repeat until LLM says success or max iterations reached.

        Args:
            step: ExperimentStep to execute.
            full_context: Rich context string (workspace + history).
            skill_instructions: Full SKILL.md text if a skill is required.
            workspace_dir: Task workspace directory.
            installed_packages: Mutable list tracking packages installed so far.

        Returns:
            ``(success, output, error, final_code)``
        """
        from .utils.code_parser import parse_agent_actions, ActionType
        from .utils.file_manager import FileManager
        from .utils.bash_executor import BashExecutor

        research_type = getattr(self, "_current_research_type", None)
        max_iterations = settings.max_agent_iterations
        
        # Initialize executors
        file_manager = FileManager(workspace_dir)
        bash_executor = BashExecutor(timeout=settings.execution_timeout)
        
        last_output = ""
        last_error = ""
        last_returncode = -1
        raw_response = ""

        for iteration in range(1, max_iterations + 1):
            self.console.print()
            self.console.print(
                Rule(
                    f"[bold yellow]Agent Loop Attempt {iteration}/{max_iterations}[/bold yellow]",
                    style="yellow",
                )
            )
            logger.info(
                "Agent loop attempt %d/%d for step %d",
                iteration, max_iterations, step.step_number,
            )

            # ── 1. First iteration: generate fresh response ──────────────────
            if iteration == 1:
                use_streaming = hasattr(self.llm, 'stream_execute_step_with_thinking')
                
                if use_streaming:
                    self.console.print(
                        "\n[bold yellow]Generating response with streaming thinking...[/bold yellow]"
                    )
                    
                    thinking_cb = self._thinking_callback_factory(step.step_number, iteration)
                    
                    self.console.print("\n[bold yellow]Streaming thinking output:[/bold yellow]")
                    raw_response, _ = self.llm.stream_execute_step_with_thinking(
                        research_type, 
                        step.description, 
                        full_context, 
                        skill_instructions,
                        callback=thinking_cb,
                    )
                    
                    self.console.print()
                else:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=self.console,
                    ) as progress:
                        t = progress.add_task("[cyan]Generating response...", total=None)
                        raw_response = self.llm.execute_step(
                            research_type, step.description, full_context, skill_instructions,
                        )
                        progress.update(t, completed=True)

                self._display_response_sections(raw_response, "Generated Response", border_style="green")

            # ── 2. Parse actions ─────────────────────────────────────────────
            actions = parse_agent_actions(raw_response)
            
            if not actions:
                logger.warning("No actions found in response")
                last_error = "No actions found in LLM response"
                last_returncode = 1
            else:
                # ── 3. Execute each action ───────────────────────────────────
                action_outputs = []
                has_failure = False
                
                self.console.print(f"\n[bold cyan]Executing {len(actions)} action(s)...[/bold cyan]")
                
                for action in actions:
                    self.console.print(f"\n  [dim]Action: {action.action_type.value}[/dim]")
                    
                    try:
                        if action.action_type == ActionType.THINKING:
                            # Thinking is just logged, not executed
                            self.console.print(f"    [cyan]Thinking: {action.content[:100]}...[/cyan]")
                            action_outputs.append(f"[THINKING] {action.content}")
                            
                        elif action.action_type == ActionType.MKDIR:
                            path = action.params.get("path", "") if action.params else ""
                            if not path and action.content:
                                path = action.content
                            result = file_manager.mkdir(path)
                            if result["success"]:
                                self.console.print(f"    [green]{result['message']}[/green]")
                                action_outputs.append(f"[MKDIR OK] {result['message']}")
                            else:
                                self.console.print(f"    [red]{result['error']}[/red]")
                                action_outputs.append(f"[MKDIR FAIL] {result['error']}")
                                has_failure = True
                                
                        elif action.action_type == ActionType.WRITE_FILE:
                            path = action.params.get("path", "") if action.params else ""
                            if not path:
                                self.console.print(f"    [red]WRITE_FILE missing path attribute[/red]")
                                action_outputs.append("[WRITE_FILE FAIL] Missing path attribute")
                                continue
                            content = action.content
                            result = file_manager.write_file(path, content) 
                            if result["success"]:
                                self.console.print(f"    [green]{result['message']}[/green]")
                                action_outputs.append(f"[WRITE_FILE OK] {result['message']} ({result['bytes_written']} bytes)")
                            else:
                                self.console.print(f"    [red]{result['error']}[/red]")
                                action_outputs.append(f"[WRITE_FILE FAIL] {result['error']}")
                                has_failure = True
                                
                        elif action.action_type == ActionType.READ_FILE:
                            path = action.content
                            result = file_manager.read_file(path)
                            if result["success"]:
                                content_preview = result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
                                self.console.print(f"    [green]Read {path} ({result['bytes_read']} bytes)[/green]")
                                action_outputs.append(f"[READ_FILE OK] {path}:\n{content_preview}")
                            else:
                                self.console.print(f"    [red]{result['error']}[/red]")
                                action_outputs.append(f"[READ_FILE FAIL] {result['error']}")
                                has_failure = True
                                
                        elif action.action_type == ActionType.BASH:
                            command = action.content
                            self.console.print(f"    [yellow]Running: {command}[/yellow]")
                            returncode, stdout, stderr = bash_executor.run(
                                command=command,
                                cwd=workspace_dir,
                                timeout=settings.execution_timeout,
                                realtime_output=True,
                            )
                            action_outputs.append(f"[BASH] {command}\n[STDOUT]\n{stdout}\n[STDERR]\n{stderr}\n[RETURN CODE] {returncode}")
                            if returncode != 0:
                                has_failure = True
                                last_error = stderr
                            else:
                                last_output = stdout
                            last_returncode = returncode
                            
                        elif action.action_type == ActionType.PYTHON:
                            code = action.content
                            self.console.print(f"    [yellow]Executing Python code ({len(code)} chars)[/yellow]")
                            returncode, stdout, stderr = bash_executor.run_python(
                                script=code,
                                cwd=workspace_dir,
                                timeout=settings.execution_timeout,
                                realtime_output=True,
                            )
                            action_outputs.append(f"[PYTHON]\n[STDOUT]\n{stdout}\n[STDERR]\n{stderr}\n[RETURN CODE] {returncode}")
                            if returncode != 0:
                                has_failure = True
                                last_error = stderr
                            else:
                                last_output = stdout
                            last_returncode = returncode
                            
                        elif action.action_type == ActionType.TASK_COMPLETE:
                            message = action.content
                            self.console.print(f"    [bold green]Task Complete: {message}[/bold green]")
                            action_outputs.append(f"[TASK_COMPLETE] {message}")
                            
                        else:
                            self.console.print(f"    [red]Unknown action type: {action.action_type}[/red]")
                            action_outputs.append(f"[UNKNOWN] {action.action_type.value}")
                            has_failure = True
                            
                    except Exception as e:
                        logger.error(f"Action {action.action_type.value} failed: {e}")
                        self.console.print(f"    [red]Error: {e}[/red]")
                        action_outputs.append(f"[ERROR] {action.action_type.value}: {e}")
                        has_failure = True
                
                # Combine all outputs
                if action_outputs:
                    combined_output = "\n".join(action_outputs)
                    if last_output and last_output not in combined_output:
                        last_output = combined_output + "\n\n[PREVIOUS OUTPUT]\n" + last_output
                    else:
                         last_output = combined_output

            # ── 4. LLM judges the output ─────────────────────────────────────
            self.console.print("\n[bold cyan]Asking LLM to judge execution...[/bold cyan]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                t = progress.add_task("[cyan]LLM judging...", total=None)
                judgment = self.llm.judge_and_fix(
                    research_type=research_type,
                    step_description=step.description,
                    code=raw_response,
                    stdout=last_output,
                    stderr=last_error,
                    returncode=last_returncode,
                    context=full_context,
                    skill_instructions=skill_instructions,
                    attempt=iteration,
                    max_attempts=max_iterations,
                )
                progress.update(t, completed=True)

            if judgment.get("status") == "success":
                logger.info("Step %d: LLM judged SUCCESS on attempt %d",
                            step.step_number, iteration)
                self.console.print("[bold green][OK] LLM confirmed: step succeeded![/bold green]")
                return True, last_output, last_error, raw_response

            # LLM says fix needed
            reason = judgment.get("reason", "unknown")
            fixed_response = judgment.get("code", "")
            logger.info("Step %d: LLM says fix needed (attempt %d): %s",
                        step.step_number, iteration, reason)
            self.console.print(
                f"\n[bold red]LLM diagnosis:[/bold red] {reason}"
            )

            if not fixed_response:
                logger.warning("LLM returned fix status but no response; stopping.")
                break

            raw_response = fixed_response
            self._display_response_sections(raw_response, "Fixed Response", border_style="yellow")

        # All attempts exhausted
        logger.error("Step %d failed after %d attempts", step.step_number, max_iterations)
        return False, last_output, last_error, raw_response

    @staticmethod
    def _filter_pip_noise(text: str) -> str:
        """Strip pip [notice] and similar housekeeping lines from output."""
        if not text:
            return ""
        noise_prefixes = ("[notice]", "Notice:", "WARNING: pip", "DEPRECATION:")
        filtered = [
            line for line in text.splitlines()
            if not any(line.strip().startswith(p) for p in noise_prefixes)
        ]
        return "\n".join(filtered).strip()

    def _display_sandbox_output(
        self, success: bool, output: str, error: str
    ) -> None:
        """Render sandbox stdout/stderr in the console.

        * stdout and stderr are shown in **separate** panels so they are easy
          to distinguish at a glance.
        * Pip ``[notice]`` / ``WARNING: pip`` housekeeping lines are stripped
          from stderr so they don't clutter the error panel.
        * Each panel is capped at 3 000 / 2 000 characters respectively to
          avoid flooding the terminal with huge traces.

        Args:
            success: Whether the execution succeeded.
            output: stdout captured from the sandbox.
            error: stderr captured from the sandbox.
        """
        clean_out = (output or "").strip()
        clean_err = self._filter_pip_noise(error or "")

        if not clean_out and not clean_err:
            return

        self.console.print()
        self.console.print(
            Rule(
                "[bold]Execution Output[/bold]",
                style="green" if success else "red",
            )
        )

        if clean_out:
            display_out = (
                clean_out[:3000] + "\n[dim]…output truncated[/dim]"
                if len(clean_out) > 3000
                else clean_out
            )
            self.console.print(
                Panel(
                    display_out,
                    title="[bold green]stdout[/bold green]"
                    if success
                    else "[bold yellow]stdout[/bold yellow]",
                    border_style="green" if success else "yellow",
                    padding=(0, 1),
                )
            )

        if clean_err:
            display_err = (
                clean_err[:2000] + "\n[dim]…stderr truncated[/dim]"
                if len(clean_err) > 2000
                else clean_err
            )
            self.console.print(
                Panel(
                    display_err,
                    title="[bold red]stderr[/bold red]",
                    border_style="red",
                    padding=(0, 1),
                )
            )

        self.console.print()

    # ── Main plan execution ───────────────────────────────────────────────────

    async def _execute_plan(
        self,
        state_manager,
        plan,
        idea: str,
        effective_workspace: Optional[Path] = None,
    ) -> None:
        """Execute the approved plan step by step.

        This is the main execution loop.  For each step it:
        1. Loads optional skill instructions.
        2. Builds a rich context (workspace structure + history + environment).
        3. Calls :meth:`_execute_step_with_agent_loop` which runs the inner
           agent loop (generate → validate → execute → fix & retry).
        4. Records the result and persists the task state.

        Args:
            state_manager: :class:`~ideaagent.state.TaskStateManager` for the
                current task.
            plan: :class:`~ideaagent.models.ExperimentPlan` to execute.
            idea: Original research idea description.
            effective_workspace: The resolved user workspace for **this run only**.
                Passed down from :meth:`run_task_interactive` so the sandbox
                workspace creation logic (``self.sandbox.create_task_workspace``)
                is never affected – only the AgenticRAG context injection uses it.
        """
        import time
        from .models import ExecutionResult
        from .utils.workspace import build_rich_context

        logger.info("Starting execution of plan with %d steps", len(plan.steps))

        # Store research type so the agent-loop helper can access it
        self._current_research_type = state_manager.task.research_type

        # Create task workspace
        task_name = (
            state_manager.task.idea_description[:50]
            if state_manager.task.idea_description
            else f"task_{state_manager.task.id}"
        )
        workspace_dir = self.sandbox.create_task_workspace(task_name)
        logger.info("Created task workspace: %s", workspace_dir)
        self.console.print(
            f"\n[bold green]📁 Task Workspace:[/bold green] [dim]{workspace_dir}[/dim]"
        )

        # Running state
        execution_context: list[dict] = []
        installed_packages: list[str] = []

        for step in plan.steps:
            current_step_num = state_manager.get_current_step()

            logger.info(
                "Step %d/%d: %s", current_step_num, len(plan.steps), step.description
            )

            # ── Step header ──────────────────────────────────────────────────
            self.console.print()
            self.console.print(
                Rule(
                    f"[bold cyan]Step {current_step_num}/{len(plan.steps)}[/bold cyan]",
                    style="cyan",
                )
            )
            self.console.print(f"  [bold white]{step.description}[/bold white]")

            # ── Skill loading (Issue 2.1 fix) ─────────────────────────────
            skill_instructions: Optional[str] = None
            if step.skill_required:
                logger.info("Using skill: %s", step.skill_required)
                self.console.print(
                    f"\n  [dim]Using skill: {step.skill_required}[/dim]"
                )
                skill_obj = self.skill_manager.get_skill(step.skill_required)
                if skill_obj:
                    # ── Pre-install skill requirements ─────────────────────
                    try:
                        pkgs_installed = self.skill_manager.install_skill_requirements(
                            step.skill_required, self.sandbox
                        )
                        if pkgs_installed:
                            logger.info(
                                "Pre-installed skill requirements for '%s': %s",
                                step.skill_required,
                                pkgs_installed,
                            )
                            self.console.print(
                                f"  [dim]Installed skill requirements: "
                                f"{', '.join(pkgs_installed)}[/dim]"
                            )
                            installed_packages.extend(pkgs_installed)
                    except Exception as req_exc:
                        logger.warning(
                            "Could not install requirements for skill '%s': %s",
                            step.skill_required,
                            req_exc,
                        )
                        self.console.print(
                            f"  [yellow]Warning: skill requirements install failed: "
                            f"{req_exc}[/yellow]"
                        )

                    skill_instructions = self._build_skill_system_prompt(skill_obj)
                    logger.info(
                        "Loaded skill '%s' with %d chars of instructions",
                        step.skill_required,
                        len(skill_instructions),
                    )
                else:
                    logger.warning("Skill '%s' not found", step.skill_required)
                    self.console.print(
                        f"  [yellow]⚠ Skill '{step.skill_required}' not found – "
                        "continuing without it.[/yellow]"
                    )

            # ── Rich context building (Issue 3.1 fix) ─────────────────────
            full_context = build_rich_context(
                workspace_dir=workspace_dir,
                execution_context=execution_context,
                current_step=step.step_number,
                installed_packages=installed_packages,
                user_workspace_path=effective_workspace,
            )

            self.console.print(
                "\n[bold yellow]📞 Calling LLM for execution instructions...[/bold yellow]"
            )
            logger.info("Calling LLM for execution instructions")

            start_time = time.time()

            try:
                if not self.llm:
                    self.console.print("[red]LLM not configured[/red]")
                    logger.error("LLM client not configured")
                    success, output, error, final_code = (
                        False, "", "LLM client not initialized", ""
                    )
                else:
                    # ── Agent loop (Issue 1.1 fix) ─────────────────────────
                    success, output, error, final_code = (
                        await self._execute_step_with_agent_loop(
                            step=step,
                            full_context=full_context,
                            skill_instructions=skill_instructions,
                            workspace_dir=workspace_dir,
                            installed_packages=installed_packages,
                        )
                    )

            except Exception as exc:
                success, output, error, final_code = False, "", str(exc), ""
                logger.error("Error during step execution: %s", exc, exc_info=True)
                self.console.print(f"\n[red]Error during execution: {exc}[/red]")

            execution_time = time.time() - start_time

            # ── Record result ────────────────────────────────────────────────
            result = ExecutionResult(
                step_number=step.step_number,
                success=success,
                output=output,
                error=error,
                execution_time=execution_time,
            )
            state_manager.add_execution_result(result)
            self.db.save_task(state_manager.task)

            # ── Update execution context for next step (Issue 3.1 fix) ─────
            execution_context.append(
                {
                    "step_number": step.step_number,
                    "description": step.description,
                    "success": success,
                    "output": output,
                    "error": error,
                    "packages_installed": list(installed_packages),
                    "files_created": [
                        str(p.name)
                        for p in workspace_dir.iterdir()
                        if p.is_file() and p.name != "script.py"
                    ]
                    if workspace_dir.exists()
                    else [],
                }
            )

            # ── Step summary ─────────────────────────────────────────────────
            if success:
                self.console.print(
                    f"\n  [green]✓ Step Completed[/green] "
                    f"(Execution time: {execution_time:.2f}s)"
                )
                logger.info(
                    "Step %d completed in %.2fs", step.step_number, execution_time
                )
            else:
                self.console.print(
                    f"\n  [red]✗ Step Failed: {error}[/red]"
                )
                logger.error("Step %d failed: %s", step.step_number, error)
                break

            # ── Loop detection ───────────────────────────────────────────────
            should_stop, reason = self.loop_detector.analyze(state_manager.task)
            if should_stop:
                logger.warning("Loop detected: %s", reason)
                self.console.print(f"\n[yellow]⚠ Loop detected: {reason}[/yellow]")
                state_manager.stop_task(reason)
                self.db.save_task(state_manager.task)
                break

        # ── Task completion ──────────────────────────────────────────────────
        if state_manager.task.status.value == "running":
            # Only mark completed when EVERY planned step actually succeeded.
            # A step failure causes an early `break`, so either
            # execution_results is shorter than plan.steps, or its last entry
            # has success=False.
            all_succeeded = (
                len(state_manager.task.execution_results) == len(plan.steps)
                and all(r.success for r in state_manager.task.execution_results)
            )
            if all_succeeded:
                state_manager.complete_task()
                self.db.save_task(state_manager.task)
                logger.info("Task completed successfully")
                self.console.print(
                    f"\n[bold green]{'=' * self.width}[/bold green]"
                )
                self.console.print(
                    "[bold green]Task completed successfully![/bold green]"
                )
                self.console.print(
                    f"[bold green]{'=' * self.width}[/bold green]"
                )
            else:
                last_result = (
                    state_manager.task.execution_results[-1]
                    if state_manager.task.execution_results
                    else None
                )
                fail_reason = (
                    f"Step {last_result.step_number} failed after all retries"
                    if last_result
                    else "No steps were executed"
                )
                state_manager.fail_task(fail_reason)
                self.db.save_task(state_manager.task)
                logger.error("Task failed: %s", fail_reason)
                self.console.print(
                    f"\n[bold red]{'=' * self.width}[/bold red]"
                )
                self.console.print(
                    f"[bold red]Task failed: {fail_reason}[/bold red]"
                )
                self.console.print(
                    f"[bold red]{'=' * self.width}[/bold red]"
                )

    # ── Interactive task runner ───────────────────────────────────────────────

    async def run_task_interactive(
        self,
        research_type: str,
        idea: str,
        run_workspace: Optional[Path] = None,
    ) -> None:
        """Run a task interactively with progress display.

        Args:
            research_type: One of ``deep-learning``, ``machine-learning``, ``agent``.
            idea: Natural language description of the research idea.
            run_workspace: Optional path to a user workspace **for this run only**.
                Overrides ``self.user_workspace`` for the duration of this single run.
                When the run finishes (success, failure, or exception), this value
                is NOT stored back to ``self.user_workspace`` – the next run starts
                fresh.  If ``None``, falls back to ``self.user_workspace`` (the
                session-level default set via ``--workspace`` at launch).
        """
        # Resolve the effective workspace for this run only.
        # Priority: run-level > session-level (--workspace flag) > None
        effective_workspace: Optional[Path] = run_workspace or self.user_workspace
        if effective_workspace is not None:
            self.console.print(
                f"\n[bold green]AgenticRAG workspace (this run):[/bold green] "
                f"[dim]{effective_workspace}[/dim]"
            )
        logger.info(
            "run_task_interactive: effective_workspace=%s", effective_workspace
        )

        from .models import ResearchType, Task, TaskStatus
        from .state import TaskStateManager

        try:
            rtype = ResearchType(research_type)
        except ValueError:
            self.console.print(
                f"[red]Invalid research type: {research_type}[/red]"
            )
            self.console.print(
                "[dim]Valid types: deep-learning, machine-learning, agent[/dim]"
            )
            return

        task = Task(research_type=rtype, idea_description=idea)
        state_manager = TaskStateManager(task)

        self.console.print(
            f"\n[bold magenta]Created task: {task.id}[/bold magenta]\n"
        )

        try:
            self.db.save_task(task)
            state_manager.start_planning()
            self.db.save_task(task)

            # ── Plan generation ──────────────────────────────────────────────
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task_gen = progress.add_task(
                    "[cyan]Generating experiment plan...", total=None
                )

                if not self.llm:
                    self.console.print(
                        "[red]LLM not configured. Cannot generate plan.[/red]"
                    )
                    return

                available_skills_xml = self.skill_manager.to_prompt_xml()

                # ── AgenticRAG: inject user workspace context into planning ──
                workspace_rag_context = ""
                if effective_workspace is not None:
                    try:
                        from .utils.workspace_rag import build_workspace_rag_context
                        workspace_rag_context = build_workspace_rag_context(
                            effective_workspace
                        )
                        if workspace_rag_context:
                            logger.info(
                                "Injecting %d chars of AgenticRAG context from %s into plan generation",
                                len(workspace_rag_context),
                                effective_workspace,
                            )
                    except Exception as rag_exc:
                        logger.warning(
                            "Failed to build AgenticRAG context for plan generation: %s",
                            rag_exc,
                        )

                # Augment the idea description with workspace context so the
                # plan can take existing files and data into account.
                augmented_idea = idea
                if workspace_rag_context:
                    augmented_idea = (
                        idea
                        + "\n\n"
                        + workspace_rag_context
                    )

                plan = self.llm.generate_plan(rtype, augmented_idea, available_skills_xml)
                progress.update(task_gen, completed=True)

            task.plan = plan
            state_manager.set_plan(plan)
            self.db.save_task(task)

            # ── Show plan ────────────────────────────────────────────────────
            self.console.print("\n[bold]Generated Experiment Plan:[/bold]")
            self.console.print(
                Panel(
                    f"[cyan]{plan.title}[/cyan]\n\n{plan.description}",
                    border_style="cyan",
                )
            )

            self.console.print("\n[bold]Steps:[/bold]")
            for step in plan.steps:
                skill_info = (
                    f" (using {step.skill_required})" if step.skill_required else ""
                )
                time_info = (
                    f" - {step.estimated_duration}min"
                    if step.estimated_duration
                    else ""
                )
                self.console.print(
                    f"  {step.step_number}. {step.description}{skill_info}{time_info}"
                )

            # ── Approval ─────────────────────────────────────────────────────
            self.console.print()
            approved = Confirm.ask(
                "[bold]Do you want to proceed with this plan?[/bold]"
            )

            if approved:
                state_manager.approve_plan()
                state_manager.start_execution()
                self.db.save_task(task)
                await self._execute_plan(
                    state_manager, plan, idea,
                    effective_workspace=effective_workspace,
                )
            else:
                # ── Plan rejected: get feedback and regenerate ─────────────────
                feedback = Prompt.ask(
                    "\n[bold]What would you like to change?[/bold]"
                )
                state_manager.reject_plan(feedback)
                self.db.save_task(task)
                
                # Regeneration loop: allow up to 3 attempts
                max_regeneration_attempts = 3
                regeneration_attempt = 0
                
                while regeneration_attempt < max_regeneration_attempts:
                    regeneration_attempt += 1
                    
                    self.console.print(
                        f"\n[bold yellow]🔄 Regenerating plan (attempt {regeneration_attempt}/{max_regeneration_attempts})...[/bold yellow]"
                    )
                    logger.info(
                        "Regenerating plan with feedback (attempt %d/%d)",
                        regeneration_attempt,
                        max_regeneration_attempts,
                    )
                    
                    try:
                        # Get available skills
                        skills = self.skill_manager.discover_skills()
                        available_skills_xml = self._format_skills_as_xml(skills)
                        
                        # Regenerate plan with feedback
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            console=self.console,
                        ) as progress:
                            task_gen = progress.add_task(
                                "[cyan]Regenerating experiment plan...",
                                total=None,
                            )
                            
                            # Convert previous plan to string for LLM
                            previous_plan_str = self._plan_to_string(plan)
                            
                            # Call LLM to regenerate
                            plan = self.llm.regenerate_plan(
                                research_type=rtype,
                                idea_description=augmented_idea,
                                previous_plan=previous_plan_str,
                                feedback=feedback,
                                available_skills=available_skills_xml,
                            )
                            progress.update(task_gen, completed=True)
                        
                        # Show regenerated plan
                        self.console.print("\n[bold]📋 Regenerated Experiment Plan:[/bold]")
                        self.console.print(
                            Panel(
                                f"[cyan]{plan.title}[/cyan]\n\n{plan.description}",
                                border_style="cyan",
                            )
                        )
                        
                        self.console.print("\n[bold]Steps:[/bold]")
                        for step in plan.steps:
                            skill_info = (
                                f" (using {step.skill_required})" if step.skill_required else ""
                            )
                            time_info = (
                                f" - {step.estimated_duration}min"
                                if step.estimated_duration
                                else ""
                            )
                            self.console.print(
                                f"  {step.step_number}. {step.description}{skill_info}{time_info}"
                            )
                        
                        # Ask for approval again
                        self.console.print()
                        approved = Confirm.ask(
                            "[bold]Do you want to proceed with this plan?[/bold]"
                        )
                        
                        if approved:
                            # Update task with new plan
                            task.plan = plan
                            state_manager.set_plan(plan)
                            state_manager.approve_plan()
                            state_manager.start_execution()
                            self.db.save_task(task)
                            
                            self.console.print(
                                "\n[bold green]✓ Starting execution...[/bold green]"
                            )
                            
                            # Execute the approved plan
                            await self._execute_plan(
                                state_manager, plan, idea,
                                effective_workspace=effective_workspace,
                            )
                            break  # Exit regeneration loop
                        else:
                            # Ask for new feedback
                            feedback = Prompt.ask(
                                "\n[bold]What else would you like to change?[/bold]"
                            )
                            state_manager.reject_plan(feedback)
                            self.db.save_task(task)
                    
                    except Exception as e:
                        logger.error("Plan regeneration failed: %s", e)
                        self.console.print(
                            f"\n[red]✗ Plan regeneration failed: {e}[/red]"
                        )
                        
                        if regeneration_attempt < max_regeneration_attempts:
                            self.console.print(
                                "[yellow]Will retry...[/yellow]"
                            )
                        else:
                            self.console.print(
                                "[red]✗ Maximum regeneration attempts reached.[/red]"
                            )
                            break  # Exit regeneration loop
                
                # If we exhausted all regeneration attempts
                if regeneration_attempt >= max_regeneration_attempts:
                    self.console.print(
                        "\n[yellow]⚠ Plan generation failed after multiple attempts. "
                        "Please try with a different idea description.[/yellow]"
                    )

        except Exception as exc:
            self.console.print(f"\n[red]Error: {exc}[/red]")
            if task:
                state_manager.fail_task(str(exc))
                self.db.save_task(task)

    # ── Main REPL loop ────────────────────────────────────────────────────────

    def run(self) -> None:
        """Run the main CLI loop."""
        self.print_welcome()

        while True:
            try:
                user_input = Prompt.ask(
                    "[bold cyan]IdeaAgent[/bold cyan]",
                    default="",
                    show_default=False,
                ).strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    self._handle_command(user_input)
                else:
                    self.console.print(
                        "[yellow]Please use /new or /run command to start a task.[/yellow]"
                    )
                    self.console.print(
                        "[dim]Type /help for available commands.[/dim]"
                    )

            except KeyboardInterrupt:
                self.console.print(
                    "\n[yellow]Interrupted. Type /quit to exit.[/yellow]"
                )
            except EOFError:
                break

    def _handle_command(self, command: str) -> None:
        """Dispatch a CLI command string.

        Args:
            command: Full command string starting with ``/``.
        """
        # Split into at most 3 parts so that the third part can be a
        # path/string containing spaces (e.g. /workspace C:\My Data\project)
        parts = command.split(maxsplit=2)
        cmd = parts[0].lower()

        if cmd in ["/help", "/h", "/?"]:
            self.print_help()

        elif cmd in ["/quit", "/exit", "/q"]:
            raise EOFError()

        elif cmd in ["/status"]:
            status_info = self.get_status_info()
            for label, value in status_info.items():
                self.console.print(f"{label}: {value}")

        elif cmd in ["/clear", "/cls"]:
            os.system("cls" if os.name == "nt" else "clear")
            self.print_welcome()

        elif cmd in ["/new"]:
            research_type = Prompt.ask(
                "[bold]Research type[/bold]",
                choices=["deep-learning", "machine-learning", "agent"],
                default="machine-learning",
            )
            idea = Prompt.ask("[bold]Describe your research idea[/bold]")
            # ── Per-run workspace (AgenticRAG) ────────────────────────────
            ws_input = Prompt.ask(
                "[bold]Workspace path for this run[/bold] "
                "[dim](leave empty to skip / use session default)[/dim]",
                default="",
            ).strip().strip('"').strip("'")
            run_ws: Optional[Path] = None
            if ws_input:
                ws_path = Path(ws_input).expanduser().resolve()
                if ws_path.exists() and ws_path.is_dir():
                    run_ws = ws_path
                    self.console.print(
                        f"[green]AgenticRAG workspace set for this run:[/green] "
                        f"[dim]{run_ws}[/dim]"
                    )
                else:
                    self.console.print(
                        f"[yellow]Workspace path not found, skipping: {ws_path}[/yellow]"
                    )
            asyncio.run(self.run_task_interactive(research_type, idea, run_workspace=run_ws))

        elif cmd == "/run":
            if len(parts) < 3:
                self.console.print(
                    "[red]Usage: /run <type> <prompt> [--workspace <path>][/red]"
                )
                self.console.print(
                    "[dim]Types: deep-learning, machine-learning, agent[/dim]"
                )
                self.console.print(
                    "[dim]Example: /run machine-learning \"compare Ridge alphas\" "
                    "--workspace ./my_data[/dim]"
                )
                return
            # Parse optional --workspace / -w at end of prompt string
            # e.g. /run machine-learning "my idea" --workspace ./folder
            run_type = parts[1]
            rest = parts[2]  # everything after the type
            run_ws_path: Optional[Path] = None
            for flag in ("--workspace ", "-w "):
                if flag in rest:
                    prompt_part, _, ws_part = rest.partition(flag)
                    # ws_part may have trailing text — take the next token
                    ws_token = ws_part.strip().split()[0].strip('"').strip("'") if ws_part.strip() else ""
                    if ws_token:
                        candidate = Path(ws_token).expanduser().resolve()
                        if candidate.exists() and candidate.is_dir():
                            run_ws_path = candidate
                            rest = prompt_part.strip()
                            self.console.print(
                                f"[green]AgenticRAG workspace for this run:[/green] "
                                f"[dim]{run_ws_path}[/dim]"
                            )
                        else:
                            self.console.print(
                                f"[yellow]Workspace path not found, ignoring: {candidate}[/yellow]"
                            )
                            rest = prompt_part.strip()
                    break
            asyncio.run(
                self.run_task_interactive(run_type, rest, run_workspace=run_ws_path)
            )

        elif cmd in ["/list", "/ls"]:
            self.print_task_list()

        elif cmd == "/show":
            if len(parts) < 2:
                self.console.print("[red]Usage: /show <task-id>[/red]")
                return
            self.print_task_details(parts[1])

        elif cmd == "/delete":
            if len(parts) < 2:
                self.console.print("[red]Usage: /delete <task-id>[/red]")
                return
            if Confirm.ask(f"[yellow]Delete task {parts[1]}?[/yellow]"):
                if self.db.delete_task(parts[1]):
                    self.console.print("[green]Task deleted.[/green]")
                else:
                    self.console.print("[red]Failed to delete task.[/red]")

        elif cmd in ["/skills"]:
            self.print_skills()

        elif cmd == "/validate":
            if len(parts) < 2:
                self.console.print("[red]Usage: /validate <skill-path>[/red]")
                return
            self.validate_skill(parts[1])

        elif cmd == "/workspace":
            self._handle_workspace_command(parts)

        elif cmd == "/config":
            self._show_config()

        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")
            self.console.print("[dim]Type /help for available commands.[/dim]")

    def _handle_workspace_command(self, parts: list[str]) -> None:
        """Handle /workspace [path|clear] command.

        - ``/workspace``        – show current workspace status
        - ``/workspace <path>`` – set a new workspace for this session
        - ``/workspace clear``  – clear the workspace (disable AgenticRAG)
        """
        if len(parts) < 2:
            # Show current workspace status
            if self.user_workspace is not None:
                try:
                    file_count = sum(
                        1 for p in self.user_workspace.rglob("*") if p.is_file()
                    )
                    self.console.print(
                        f"[bold]Current workspace:[/bold] [green]{self.user_workspace}[/green]"
                        f" [dim]({file_count} files)[/dim]"
                    )
                    # Show the files that will be read
                    from .utils.workspace_rag import scan_workspace
                    records = scan_workspace(self.user_workspace)
                    if records:
                        self.console.print("\n[bold]Files that will be injected as RAG context:[/bold]")
                        for r in records:
                            self.console.print(f"  [cyan]{r['rel_path']}[/cyan]  [dim]({r['size_bytes']:,} bytes)[/dim]")
                    else:
                        self.console.print("[dim]No readable files found in workspace.[/dim]")
                except Exception as exc:
                    self.console.print(f"[red]Error reading workspace: {exc}[/red]")
            else:
                self.console.print("[dim]No workspace set. Use /workspace <path> to set one.[/dim]")
            return

        # Join all remaining parts so paths with spaces work correctly.
        # e.g. /workspace C:\My Data\project → parts[1:] = ['C:\\My', 'Data\\project']
        arg = " ".join(parts[1:]).strip().strip('"').strip("'")

        if arg.lower() == "clear":
            self.user_workspace = None
            self.console.print("[green]Workspace cleared. AgenticRAG disabled.[/green]")
            return

        # Treat as a path
        new_path = Path(arg).expanduser().resolve()
        if not new_path.exists():
            self.console.print(f"[red]Path does not exist: {new_path}[/red]")
            return
        if not new_path.is_dir():
            self.console.print(f"[red]Path is not a directory: {new_path}[/red]")
            return

        self.user_workspace = new_path
        try:
            file_count = sum(1 for p in new_path.rglob("*") if p.is_file())
            self.console.print(
                f"[bold green]Workspace set:[/bold green] [dim]{new_path}[/dim] "
                f"[dim]({file_count} files) – AgenticRAG active for this session[/dim]"
            )
        except Exception:
            self.console.print(f"[bold green]Workspace set:[/bold green] [dim]{new_path}[/dim]")

    def _show_config(self) -> None:
        """Show current configuration."""
        from dotenv import dotenv_values

        self.console.print(Panel("Configuration", border_style="cyan"))

        config = dotenv_values(Path.cwd() / ".env")

        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="cyan", justify="right")
        grid.add_column(style="white")

        for key, value in config.items():
            if any(s in key for s in ("KEY", "SECRET", "PASSWORD")):
                display_value = "[dim]***hidden***[/dim]"
            else:
                display_value = value
            grid.add_row(f"{key}:", display_value)

        self.console.print(grid)


# ── Click entry-point ─────────────────────────────────────────────────────────

@click.command()
@click.version_option(version="0.1.0")
@click.option(
    "--workspace",
    "-w",
    default=None,
    metavar="PATH",
    help=(
        "Path to a user workspace directory. All readable files in the directory "
        "are scanned and injected as AgenticRAG context into every LLM call for "
        "this run. Only affects the current run – never persisted. "
        "Example: --workspace ./my_project"
    ),
)
def main(workspace: Optional[str]):
    """IdeaAgent - Experimental Agent for ML Research.

    Interactive CLI for validating machine learning research ideas.
    """
    try:
        cli = IdeaAgentCLI(
            user_workspace=Path(workspace) if workspace else None,
        )
        cli.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    except Exception as exc:
        console.print(f"[red]Fatal error: {exc}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()