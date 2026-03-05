"""Context management for IdeaAgent.

This module provides a unified context management system that ensures
the initial instruction and workspace information are always present
in the LLM context throughout the entire task lifecycle.

Key features:
- Persistent context: Initial instruction and workspace info are always included
- Unified context building: Integrates all context sources
- Message history management: Maintains conversation history for LLM calls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import ResearchType, ExecutionResult

logger = logging.getLogger("IdeaAgent.context")


@dataclass
class PersistentContext:
    """Holds persistent context that should always be in LLM context.
    
    This includes:
    - Initial instruction: The original research idea/description
    - Workspace information: Both task workspace and user workspace
    - Research type: The type of research being conducted
    """
    initial_instruction: str
    workspace_dir: Path
    research_type: ResearchType
    user_workspace_path: Optional[Path] = None
    workspace_structure: str = ""
    workspace_rag_context: str = ""
    
    def __post_init__(self):
        """Load workspace information after initialization."""
        self._load_workspace_info()
    
    def _load_workspace_info(self):
        """Load workspace structure and RAG context."""
        from .utils.workspace import get_workspace_structure
        from .utils.workspace_rag import build_workspace_rag_context
        
        # Load workspace structure
        try:
            self.workspace_structure = get_workspace_structure(self.workspace_dir)
        except Exception as e:
            logger.warning(f"Failed to get workspace structure: {e}")
            self.workspace_structure = "Workspace directory does not exist yet."
        
        # Load user workspace RAG context if specified
        if self.user_workspace_path is not None:
            try:
                self.workspace_rag_context = build_workspace_rag_context(
                    self.user_workspace_path
                )
            except Exception as e:
                logger.warning(f"Failed to build workspace RAG context: {e}")
                self.workspace_rag_context = ""


class ContextManager:
    """Manages the complete context for LLM interactions.
    
    This class ensures that the initial instruction and workspace information
    are always present in the LLM context, regardless of which phase of the
    task we're in (planning, execution, error fixing, etc.).
    
    The context is built incrementally:
    1. Persistent context (always present): initial instruction, workspace
    2. Execution history (accumulates as steps are executed)
    3. Current step context (the specific step being executed)
    """
    
    def __init__(
        self,
        initial_instruction: str,
        workspace_dir: Path,
        research_type: ResearchType,
        user_workspace_path: Optional[Path] = None,
    ):
        """Initialize the context manager.
        
        Args:
            initial_instruction: The original research idea/description
            workspace_dir: The task workspace directory
            research_type: The type of research being conducted
            user_workspace_path: Optional user-specified workspace path
        """
        self.persistent = PersistentContext(
            initial_instruction=initial_instruction,
            workspace_dir=workspace_dir,
            research_type=research_type,
            user_workspace_path=user_workspace_path,
        )
        self.execution_history: list[dict] = []
        self.installed_packages: list[str] = []
        self.files_created: list[str] = []
        self.current_step: int = 0
        self.total_steps: int = 0
        
        # Message history for multi-turn conversations
        self.message_history: list[dict] = []
    
    def update_workspace(self, workspace_dir: Optional[Path] = None):
        """Update workspace information.
        
        Args:
            workspace_dir: New workspace directory path
        """
        if workspace_dir is not None:
            self.persistent.workspace_dir = workspace_dir
            self.persistent._load_workspace_info()
    
    def add_execution_result(
        self,
        step_number: int,
        description: str,
        success: bool,
        output: str,
        error: Optional[str] = None,
        packages_installed: Optional[list[str]] = None,
        files_created: Optional[list[str]] = None,
    ) -> None:
        """Add an execution result to the history.
        
        Args:
            step_number: The step number that was executed
            description: Description of the step
            success: Whether the execution succeeded
            output: stdout from the execution
            error: stderr from the execution
            packages_installed: List of packages installed during this step
            files_created: List of files created during this step
        """
        self.execution_history.append({
            "step_number": step_number,
            "description": description,
            "success": success,
            "output": output,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })
        
        if packages_installed:
            self.installed_packages.extend(packages_installed)
        
        if files_created:
            self.files_created.extend(files_created)
    
    def set_current_step(self, step_number: int, total_steps: int) -> None:
        """Set the current step being executed.
        
        Args:
            step_number: The current step number (1-indexed)
            total_steps: Total number of steps in the plan
        """
        self.current_step = step_number
        self.total_steps = total_steps
    
    def build_persistent_context_section(self) -> str:
        """Build the persistent context section that's always included.
        
        This includes:
        - Initial instruction
        - Workspace information
        - Research type
        """
        sections = []
        
        # Initial instruction section
        sections.append(
            "=== INITIAL INSTRUCTION (ALWAYS KEEP IN MIND) ===\n"
            f"{self.persistent.initial_instruction}\n"
            "This is your primary objective. Keep this in mind throughout "
            "all operations.\n"
        )
        
        # Research type section
        sections.append(
            f"=== RESEARCH TYPE ===\n"
            f"{self.persistent.research_type.value}\n"
        )
        
        # Workspace RAG context (if user workspace is specified)
        if self.persistent.workspace_rag_context:
            sections.append(
                "=== USER WORKSPACE CONTEXT (REFERENCE FILES) ===\n"
                f"User workspace path: {self.persistent.user_workspace_path}\n"
                "The following files are available for reference:\n\n"
                f"{self.persistent.workspace_rag_context}\n"
            )
        
        # Workspace structure
        sections.append(
            "=== WORKSPACE STRUCTURE ===\n"
            f"Task workspace: {self.persistent.workspace_dir}\n"
            f"{self.persistent.workspace_structure}\n"
        )
        
        return "\n".join(sections)
    
    def build_execution_history_section(self, max_steps: int = 10) -> str:
        """Build the execution history section.
        
        Args:
            max_steps: Maximum number of recent steps to include
            
        Returns:
            Formatted execution history string
        """
        if not self.execution_history:
            return "=== EXECUTION HISTORY ===\nNo steps executed yet.\n"
        
        # Keep only recent steps to avoid context overflow
        recent_history = self.execution_history[-max_steps:]
        
        sections = ["=== EXECUTION HISTORY ==="]
        
        for entry in recent_history:
            status = "SUCCESS" if entry["success"] else "FAILED"
            section = (
                f"--- Step {entry['step_number']}: {entry['description']} ---\n"
                f"Status: {status}\n"
            )
            
            if entry["output"]:
                output_preview = entry["output"][:500]
                if len(entry["output"]) > 500:
                    output_preview += "\n... [truncated]"
                section += f"Output:\n{output_preview}\n"
            
            if entry["error"]:
                error_preview = entry["error"][:300]
                if len(entry["error"]) > 300:
                    error_preview += "\n... [truncated]"
                section += f"Error:\n{error_preview}\n"
            
            sections.append(section)
        
        # Add summary of installed packages and created files
        if self.installed_packages:
            sections.append(
                f"=== PACKAGES INSTALLED ===\n"
                f"{', '.join(self.installed_packages)}\n"
            )
        
        if self.files_created:
            sections.append(
                f"=== FILES CREATED ===\n"
                f"{', '.join(self.files_created)}\n"
            )
        
        return "\n".join(sections)
    
    def build_full_context(self) -> str:
        """Build the complete context string for LLM.
        
        This combines:
        1. Persistent context (always present)
        2. Execution history
        3. Current step information
        
        Returns:
            Complete context string
        """
        sections = []
        
        # 1. Persistent context (ALWAYS included)
        sections.append(self.build_persistent_context_section())
        
        # 2. Execution history
        sections.append(self.build_execution_history_section())
        
        # 3. Current step progress
        if self.total_steps > 0:
            sections.append(
                f"=== PROGRESS ===\n"
                f"Current step: {self.current_step}/{self.total_steps}\n"
            )
        
        return "\n\n".join(sections)
    
    def get_messages_for_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> list[dict]:
        """Get the complete message list for LLM API call.
        
        This method constructs the messages list that should be passed
        to the LLM API, ensuring that the persistent context is always
        included.
        
        Args:
            system_prompt: The system prompt to use
            user_prompt: The current user prompt/request
            
        Returns:
            List of message dicts for LLM API
        """
        # Build full context
        full_context = self.build_full_context()
        
        # Combine context with user prompt
        combined_user_prompt = f"{full_context}\n\n=== CURRENT REQUEST ===\n{user_prompt}"
        
        # Return messages for API
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_user_prompt},
        ]
    
    def add_to_message_history(self, role: str, content: str) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: Message role ("user", "assistant", or "system")
            content: Message content
        """
        self.message_history.append({"role": role, "content": content})
    
    def get_conversation_history(self) -> list[dict]:
        """Get the full conversation history.
        
        Returns:
            List of message dicts representing conversation history
        """
        return list(self.message_history)
    
    def clear_execution_history(self) -> None:
        """Clear the execution history."""
        self.execution_history.clear()
        self.installed_packages.clear()
        self.files_created.clear()
    
    def get_summary(self) -> dict:
        """Get a summary of the current context state.
        
        Returns:
            Dict with context summary information
        """
        return {
            "initial_instruction_length": len(self.persistent.initial_instruction),
            "workspace_dir": str(self.persistent.workspace_dir),
            "user_workspace_path": str(self.persistent.user_workspace_path) if self.persistent.user_workspace_path else None,
            "research_type": self.persistent.research_type.value,
            "execution_history_length": len(self.execution_history),
            "installed_packages_count": len(self.installed_packages),
            "files_created_count": len(self.files_created),
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "message_history_length": len(self.message_history),
        }