"""Context management for IdeaAgent.

Provides a unified way to manage constant context information that should
always be included in LLM prompts:
- Workspace path (specified or auto-generated)
- Initial instruction (user's original request)
"""

from pathlib import Path
from typing import Optional


class ContextManager:
    """Manages constant context information for LLM calls.
    
    The context manager ensures that critical information is consistently
    available across all LLM interactions during a task execution.
    
    Attributes:
        workspace: The user-specified or default workspace directory.
        initial_instruction: The user's original research idea/request.
    """
    
    def __init__(
        self,
        workspace: Optional[Path],
        initial_instruction: str,
    ):
        """Initialize the context manager.
        
        Args:
            workspace: Path to the user workspace directory, or None if not specified.
            initial_instruction: The original research idea or task description.
        """
        self.workspace = workspace
        self.initial_instruction = initial_instruction
    
    def build_constant_context(self) -> str:
        """Build the constant context section for prompts.
        
        Returns:
            A formatted string containing workspace and initial instruction
            information. Always returns non-empty content since
            initial_instruction is required.
        """
        lines = [
            "=== CONSTANT CONTEXT ===",
            "",
            "# Initial Instruction",
            self.initial_instruction,
            "",
        ]
        
        if self.workspace is not None:
            lines.extend([
                "# User Workspace",
                f"Path: {self.workspace}",
                "",
            ])
        else:
            lines.append("# User Workspace: Not specified\n")
        
        return "\n".join(lines)
    
    def get_workspace_path(self) -> Optional[Path]:
        """Get the workspace path.
        
        Returns:
            The configured workspace path, or None if not set.
        """
        return self.workspace
    
    def get_initial_instruction(self) -> str:
        """Get the initial instruction.
        
        Returns:
            The original research idea or task description.
        """
        return self.initial_instruction