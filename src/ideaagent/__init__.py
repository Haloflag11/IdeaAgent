"""IdeaAgent - Experimental Agent for validating machine learning research ideas."""

__version__ = "0.1.0"
__author__ = "IdeaAgent Team"

from .models import (
    ResearchType,
    Task,
    TaskStatus,
    ExperimentPlan,
    ExecutionResult,
)
from .database import Database
from .state import TaskStateManager
from .context import ContextManager, PersistentContext

__all__ = [
    "ResearchType",
    "Task",
    "TaskStatus",
    "ExperimentPlan",
    "ExecutionResult",
    "Database",
    "TaskStateManager",
    "ContextManager",
    "PersistentContext",
]
