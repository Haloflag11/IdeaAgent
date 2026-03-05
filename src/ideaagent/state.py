"""Task state management for IdeaAgent."""

from typing import Optional, Callable
from datetime import datetime

from .models import Task, TaskStatus, ExperimentPlan, ExecutionResult


class TaskStateManager:
    """Manages the state of tasks throughout their lifecycle."""

    def __init__(self, task: Task, on_state_change: Optional[Callable] = None):
        """Initialize task state manager.
        
        Args:
            task: Task to manage
            on_state_change: Optional callback function called on state changes
        """
        self.task = task
        self.on_state_change = on_state_change
        self.state_history: list[dict] = []
        self._record_state("Initial state")

    def _record_state(self, note: str) -> None:
        """Record current state in history."""
        self.state_history.append({
            "timestamp": datetime.now().isoformat(),
            "status": self.task.status.value,
            "loop_count": self.task.loop_count,
            "note": note,
        })

    def _update_status(self, new_status: TaskStatus, note: str = "") -> None:
        """Update task status and trigger callback."""
        old_status = self.task.status
        self.task.status = new_status
        self.task.updated_at = datetime.now()
        self._record_state(note)
        
        if self.on_state_change:
            self.on_state_change(self.task, old_status, new_status)

    def start_planning(self) -> None:
        """Transition to planning state."""
        if self.task.status == TaskStatus.PENDING:
            self._update_status(TaskStatus.PLANNING, "Started planning phase")
        else:
            raise ValueError(f"Cannot start planning from status: {self.task.status}")

    def set_plan(self, plan: ExperimentPlan) -> None:
        """Set the experiment plan and wait for approval."""
        if self.task.status == TaskStatus.PLANNING:
            self.task.plan = plan
            self._update_status(TaskStatus.WAITING_APPROVAL, "Plan created, waiting for approval")
        else:
            raise ValueError(f"Cannot set plan in status: {self.task.status}")

    def approve_plan(self) -> None:
        """Approve the current plan."""
        if self.task.status == TaskStatus.WAITING_APPROVAL:
            self._update_status(TaskStatus.APPROVED, "Plan approved")
        else:
            raise ValueError(f"Cannot approve plan in status: {self.task.status}")

    def reject_plan(self, reason: str) -> None:
        """Reject the current plan."""
        if self.task.status == TaskStatus.WAITING_APPROVAL:
            self.task.error_message = reason
            self._update_status(TaskStatus.REJECTED, f"Plan rejected: {reason}")
        else:
            raise ValueError(f"Cannot reject plan in status: {self.task.status}")

    def start_execution(self) -> None:
        """Start executing the approved plan."""
        if self.task.status == TaskStatus.APPROVED:
            self._update_status(TaskStatus.RUNNING, "Started execution")
        else:
            raise ValueError(f"Cannot start execution from status: {self.task.status}")

    def add_execution_result(self, result: ExecutionResult) -> None:
        """Add an execution result."""
        if self.task.status == TaskStatus.RUNNING:
            self.task.execution_results.append(result)
            self.task.updated_at = datetime.now()
            self._record_state(f"Added result for step {result.step_number}")
        else:
            raise ValueError(f"Cannot add result in status: {self.task.status}")

    def complete_task(self) -> None:
        """Mark task as completed."""
        if self.task.status == TaskStatus.RUNNING:
            self._update_status(TaskStatus.COMPLETED, "Task completed successfully")
        else:
            raise ValueError(f"Cannot complete task in status: {self.task.status}")

    def fail_task(self, error_message: str) -> None:
        """Mark task as failed."""
        if self.task.status in [TaskStatus.RUNNING, TaskStatus.PLANNING]:
            self.task.error_message = error_message
            self._update_status(TaskStatus.FAILED, f"Task failed: {error_message}")
        else:
            raise ValueError(f"Cannot fail task in status: {self.task.status}")

    def stop_task(self, reason: str = "User requested") -> None:
        """Stop task execution."""
        if self.task.status == TaskStatus.RUNNING:
            self.task.error_message = reason
            self._update_status(TaskStatus.STOPPED, f"Task stopped: {reason}")
        else:
            raise ValueError(f"Cannot stop task in status: {self.task.status}")

    def increment_loop_count(self) -> int:
        """Increment and return the loop count."""
        self.task.loop_count += 1
        self.task.updated_at = datetime.now()
        self._record_state(f"Loop count incremented to {self.task.loop_count}")
        return self.task.loop_count

    def get_loop_count(self) -> int:
        """Get current loop count."""
        return self.task.loop_count

    def reset_loop_count(self) -> None:
        """Reset loop count to zero."""
        self.task.loop_count = 0
        self._record_state("Loop count reset")

    def get_current_step(self) -> Optional[int]:
        """Get the current step number being executed."""
        if not self.task.plan:
            return None
        
        completed_steps = len(self.task.execution_results)
        if completed_steps < len(self.task.plan.steps):
            return completed_steps + 1
        return None

    def get_progress(self) -> float:
        """Get execution progress as a percentage."""
        if not self.task.plan:
            return 0.0
        
        total_steps = len(self.task.plan.steps)
        if total_steps == 0:
            return 0.0
        
        completed_steps = len(self.task.execution_results)
        return (completed_steps / total_steps) * 100

    def get_state_summary(self) -> dict:
        """Get a summary of the current state."""
        return {
            "task_id": self.task.id,
            "status": self.task.status.value,
            "research_type": self.task.research_type.value,
            "loop_count": self.task.loop_count,
            "current_step": self.get_current_step(),
            "progress": self.get_progress(),
            "total_steps": len(self.task.plan.steps) if self.task.plan else 0,
            "created_at": self.task.created_at.isoformat(),
            "updated_at": self.task.updated_at.isoformat(),
            "state_history_length": len(self.state_history),
        }
