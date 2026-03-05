"""Loop detection to prevent infinite execution."""

from typing import Optional
from datetime import datetime, timedelta

from .models import Task, TaskStatus


class LoopDetector:
    """Detects potential infinite loops in agent execution."""

    def __init__(self, max_loop_count: int = 10):
        """Initialize loop detector.
        
        Args:
            max_loop_count: Maximum allowed loop iterations before stopping
        """
        self.max_loop_count = max_loop_count
        self.loop_history: list[dict] = []

    def record_action(self, task_id: str, action: str, context: Optional[dict] = None) -> None:
        """Record an action for loop detection analysis.
        
        Args:
            task_id: ID of the task
            action: Description of the action taken
            context: Optional context about the action
        """
        self.loop_history.append({
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "action": action,
            "context": context or {},
        })

    def check_loop_count(self, task: Task) -> tuple[bool, str]:
        """Check if task has exceeded loop count limit.
        
        Args:
            task: Task to check
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if task.loop_count >= self.max_loop_count:
            return True, f"Loop count exceeded maximum ({self.max_loop_count})"
        return False, ""

    def detect_repetitive_actions(self, task_id: str, window_size: int = 5) -> tuple[bool, str]:
        """Detect if the same action is being repeated.
        
        Args:
            task_id: ID of the task to check
            window_size: Number of recent actions to analyze
            
        Returns:
            Tuple of (is_repetitive, pattern_description)
        """
        if len(self.loop_history) < window_size:
            return False, ""

        # Get recent actions for this task
        recent_actions = [
            h for h in self.loop_history[-window_size:]
            if h["task_id"] == task_id
        ]

        if len(recent_actions) < window_size:
            return False, ""

        # Check if all actions are the same
        action_types = [a["action"] for a in recent_actions]
        if len(set(action_types)) == 1:
            return True, f"Repetitive action detected: {action_types[0]}"

        return False, ""

    def detect_state_oscillation(self, task: Task, window_size: int = 4) -> tuple[bool, str]:
        """Detect if task state is oscillating without progress.
        
        Args:
            task: Task to check
            window_size: Number of state changes to analyze
            
        Returns:
            Tuple of (is_oscillating, pattern_description)
        """
        # Get recent state changes for this task
        task_history = [
            h for h in self.loop_history
            if h["task_id"] == task_id
        ][-window_size:]

        if len(task_history) < window_size:
            return False, ""

        # Check for oscillating patterns
        states = [h.get("context", {}).get("status") for h in task_history]
        if len(set(states)) <= 2 and len(states) > 2:
            return True, f"State oscillation detected: {states}"

        return False, ""

    def detect_no_progress(self, task: Task, time_window_minutes: int = 30) -> tuple[bool, str]:
        """Detect if task has made no progress in a time window.
        
        Args:
            task: Task to check
            time_window_minutes: Time window to check for progress
            
        Returns:
            Tuple of (no_progress, reason)
        """
        if task.status != TaskStatus.RUNNING:
            return False, ""

        # Check if execution results have been added recently
        if not task.execution_results:
            return False, ""

        latest_result = max(task.execution_results, key=lambda r: r.timestamp)
        time_since_progress = datetime.now() - latest_result.timestamp

        if time_since_progress > timedelta(minutes=time_window_minutes):
            return True, f"No progress in {time_window_minutes} minutes"

        return False, ""

    def analyze(self, task: Task) -> tuple[bool, str]:
        """Perform comprehensive loop analysis.
        
        Args:
            task: Task to analyze
            
        Returns:
            Tuple of (should_stop, reason)
        """
        # Check loop count
        should_stop, reason = self.check_loop_count(task)
        if should_stop:
            return True, reason

        # Check repetitive actions
        is_repetitive, pattern = self.detect_repetitive_actions(task.id)
        if is_repetitive:
            return True, pattern

        # Check state oscillation
        is_oscillating, pattern = self.detect_state_oscillation(task)
        if is_oscillating:
            return True, pattern

        # Check no progress
        no_progress, reason = self.detect_no_progress(task)
        if no_progress:
            return True, reason

        return False, ""

    def reset(self) -> None:
        """Reset loop detection history."""
        self.loop_history.clear()

    def get_statistics(self) -> dict:
        """Get loop detection statistics."""
        return {
            "total_actions_recorded": len(self.loop_history),
            "max_loop_count": self.max_loop_count,
            "unique_tasks": len(set(h["task_id"] for h in self.loop_history)),
        }
