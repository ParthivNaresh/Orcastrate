"""
Progress tracking with alive-progress integration.
"""

import sys
from alive_progress import alive_bar

from .events import StepCompleted, StepStarted
from .log_manager import LogManager


class ProgressTracker:
    """Simple progress tracker with alive-progress and persistent step messages."""

    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
        self._current_bar = None
        self._step_messages = []  # Store persistent step messages
        self._current_step_line = None  # Track the current step being updated

    def start_execution_progress(
        self, total_steps: int, title: str = "🚀 Orcastrate"
    ) -> None:
        """Start tracking execution progress."""
        self._current_bar = alive_bar(
            total_steps,
            title=title,
            bar="smooth",
            spinner="dots_waves",
            stats=True,
            elapsed=True,
        )
        self._progress_context = self._current_bar.__enter__()
        self._step_messages = []  # Reset step messages for new execution

    def update_step_progress(self, step_name: str, increment: int = 1) -> None:
        """Update progress for a step."""
        if self._progress_context:
            # Just update the progress bar counter, don't print step messages yet
            self._progress_context(increment)

    def add_step_message(self, step_name: str, completed: bool = False) -> None:
        """Add a persistent step message and display it immediately."""
        status_icon = " ✅" if completed else ""
        step_msg = f"  {step_name}{status_icon}"
        print(step_msg, flush=True)
        self._step_messages.append(step_msg)

    def complete_step_progress(self, step_name: str) -> None:
        """Mark a step as completed - this is called manually for init steps."""
        # For manual completion, just store the message for later display
        self.add_step_message(step_name, completed=True)

    def complete_execution_progress(self) -> None:
        """Complete the execution progress."""
        if self._current_bar:
            self._current_bar.__exit__(None, None, None)
            self._current_bar = None
            self._progress_context = None
            
            # Step messages have already been printed in real-time
            self._step_messages = []

    def track_step_execution(
        self, step_id: str, step_name: str, correlation_id: str, execution_id: str
    ):
        """Context manager for tracking step execution with events and persistent display."""

        class StepTracker:
            def __init__(
                self, tracker, step_id, step_name, correlation_id, execution_id
            ):
                self.tracker = tracker
                self.step_id = step_id
                self.step_name = step_name
                self.correlation_id = correlation_id
                self.execution_id = execution_id
                self.start_time = None

            async def __aenter__(self):
                from datetime import datetime

                self.start_time = datetime.utcnow()

                # Emit step started event
                await self.tracker.log_manager.emit_event(
                    StepStarted(
                        timestamp=self.start_time,
                        correlation_id=self.correlation_id,
                        execution_id=self.execution_id,
                        step_id=self.step_id,
                        step_name=self.step_name,
                        tool="unknown",  # Will be filled by caller
                        action="unknown",  # Will be filled by caller
                    )
                )

                # Update progress counter and show step started
                self.tracker.update_step_progress(self.step_name)
                self.tracker.add_step_message(self.step_name, completed=False)
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                from datetime import datetime

                end_time = datetime.utcnow()
                duration = (end_time - self.start_time).total_seconds()
                success = exc_type is None

                # Show the completed step message immediately
                status_icon = "✅" if success else "❌"
                print(f"  {self.step_name} {status_icon}", flush=True)

                # Emit step completed event
                await self.tracker.log_manager.emit_event(
                    StepCompleted(
                        timestamp=end_time,
                        correlation_id=self.correlation_id,
                        execution_id=self.execution_id,
                        step_id=self.step_id,
                        step_name=self.step_name,
                        success=success,
                        duration_seconds=duration,
                        error_message=str(exc_val) if exc_val else None,
                    )
                )

        return StepTracker(self, step_id, step_name, correlation_id, execution_id)
