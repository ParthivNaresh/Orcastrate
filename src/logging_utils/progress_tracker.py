"""
Progress tracking with rich progress integration for real-time messaging.
"""

from typing import List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

from .events import StepCompleted, StepStarted
from .log_manager import LogManager


class ProgressTracker:
    """Progress tracker with rich progress for real-time messaging."""

    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
        self.console = Console()
        self._progress: Optional[Progress] = None
        self._current_task: Optional[TaskID] = None
        self._step_messages: List[str] = []
        self._live: Optional[Live] = None

    def start_execution_progress(
        self, total_steps: int, title: str = "🚀 Orcastrate"
    ) -> None:
        """Start tracking execution progress."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        self._current_task = self._progress.add_task(title, total=total_steps)

        # Create a live display with progress bar and step messages
        self._live = Live(
            self._create_display(), console=self.console, refresh_per_second=10
        )
        self._live.start()

    def _create_display(self):
        """Create the display group with progress bar and step messages."""
        if not self._progress:
            return Text("")

        # Create step messages text
        step_text = Text()
        for msg in self._step_messages:
            step_text.append(f"{msg}\n")

        # Group progress bar and step messages
        return Group(self._progress, step_text)

    def update_step_progress(self, increment: int = 1) -> None:
        """Update progress counter."""
        if self._progress and self._current_task is not None:
            self._progress.advance(self._current_task, increment)
            if self._live:
                self._live.update(self._create_display())

    def add_step_message(
        self, step_name: str, indent: int = 0, completed: bool = False
    ) -> None:
        """Add a step message and display it immediately below the progress bar."""
        status_icon = " ✅" if completed else " ❌"
        indent_amount = "  " * 2 * indent
        step_msg = f"{indent_amount}{step_name}{status_icon}"
        self._step_messages.append(step_msg)
        if self._live:
            self._live.update(self._create_display())

    def complete_step_progress(self, step_name: str) -> None:
        """Mark a step as completed - show only the completed version."""
        self.add_step_message(step_name, completed=True)

    def complete_execution_progress(self) -> None:
        """Complete the execution progress."""
        if self._live:
            self._live.stop()
            self._live = None
        if self._progress:
            self._progress = None
            self._current_task = None
            self._step_messages = []

    def log_step_success(self, message: str, indent: int = 0) -> None:
        """Helper method to log a successful step with progress update."""
        self.update_step_progress()
        self.add_step_message(message, indent, completed=True)

    def log_step_failure(self, message: str, indent: int = 0) -> None:
        """Helper method to log a failed step with progress update."""
        self.update_step_progress()
        self.add_step_message(message, indent, completed=False)

    def log_step_conditional(
        self, success_msg: str, failure_msg: str, success: bool, indent: int = 0
    ) -> None:
        """Helper method to conditionally log success or failure with progress update."""
        if success:
            self.log_step_success(success_msg, indent)
        else:
            self.log_step_failure(failure_msg, indent)

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

                # Update progress counter but don't show step message yet
                self.tracker.update_step_progress()
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                from datetime import datetime

                end_time = datetime.utcnow()
                duration = (
                    end_time - (self.start_time or datetime.utcnow())
                ).total_seconds()
                success = exc_type is None

                # Show the completed step message immediately
                status_icon = "✅" if success else "❌"
                self.tracker.add_step_message(
                    f"{self.step_name} {status_icon}", completed=False
                )

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
