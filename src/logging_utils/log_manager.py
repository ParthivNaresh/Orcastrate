"""
Simple centralized log manager.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from ..utils.directories import get_secure_app_directory
from .events import LogEvent


class LogManager:
    """Simple centralized logging manager."""

    def __init__(self, log_dir: Optional[str] = None):
        if log_dir is None:
            self.log_dir = get_secure_app_directory("orcastrate", "logs")
        else:
            self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger("LogManager")

        # Event storage (simple in-memory for now)
        self.events: List[LogEvent] = []

        # JSON event log file
        self.event_log_file = self.log_dir / "events.jsonl"

    async def emit_event(self, event: LogEvent) -> None:
        """Emit a log event."""
        # Store in memory
        self.events.append(event)

        # Write to structured log
        self._write_event_to_file(event)

        # Also emit to regular logger for backwards compatibility
        # self.logger.info(
        #     f"{event.event_type}: {event.__dict__}",
        #     extra={
        #         "event_type": event.event_type,
        #         "correlation_id": event.correlation_id,
        #         "execution_id": event.execution_id,
        #         **event.metadata,
        #     },
        # )

    def _write_event_to_file(self, event: LogEvent) -> None:
        """Write event to JSONL file."""
        try:
            event_dict = {
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat() if event.timestamp else "",
                "correlation_id": event.correlation_id,
                "execution_id": event.execution_id,
                **{
                    k: v
                    for k, v in event.__dict__.items()
                    if k
                    not in [
                        "event_type",
                        "timestamp",
                        "correlation_id",
                        "execution_id",
                        "metadata",
                    ]
                },
                **(event.metadata or {}),
            }

            with open(self.event_log_file, "a") as f:
                f.write(json.dumps(event_dict, default=str) + "\n")

        except Exception as e:
            self.logger.error(f"Failed to write event to file: {e}")

    def get_events_for_execution(self, execution_id: str) -> List[LogEvent]:
        """Get all events for a specific execution."""
        return [event for event in self.events if event.execution_id == execution_id]

    def get_recent_events(self, count: int = 50) -> List[LogEvent]:
        """Get the most recent events."""
        return self.events[-count:] if len(self.events) > count else self.events
