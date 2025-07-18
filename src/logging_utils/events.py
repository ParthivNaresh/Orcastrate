"""
Simple log events for the centralized logging system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class LogEvent:
    """Base class for all log events."""

    correlation_id: str
    event_type: str = ""
    timestamp: Optional[datetime] = None
    execution_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ExecutionStarted(LogEvent):
    """Event emitted when execution starts."""

    operation: str = ""
    requirements_description: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.event_type = "execution_started"


@dataclass
class ExecutionCompleted(LogEvent):
    """Event emitted when execution completes."""

    operation: str = ""
    success: bool = False
    duration_seconds: float = 0.0
    artifacts_count: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.event_type = "execution_completed"


@dataclass
class StepStarted(LogEvent):
    """Event emitted when a step starts."""

    step_id: str = ""
    step_name: str = ""
    tool: str = ""
    action: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.event_type = "step_started"


@dataclass
class StepCompleted(LogEvent):
    """Event emitted when a step completes."""

    step_id: str = ""
    step_name: str = ""
    success: bool = False
    duration_seconds: float = 0.0
    error_message: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.event_type = "step_completed"
