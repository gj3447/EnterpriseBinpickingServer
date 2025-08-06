import threading
import time
from typing import Dict, Any

from datetime import datetime

class EventHandler:
    """
    Handles the state of event publications, primarily for monitoring and debugging.
    It tracks the last time each type of event was published.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._last_event_timestamps: Dict[str, float] = {}

    def update_event_timestamp(self, event_name: str):
        """Updates the timestamp for a given event name."""
        with self._lock:
            self._last_event_timestamps[event_name] = time.time()

    def get_status(self) -> Dict[str, Any]:
        """
        Returns the last known timestamp for all tracked events as
        human-readable ISO 8601 strings.
        """
        with self._lock:
            # Convert float timestamps to ISO 8601 formatted strings
            return {
                event_name: datetime.fromtimestamp(ts).isoformat()
                for event_name, ts in self._last_event_timestamps.items()
            }

