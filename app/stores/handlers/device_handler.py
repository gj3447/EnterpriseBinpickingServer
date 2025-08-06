import threading
from typing import Optional

from app.schemas.camera import CameraStatus, StreamConfig

class DeviceHandler:
    def __init__(self):
        self._lock = threading.RLock()
        self._status: Optional[CameraStatus] = None

    def update_status(self, status_data: CameraStatus):
        with self._lock:
            self._status = status_data

    def get_config(self) -> Optional[StreamConfig]:
        with self._lock:
            return self._status.active_stream_config if self._status else None

    def get_full_status(self) -> Optional[CameraStatus]:
        with self._lock:
            return self._status
