import threading
from typing import Any, Dict, Optional
from datetime import datetime

from app.schemas.camera import CameraCalibration

class CalibrationHandler:
    def __init__(self):
        self._lock = threading.RLock()
        self._calibration_data: Optional[CameraCalibration] = None
        self._last_updated: Optional[datetime] = None

    def set_data(self, data: CameraCalibration):
        with self._lock:
            self._calibration_data = data
            self._last_updated = datetime.utcnow()

    def get_data(self) -> Optional[CameraCalibration]:
        with self._lock:
            return self._calibration_data

    def get_data_with_timestamp(self) -> Dict[str, Any]:
        with self._lock:
            return {"last_updated_utc": self._last_updated, "calibration": self._calibration_data}
