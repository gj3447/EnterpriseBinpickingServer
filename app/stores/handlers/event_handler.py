import threading
import time
from typing import Dict, Any, List, Deque
from collections import deque, defaultdict
from datetime import datetime

class EventHandler:
    """
    Handles the state of event publications, primarily for monitoring and debugging.
    It tracks the event publication timestamps and calculates average FPS for each event type.
    """
    def __init__(self, window_size: int = 30):
        self._lock = threading.RLock()
        self._last_event_timestamps: Dict[str, float] = {}
        self._event_timestamps_window: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self._window_size = window_size
        self._event_counts: Dict[str, int] = defaultdict(int)
        self._fps_cache: Dict[str, float] = {}
        self._last_fps_calc_time: Dict[str, float] = defaultdict(float)

    def update_event_timestamp(self, event_name: str):
        """Updates the timestamp for a given event name and maintains FPS calculation window."""
        current_time = time.time()
        with self._lock:
            self._last_event_timestamps[event_name] = current_time
            self._event_timestamps_window[event_name].append(current_time)
            self._event_counts[event_name] += 1
            
            # 1초마다 FPS 재계산 (캐싱으로 성능 개선)
            if current_time - self._last_fps_calc_time[event_name] >= 1.0:
                self._calculate_fps(event_name)
                self._last_fps_calc_time[event_name] = current_time

    def _calculate_fps(self, event_name: str) -> float:
        """Calculate FPS based on the sliding window of timestamps."""
        timestamps = self._event_timestamps_window[event_name]
        if len(timestamps) < 2:
            self._fps_cache[event_name] = 0.0
            return 0.0
        
        # 윈도우 내의 시간 범위 계산
        time_span = timestamps[-1] - timestamps[0]
        if time_span > 0:
            # 실제 이벤트 수는 타임스탬프 수 - 1 (간격의 수)
            fps = (len(timestamps) - 1) / time_span
            self._fps_cache[event_name] = round(fps, 1)
        else:
            self._fps_cache[event_name] = 0.0
        
        return self._fps_cache[event_name]

    def get_fps(self, event_name: str) -> float:
        """Get the current FPS for a specific event."""
        with self._lock:
            return self._fps_cache.get(event_name, 0.0)

    def get_status(self) -> Dict[str, Any]:
        """
        Returns comprehensive status including last timestamp, FPS, and total count for all events.
        """
        with self._lock:
            status = {}
            for event_name in self._last_event_timestamps:
                # 최신 FPS 계산 (캐시된 값이 오래되었을 수 있음)
                self._calculate_fps(event_name)
                
                status[event_name] = {
                    "last_timestamp": datetime.fromtimestamp(
                        self._last_event_timestamps[event_name]
                    ).isoformat(),
                    "fps": self._fps_cache.get(event_name, 0.0),
                    "total_count": self._event_counts[event_name],
                    "window_events": len(self._event_timestamps_window[event_name])
                }
            return status

    def get_all_fps(self) -> Dict[str, float]:
        """Get FPS values for all tracked events."""
        with self._lock:
            return {
                event_name: self.get_fps(event_name)
                for event_name in self._last_event_timestamps
            }

