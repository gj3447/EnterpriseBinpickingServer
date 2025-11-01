import threading
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import numpy as np


class PointcloudHandler:
    """포인트클라우드 데이터를 저장하고 관리하는 핸들러"""
    
    def __init__(self):
        self._lock = threading.RLock()
        # 포인트클라우드 데이터 (xyz coordinates + rgb colors)
        self._points: Optional[np.ndarray] = None  # Shape: (N, 3) for xyz
        self._colors: Optional[np.ndarray] = None  # Shape: (N, 3) for rgb
        self._timestamp: Optional[float] = None
        self._last_updated: Optional[datetime] = None
        
        # 통계 정보
        self._total_points: int = 0
        self._valid_points: int = 0
        
    def update_pointcloud(self, points: np.ndarray, colors: Optional[np.ndarray] = None, timestamp: Optional[float] = None):
        """포인트클라우드 데이터를 업데이트합니다."""
        with self._lock:
            self._points = points
            self._colors = colors
            self._timestamp = timestamp
            self._last_updated = datetime.utcnow()
            
            # 통계 업데이트
            self._total_points = len(points) if points is not None else 0
            # NaN이나 Inf가 아닌 유효한 포인트 수 계산
            if points is not None:
                self._valid_points = int(np.sum(np.isfinite(points).all(axis=1)))
            else:
                self._valid_points = 0
    
    def get_pointcloud(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """현재 저장된 포인트클라우드 데이터를 반환합니다."""
        with self._lock:
            return self._points, self._colors
    
    def get_pointcloud_with_metadata(self) -> Dict[str, Any]:
        """포인트클라우드 데이터와 메타데이터를 함께 반환합니다."""
        with self._lock:
            return {
                "points": self._points,
                "colors": self._colors,
                "timestamp": self._timestamp,
                "last_updated_utc": self._last_updated,
                "total_points": int(self._total_points),
                "valid_points": int(self._valid_points)
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """포인트클라우드 통계 정보를 반환합니다."""
        with self._lock:
            return {
                "total_points": int(self._total_points),
                "valid_points": int(self._valid_points),
                "invalid_points": int(self._total_points - self._valid_points),
                "timestamp": self._timestamp,
                "last_updated_utc": self._last_updated
            }
    
    def clear(self):
        """저장된 포인트클라우드 데이터를 삭제합니다."""
        with self._lock:
            self._points = None
            self._colors = None
            self._timestamp = None
            self._last_updated = None
            self._total_points = 0
            self._valid_points = 0
    
    def get_downsampled_pointcloud(self, max_points: int = 10000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """다운샘플링된 포인트클라우드를 반환합니다."""
        with self._lock:
            if self._points is None or len(self._points) <= max_points:
                return self._points, self._colors
            
            # 균등하게 다운샘플링
            indices = np.linspace(0, len(self._points) - 1, max_points, dtype=int)
            downsampled_points = self._points[indices]
            downsampled_colors = self._colors[indices] if self._colors is not None else None
            
            return downsampled_points, downsampled_colors
