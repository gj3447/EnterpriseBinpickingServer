import threading
from typing import Any, Dict, Optional, List, Set
from datetime import datetime

from app.schemas.aruco import Pose, ArucoDetectionResult, DetectedMarker

class ArucoHandler:
    """ArUco 관련 모든 상태 정보를 담당합니다."""
    def __init__(self):
        self._lock = threading.RLock()
        self._board_result: ArucoDetectionResult = ArucoDetectionResult(detected=False, last_updated_utc=datetime.utcnow())
        self._board_markers: Dict[int, DetectedMarker] = {}
        self._external_markers: Dict[int, DetectedMarker] = {}
        self._robot_pose_on_board: Optional[Pose] = None
        self._board_marker_ids: Set[int] = set()  # 보드에 속하는 마커 ID들

    def set_robot_pose(self, pose: Pose):
        with self._lock:
            self._robot_pose_on_board = pose
            
    def set_board_marker_ids(self, marker_ids: Set[int]):
        """보드에 속하는 마커 ID들을 설정합니다."""
        with self._lock:
            self._board_marker_ids = marker_ids
            
    def get_board_marker_ids(self) -> Set[int]:
        """보드에 속하는 마커 ID들을 반환합니다."""
        with self._lock:
            return self._board_marker_ids.copy()
            
    def update_detection_results(
        self,
        board_markers: List[DetectedMarker],
        external_markers: List[DetectedMarker],
        board_pose: Optional[Pose] = None
    ):
        with self._lock:
            self._board_markers = {marker.id: marker for marker in board_markers}
            self._external_markers = {marker.id: marker for marker in external_markers}
            if board_pose:
                self._board_result = ArucoDetectionResult(
                    detected=True, pose=board_pose,
                    detected_markers_count=len(board_markers), last_updated_utc=datetime.utcnow()
                )
            else:
                 self._board_result = ArucoDetectionResult(
                    detected=False, pose=self._board_result.pose,
                    detected_markers_count=0, last_updated_utc=datetime.utcnow()
                )

    def get_board_status(self) -> ArucoDetectionResult:
        with self._lock:
            return self._board_result
    
    def get_board_markers(self) -> List[DetectedMarker]:
        with self._lock:
            return list(self._board_markers.values())

    def get_external_markers(self) -> List[DetectedMarker]:
        with self._lock:
            return list(self._external_markers.values())
    
    def get_marker_by_id(self, marker_id: int) -> Optional[DetectedMarker]:
        with self._lock:
            if marker_id in self._board_markers:
                return self._board_markers[marker_id]
            if marker_id in self._external_markers:
                return self._external_markers[marker_id]
            return None
            
    def get_robot_pose_on_board(self) -> Optional[Pose]:
        with self._lock:
            return self._robot_pose_on_board
        
    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "board_pose": self._board_result.pose,
                "board_markers": list(self._board_markers.values()),
                "external_markers": list(self._external_markers.values()),
                "robot_pose": self._robot_pose_on_board,
                "detected": self._board_result.detected
            }
