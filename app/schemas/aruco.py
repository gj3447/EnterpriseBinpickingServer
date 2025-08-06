"""
3D 자세(Pose) 및 ArUco 탐지 결과 API 응답을 위한 DTO를 정의합니다.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class Pose(BaseModel):
    """3D 공간에서의 위치와 자세를 나타내는 표준 모델입니다."""
    translation: List[float] = Field(..., min_length=3, max_length=3)
    orientation_quaternion: List[float] = Field(..., min_length=4, max_length=4)

class Corner(BaseModel):
    """2D 이미지 좌표를 나타내는 모델입니다."""
    x: float
    y: float

class DetectedMarker(BaseModel):
    """
    탐지된 단일 ArUco 마커의 모든 정보(내부 데이터 전달용).
    """
    id: int
    pose: Pose
    corners: List[Corner] = Field(..., min_length=4, max_length=4)

class DetectedMarkerPose(BaseModel):
    """
    탐지된 단일 ArUco 마커의 6D Pose 정보 (API 응답 전용).
    """
    id: int
    pose: Pose

class ArucoDetectionResult(BaseModel):
    """ArUco 보드 탐지 전체 결과를 담는 API 응답 모델입니다."""
    detected: bool
    pose: Optional[Pose] = None
    detected_markers_count: int = 0
    last_updated_utc: datetime
