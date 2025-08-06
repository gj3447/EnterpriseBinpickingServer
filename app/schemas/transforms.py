"""
시스템의 전체 기하학적 상태를 특정 좌표계 기준으로 변환하여 제공하기 위한 DTO를 정의합니다.
"""
from pydantic import BaseModel, Field
from typing import Optional, List

from .aruco import Pose, DetectedMarker, DetectedMarkerPose

class SystemTransformSnapshot(BaseModel):
    """
    [내부용] 시스템의 모든 주요 컴포넌트의 6D Pose를
    지정된 단일 좌표계(frame) 기준으로 표현합니다. 코너 정보를 포함합니다.
    """
    frame: str = Field(..., description="이 스냅샷의 기준이 되는 좌표계 이름 ('camera', 'board', 'robot')")
    board_detected: bool = Field(..., description="좌표계 변환의 기준이 되는 ArUco 보드의 탐지 여부")
    
    board: Optional[Pose] = Field(None, description="기준 좌표계에서의 ArUco 보드 Pose")
    robot: Optional[Pose] = Field(None, description="기준 좌표계에서의 로봇 베이스 Pose")
    camera: Optional[Pose] = Field(None, description="기준 좌표계에서의 카메라 Pose")
    external_markers: List[DetectedMarker] = Field([], description="기준 좌표계에서의 외부 마커 Pose 목록 (코너 정보 포함)")

    class Config:
        # 이 모델의 인스턴스를 다른 모델로 변환할 수 있도록 허용
        from_attributes = True

class SystemTransformSnapshotResponse(BaseModel):
    """
    [API/외부용] 시스템 변환 스냅샷 응답 모델.
    내부용 모델과 달리, 클라이언트에게 불필요한 마커의 2D 코너 정보를 제외합니다.
    """
    frame: str = Field(..., description="이 스냅샷의 기준이 되는 좌표계 이름 ('camera', 'board', 'robot')")
    board_detected: bool = Field(..., description="좌표계 변환의 기준이 되는 ArUco 보드의 탐지 여부")
    
    board: Optional[Pose] = Field(None, description="기준 좌표계에서의 ArUco 보드 Pose")
    robot: Optional[Pose] = Field(None, description="기준 좌표계에서의 로봇 베이스 Pose")
    camera: Optional[Pose] = Field(None, description="기준 좌표계에서의 카메라 Pose")
    external_markers: List[DetectedMarkerPose] = Field([], description="기준 좌표계에서의 외부 마커 Pose 목록 (코너 정보 제외)")
    
    class Config:
        # SystemTransformSnapshot 같은 다른 모델 객체의 속성을 읽어서
        # 이 모델의 인스턴스를 생성하는 것을 허용합니다.
        from_attributes = True
