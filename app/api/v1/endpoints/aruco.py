from fastapi import APIRouter, Depends
from typing import List

from app.stores.application_store import ApplicationStore
from app.dependencies import get_store
from app.schemas.aruco import ArucoDetectionResult, DetectedMarkerPose

router = APIRouter()

@router.get(
    "/status",
    summary="Get the current status of the main ArUco board",
    response_model=ArucoDetectionResult
)
def get_board_status(store: ApplicationStore = Depends(get_store)):
    """ArUco 보드의 3D 자세(Pose) 추정 결과를 반환합니다."""
    return store.aruco.get_board_status()

@router.get(
    "/board_markers",
    summary="Get markers belonging to the board (Pose only)",
    response_model=List[DetectedMarkerPose]
)
def get_board_markers(store: ApplicationStore = Depends(get_store)):
    """탐지된 마커 중 ArUco 보드에 속하는 마커들의 목록을 반환합니다."""
    return store.aruco.get_board_markers()

@router.get(
    "/external_markers",
    summary="Get external markers (Pose only)",
    response_model=List[DetectedMarkerPose]
)
def get_external_markers(store: ApplicationStore = Depends(get_store)):
    """탐지된 마커 중 ArUco 보드에 속하지 않는 외부 마커들의 목록을 반환합니다."""
    return store.aruco.get_external_markers()
