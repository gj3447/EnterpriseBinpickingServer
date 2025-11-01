from fastapi import APIRouter, Depends, HTTPException

from app.stores.application_store import ApplicationStore
from app.dependencies import get_store
from app.schemas.camera import CameraCalibration

router = APIRouter()

@router.get("/status", summary="Get the combined status of all stores")
def get_store_status(store: ApplicationStore = Depends(get_store)):
    """
    애플리케이션의 모든 주요 상태 정보(장치, 이미지, 캘리브레이션, ArUco 등)를
    한 번에 조회합니다.
    """
    return store.get_status()

@router.get("/events/status", summary="Get the status of all event publications")
def get_event_status(store: ApplicationStore = Depends(get_store)):
    """
    각 이벤트의 상세 정보를 반환합니다:
    - 마지막 발행 시간 (ISO 8601 형식)
    - 현재 FPS (초당 이벤트 수)
    - 총 발행 횟수
    - 윈도우 내 이벤트 수
    """
    return store.events.get_status()

@router.get("/events/fps", summary="Get FPS for all events")
def get_event_fps(store: ApplicationStore = Depends(get_store)):
    """
    모든 이벤트의 현재 FPS(초당 발행 횟수)를 간단한 딕셔너리 형태로 반환합니다.
    성능 모니터링에 유용합니다.
    """
    return store.events.get_all_fps()


@router.get(
    "/calibration",
    summary="Get current camera calibration data from the store",
    response_model=CameraCalibration
)
def get_calibration_data(store: ApplicationStore = Depends(get_store)):
    """
    Store에 저장된 최신 카메라 캘리브레이션 데이터를 반환합니다.
    """
    calib_data = store.calibration.get_data()
    if calib_data is None:
        raise HTTPException(
            status_code=404,
            detail="Calibration data not available. The server may not have synced with the camera API yet."
        )
    return calib_data
