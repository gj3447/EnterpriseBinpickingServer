from fastapi import APIRouter, Depends, HTTPException

from app.stores.application_store import ApplicationStore
from app.dependencies import get_store

router = APIRouter()

@router.get(
    "/status",
    summary="Get current device status",
    description="Returns the latest device status information (device info, active stream configuration) fetched from the camera API."
)
def get_device_status(store: ApplicationStore = Depends(get_store)):
    """
    Store에 저장된 최신 장치 상태 정보를 반환합니다.
    """
    device_status = store.device.get_full_status()
    # Pydantic 모델이 None이면 아직 동기화되지 않은 것
    if device_status is None:
        raise HTTPException(
            status_code=404,
            detail="Device status not available. The server may not have synced with the camera API yet."
        )
    return device_status
