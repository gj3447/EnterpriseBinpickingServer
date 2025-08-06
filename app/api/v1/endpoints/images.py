from fastapi import APIRouter, Depends, HTTPException, Response

from app.services.image_service import ImageService
from app.stores.application_store import ApplicationStore
from app.dependencies import get_image_service, get_store
from app.core.logging import logger

router = APIRouter()

@router.get("/color.jpg", summary="Get Latest Color Image")
def get_color_image(store: ApplicationStore = Depends(get_store)):
    """가장 최신의 변환된 컬러 이미지를 JPEG 형식으로 반환합니다."""
    jpeg_data = store.images.get_color_jpeg()
    if jpeg_data is None:
        raise HTTPException(status_code=404, detail="Color JPEG image not found in store. It might not have been processed yet.")
    return Response(content=jpeg_data, media_type="image/jpeg")

@router.get("/depth.jpg", summary="Get Latest Depth Image")
def get_depth_image(store: ApplicationStore = Depends(get_store)):
    """가장 최신의 변환된 뎁스 이미지를 정규화된 JPEG 형식으로 반환합니다."""
    jpeg_data = store.images.get_depth_jpeg()
    if jpeg_data is None:
        raise HTTPException(status_code=404, detail="Depth JPEG image not found in store. It might not have been processed yet.")
    return Response(content=jpeg_data, media_type="image/jpeg")

@router.get("/board_perspective.jpg", summary="Get Perspective-Corrected Board Image")
def get_board_perspective_image(store: ApplicationStore = Depends(get_store)):
    """미리 계산된, ArUco 보드 기준 원근 보정 이미지를 반환합니다."""
    jpeg_data = store.images.get_board_perspective_image()
    if jpeg_data is None:
        raise HTTPException(status_code=404, detail="Perspective corrected image not found in store. Aruco board might not be detected.")
    return Response(content=jpeg_data, media_type="image/jpeg")

@router.get("/aruco_debug.jpg", summary="Get ArUco Debug Image")
def get_aruco_debug_image(store: ApplicationStore = Depends(get_store)):
    """미리 계산된, ArUco 정보가 시각화된 디버그 이미지를 반환합니다."""
    jpeg_data = store.images.get_aruco_debug_image()
    if jpeg_data is None:
        raise HTTPException(status_code=404, detail="Aruco debug image not found in store. It might not have been processed yet.")
    return Response(content=jpeg_data, media_type="image/jpeg")

@router.get("/color/raw", summary="Get Latest Raw Color Image Data")
def get_raw_color_image(store: ApplicationStore = Depends(get_store)):
    """
    가장 최신의 원본 컬러 이미지(BGR, uint8)를 raw bytes 형식으로 반환합니다.
    응답 헤더에 이미지의 shape와 dtype 정보가 포함됩니다.
    """
    stored_image = store.camera_raw.get_color_image()
    if stored_image is None:
        raise HTTPException(status_code=404, detail="Raw color image not found.")
    
    image = stored_image.data
    headers = {
        "X-Image-Height": str(image.shape[0]),
        "X-Image-Width": str(image.shape[1]),
        "X-Image-Channels": str(image.shape[2]),
        "X-Image-Dtype": str(image.dtype),
        "X-Image-Timestamp": str(stored_image.timestamp)
    }
    
    return Response(content=image.tobytes(), media_type="application/octet-stream", headers=headers)

@router.get("/depth/raw", summary="Get Latest Raw Depth Image Data")
def get_raw_depth_image(store: ApplicationStore = Depends(get_store)):
    """
    가장 최신의 원본 뎁스 이미지(Z16, uint16)를 raw bytes 형식으로 반환합니다.
    응답 헤더에 이미지의 shape와 dtype 정보가 포함됩니다.
    """
    stored_image = store.camera_raw.get_depth_image()
    if stored_image is None:
        raise HTTPException(status_code=404, detail="Raw depth image not found.")
    
    image = stored_image.data
    headers = {
        "X-Image-Height": str(image.shape[0]),
        "X-Image-Width": str(image.shape[1]),
        "X-Image-Dtype": str(image.dtype),
        "X-Image-Timestamp": str(stored_image.timestamp)
    }
    
    return Response(content=image.tobytes(), media_type="application/octet-stream", headers=headers)
