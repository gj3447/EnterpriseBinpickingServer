from fastapi import APIRouter, Depends, HTTPException, Response, Query

from app.services.image_service import ImageService
from app.stores.application_store import ApplicationStore
from app.dependencies import get_image_service, get_store

router = APIRouter()

@router.get("/board.jpg")
def get_board_mask_image(
    store: ApplicationStore = Depends(get_store),
    image_service: ImageService = Depends(get_image_service)
):
    stored_image = store.camera_raw.get_color_image()
    aruco_status = store.aruco.get_board_status()
    calib_data = store.calibration.get_data()
    
    if not (stored_image and calib_data and aruco_status.detected and aruco_status.pose):
        raise HTTPException(status_code=404, detail="Data not ready for board mask generation.")
    
    jpeg_data = image_service.get_board_mask_image_as_jpeg(stored_image.data, aruco_status.pose, calib_data)
    if jpeg_data is None: raise HTTPException(status_code=500, detail="Failed to generate board mask.")
    return Response(content=jpeg_data, media_type="image/jpeg")

@router.get("/marker.jpg")
def get_single_marker_mask_image(
    id: int = Query(..., description="The ID of the marker to generate a mask for."),
    store: ApplicationStore = Depends(get_store),
    image_service: ImageService = Depends(get_image_service)
):
    stored_image = store.camera_raw.get_color_image()
    calib_data = store.calibration.get_data()
    marker = store.aruco.get_marker_by_id(id)
    
    if not (stored_image and calib_data and marker):
        raise HTTPException(status_code=404, detail=f"Data not ready for marker mask generation for ID {id}.")
    
    jpeg_data = image_service.get_single_marker_mask_image_as_jpeg(stored_image.data, marker, calib_data)
    if jpeg_data is None: raise HTTPException(status_code=500, detail=f"Failed to generate mask for marker ID {id}.")
    return Response(content=jpeg_data, media_type="image/jpeg")
