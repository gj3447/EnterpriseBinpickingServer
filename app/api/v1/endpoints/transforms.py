from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Literal

from app.services.aruco_service import ArucoService
from app.dependencies import get_aruco_service
from app.schemas.aruco import Pose, DetectedMarker, DetectedMarkerPose
from app.schemas.transforms import SystemTransformSnapshot, SystemTransformSnapshotResponse
from app.core.logging import logger

router = APIRouter()

FrameName = Literal["camera", "board", "robot"]

# --- Endpoints ---

@router.get("/board", response_model=Pose)
def get_board_pose(
    frame: FrameName = Query("camera"),
    aruco_service: ArucoService = Depends(get_aruco_service)
):
    """Gets the board's pose relative to the specified coordinate frame."""
    pose = aruco_service.get_board_pose(frame)
    if pose is None:
        raise HTTPException(status_code=404, detail=f"Board pose for frame '{frame}' could not be calculated.")
    return pose

@router.get("/robot", response_model=Pose)
def get_robot_pose(
    frame: FrameName = Query("camera"),
    aruco_service: ArucoService = Depends(get_aruco_service)
):
    """Gets the robot's pose relative to the specified coordinate frame."""
    pose = aruco_service.get_robot_pose(frame)
    if pose is None:
        raise HTTPException(status_code=404, detail=f"Robot pose for frame '{frame}' could not be calculated.")
    return pose

@router.get("/camera", response_model=Pose)
def get_camera_pose(
    frame: FrameName = Query("camera"),
    aruco_service: ArucoService = Depends(get_aruco_service)
):
    """Gets the camera's pose relative to the specified coordinate frame."""
    pose = aruco_service.get_camera_pose(frame)
    if pose is None:
        raise HTTPException(status_code=404, detail=f"Camera pose for frame '{frame}' could not be calculated.")
    return pose

@router.get("/external_markers", response_model=List[DetectedMarkerPose])
def get_external_markers_pose(
    frame: FrameName = Query("camera"),
    aruco_service: ArucoService = Depends(get_aruco_service)
):
    """Gets the poses of all detected external markers relative to the specified coordinate frame."""
    markers = aruco_service.get_external_markers_pose(frame)
    # Convert DetectedMarker -> DetectedMarkerPose
    return [DetectedMarkerPose(id=m.id, pose=m.pose) for m in markers]

@router.get("/all", response_model=SystemTransformSnapshotResponse, summary="Get a complete transformation snapshot")
def get_all_transforms_snapshot(
    frame: FrameName = Query("camera", description="The reference coordinate frame for the snapshot."),
    aruco_service: ArucoService = Depends(get_aruco_service)
):
    """
    Retrieves a complete snapshot of all available transformations
    (board, robot, camera, external markers) relative to the specified
    coordinate frame.
    """
    logger.info(f"Fetching all transforms snapshot relative to '{frame}' frame.")
    
    internal_snapshot = aruco_service.get_transform_snapshot(frame)
    
    if internal_snapshot is None:
         raise HTTPException(status_code=500, detail="Failed to generate transform snapshot.")

    # Manual, explicit conversion from the internal model to the response model.
    # This is more robust than relying on automatic model validation conversion.
    
    response_markers = [
        DetectedMarkerPose(id=m.id, pose=m.pose) 
        for m in internal_snapshot.external_markers
    ]
    
    return SystemTransformSnapshotResponse(
        frame=internal_snapshot.frame,
        board_detected=internal_snapshot.board_detected,
        board=internal_snapshot.board,
        robot=internal_snapshot.robot,
        camera=internal_snapshot.camera,
        external_markers=response_markers
    )
