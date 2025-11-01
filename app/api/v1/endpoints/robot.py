from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from app.dependencies import get_robot_service
from app.schemas.robot import RobotIkRequest, RobotIkResponse
from app.services.robot_service import RobotService, RobotServiceError

router = APIRouter()

@router.get("/status", response_model=Dict[str, Any])
def get_robot_status(
    robot_service: RobotService = Depends(get_robot_service)
):
    """로봇 URDF 로드 상태를 조회합니다."""
    return robot_service.get_status()

@router.get("/urdf", response_model=Dict[str, Any])
def get_robot_urdf_info(
    robot_service: RobotService = Depends(get_robot_service)
):
    """Pinocchio 로드 결과를 기반으로 URDF 정보를 반환합니다."""
    urdf_object = robot_service.get_robot_object()
    if urdf_object is None:
        raise HTTPException(status_code=404, detail="URDF not loaded")

    if not isinstance(urdf_object, dict) or urdf_object.get("library") != "pinocchio":
        raise HTTPException(status_code=400, detail="URDF is not managed by Pinocchio model.")

    metadata = robot_service.get_metadata()

    return {
        "robot_name": urdf_object.get("robot_name"),
        "library": urdf_object.get("library"),
        "dof": urdf_object.get("dof"),
        "joint_names": urdf_object.get("joint_names", []),
        "joint_limits": urdf_object.get("joint_limits", {}),
        "urdf_path": urdf_object.get("urdf_path"),
        "has_gripper_joint": metadata.get("has_gripper_joint", False),
        "gripper_joint_name": metadata.get("gripper_joint_name"),
    }


@router.post("/ik", response_model=RobotIkResponse)
async def solve_robot_ik(
    request: RobotIkRequest,
    robot_service: RobotService = Depends(get_robot_service),
):
    try:
        return await robot_service.solve_ik(request)
    except RobotServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
