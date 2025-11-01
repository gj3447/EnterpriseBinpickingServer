from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from app.services.robot_service import RobotService
from app.dependencies import get_robot_service
from app.core.logging import logger

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
    """로봇 URDF 객체의 정보를 조회합니다."""
    urdf_object = robot_service.get_robot_object()
    if urdf_object is None:
        raise HTTPException(status_code=404, detail="URDF not loaded")
    
    # 객체 타입에 따라 다른 정보 반환
    response = {
        "object_type": type(urdf_object).__name__,
        "available": True
    }
    
    # urdfpy 객체인 경우
    if hasattr(urdf_object, 'name'):
        response["robot_name"] = urdf_object.name
    if hasattr(urdf_object, 'links'):
        response["num_links"] = len(urdf_object.links)
    if hasattr(urdf_object, 'joints'):
        response["num_joints"] = len(urdf_object.joints)
        response["joint_names"] = [j.name for j in urdf_object.joints]
    
    # XML ElementTree인 경우
    elif hasattr(urdf_object, 'getroot'):
        root = urdf_object.getroot()
        response["robot_name"] = root.get('name', 'unknown')
        response["xml_available"] = True
    
    return response
