from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional

from app.stores.application_store import ApplicationStore
from app.dependencies import get_store

router = APIRouter()

@router.get(
    "camera/status",
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


@router.get(
    "/robot/status",
    summary="Get robot loading status",
    description="Returns the current robot loading status and basic information."
)
def get_robot_status(store: ApplicationStore = Depends(get_store)) -> Dict[str, Any]:
    """
    로봇 로딩 상태와 기본 정보를 반환합니다.
    """
    robot_status = store.robot.get_status()
    return robot_status


@router.get(
    "/robot/info",
    summary="Get detailed robot information",
    description="Returns detailed robot information including DOF, joints, and limits if robot is loaded."
)
def get_robot_info(store: ApplicationStore = Depends(get_store)) -> Dict[str, Any]:
    """
    로봇의 상세 정보를 반환합니다 (DOF, 조인트, 제한값 등).
    """
    robot_object = store.robot.get_urdf_object()
    
    if robot_object is None:
        raise HTTPException(
            status_code=404,
            detail="Robot not loaded. The URDF file may not have been loaded successfully."
        )
    
    # Pinocchio 모델에서 상세 정보 추출
    if isinstance(robot_object, dict) and robot_object.get('library') == 'pinocchio':
        return {
            "robot_name": robot_object.get('robot_name'),
            "urdf_path": robot_object.get('urdf_path'),
            "library": robot_object.get('library'),
            "dof": robot_object.get('dof'),
            "joint_names": robot_object.get('joint_names', []),
            "joint_limits": robot_object.get('joint_limits', {}),
            "model_info": {
                "njoints": robot_object.get('model').njoints if robot_object.get('model') else None,
                "nbodies": robot_object.get('model').nbodies if robot_object.get('model') else None,
                "has_model": robot_object.get('model') is not None,
                "has_data": robot_object.get('data') is not None
            }
        }
    else:
        # 다른 라이브러리나 형식의 경우
        return {
            "robot_object_type": type(robot_object).__name__,
            "robot_object": str(robot_object)[:200] + "..." if len(str(robot_object)) > 200 else str(robot_object)
        }


@router.get(
    "/robot/pinocchio",
    summary="Get Pinocchio model access information",
    description="Returns information about accessing the Pinocchio model for advanced robotics computations."
)
def get_pinocchio_info(store: ApplicationStore = Depends(get_store)) -> Dict[str, Any]:
    """
    Pinocchio 모델에 직접 접근하기 위한 정보를 반환합니다.
    """
    robot_object = store.robot.get_urdf_object()
    
    if robot_object is None:
        raise HTTPException(
            status_code=404,
            detail="Robot not loaded."
        )
    
    if not isinstance(robot_object, dict) or robot_object.get('library') != 'pinocchio':
        raise HTTPException(
            status_code=400,
            detail="Robot was not loaded with Pinocchio library."
        )
    
    model = robot_object.get('model')
    data = robot_object.get('data')
    
    if model is None or data is None:
        raise HTTPException(
            status_code=500,
            detail="Pinocchio model or data is not available."
        )
    
    return {
        "library": "pinocchio",
        "robot_name": robot_object.get('robot_name'),
        "capabilities": {
            "forward_kinematics": True,
            "inverse_kinematics": True,
            "dynamics": True,
            "collision_detection": True,
            "jacobian_computation": True
        },
        "model_details": {
            "name": model.name,
            "nq": model.nq,  # configuration space dimension
            "nv": model.nv,  # velocity space dimension  
            "njoints": model.njoints,
            "nbodies": model.nbodies,
            "nframes": model.nframes if hasattr(model, 'nframes') else None
        },
        "usage_note": "Use store.robot.get_urdf_object()['model'] and ['data'] to access Pinocchio objects for IK/FK computations."
    }
