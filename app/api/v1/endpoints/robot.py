from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger as app_logger
from app.dependencies import (
    get_robot_service,
    get_robot_service_ikpy,
)
from app.schemas.robot import (
    PoseTarget,
    RobotIkDownwardRequest,
    RobotIkRequest,
    RobotIkResponse,
)
from app.services.robot_service import (
    RobotBackend,
    RobotServiceError,
    RobotServiceIkpy,
)


router = APIRouter()


def _validate_pose_targets_height(request: RobotIkRequest) -> None:
    """Ensures all pose targets stay above the configured minimum Z height."""
    min_z = settings.IK_MIN_Z
    if min_z is None:
        return

    for idx, pose in enumerate(request.pose_targets):
        current_z = pose.translation[2]
        if current_z < min_z:
            app_logger.warning(
                "Pose target at index %s lies below IK_MIN_Z (z=%.4f < %.4f); clamping to min height.",
                idx,
                current_z,
                min_z,
            )
            # Clamp in-place so downstream services see the adjusted value
            pose.translation[2] = min_z


async def _solve_ik(
    service: RobotServiceIkpy,
    request: RobotIkRequest,
) -> RobotIkResponse:
    _validate_pose_targets_height(request)
    try:
        return await service.solve_ik(request)
    except RobotServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _build_urdf_response(
    service: RobotServiceIkpy,
    variant: Optional[str],
) -> Dict[str, Any]:
    urdf_object = service.get_robot_object(variant)
    if urdf_object is None:
        raise HTTPException(status_code=404, detail="URDF not loaded")

    if not isinstance(urdf_object, dict):
        raise HTTPException(status_code=500, detail="URDF object has unexpected structure.")

    metadata = service.get_metadata(variant)
    default_variant = service.store.robot.get_default_variant()

    response: Dict[str, Any] = {
        "robot_name": urdf_object.get("robot_name"),
        "library": urdf_object.get("library"),
        "dof": urdf_object.get("dof"),
        "joint_names": urdf_object.get("joint_names", []),
        "joint_limits": urdf_object.get("joint_limits", {}),
        "urdf_path": urdf_object.get("urdf_path"),
        "has_gripper_joint": metadata.get("has_gripper_joint", False),
        "gripper_joint_name": metadata.get("gripper_joint_name"),
        "configured_urdf_path": metadata.get("configured_urdf_path"),
        "configured_urdf_mode": metadata.get("configured_urdf_mode"),
        "configured_urdf_variant": metadata.get("configured_urdf_variant"),
        "configured_gripper_joint_name": metadata.get("configured_gripper_joint_name"),
        "requested_variant": variant if variant is not None else default_variant,
    }

    return response


def _build_downward_request(request: RobotIkDownwardRequest) -> RobotIkRequest:
    def _axis_label_to_vector(label: str) -> List[float]:
        mapping = {
            "x": [1.0, 0.0, 0.0],
            "-x": [-1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "-y": [0.0, -1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
            "-z": [0.0, 0.0, -1.0],
        }
        try:
            return mapping[label.lower()]
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Unsupported axis label '{label}'") from exc

    downward_quaternion = [1.0, 0.0, 0.0, 0.0]

    hover_offset: Optional[List[float]] = None
    if request.hover_height > 0.0:
        if request.coordinate_mode == "base":
            hover_offset = [0.0, 0.0, request.hover_height]
        else:
            axes = request.custom_axes
            if axes is None:
                raise HTTPException(status_code=400, detail="custom_axes must be provided when coordinate_mode is 'custom'.")
            up_vector = _axis_label_to_vector(axes.up)
            hover_offset = [component * request.hover_height for component in up_vector]

    pose_targets = [
        PoseTarget(
            translation=[
                translation[0] + (hover_offset[0] if hover_offset else 0.0),
                translation[1] + (hover_offset[1] if hover_offset else 0.0),
                translation[2] + (hover_offset[2] if hover_offset else 0.0),
            ],
            rotation_quaternion=downward_quaternion.copy(),
        )
        for translation in request.translations
    ]

    return RobotIkRequest(
        target_frame=request.target_frame,
        pose_targets=pose_targets,
        grip_offsets=request.grip_offsets,
        mode=request.mode,
        coordinate_mode=request.coordinate_mode,
        urdf_variant=request.urdf_variant,
        custom_axes=request.custom_axes,
        initial_joint_positions=request.initial_joint_positions,
        max_iterations=request.max_iterations,
        tolerance=request.tolerance,
        damping=request.damping,
    )


def _build_downward_request_for_ikpy(request: RobotIkDownwardRequest) -> RobotIkRequest:
    adjusted_request = request.model_copy(
        update={
            "urdf_variant": "fixed",
        }
    )
    base_request = _build_downward_request(adjusted_request)
    return base_request.model_copy(
        update={
            "target_frame": settings.IKPY_END_EFFECTOR_FRAME,
            "urdf_variant": "fixed",
        }
    )


@router.get("/status", response_model=Dict[str, Any])
def get_robot_status(
    robot_service: RobotServiceIkpy = Depends(get_robot_service),
):
    """설정된 기본 백엔드의 로봇 상태를 조회합니다."""
    return robot_service.get_status()


@router.get("/status/pinocchio", response_model=Dict[str, Any])
def get_robot_status_pinocchio():
    """Pinocchio backend is disabled."""
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Pinocchio backend has been disabled. Use ikpy endpoints instead.",
    )


@router.get("/status/ikpy", response_model=Dict[str, Any])
def get_robot_status_ikpy(
    robot_service_ikpy: RobotServiceIkpy = Depends(get_robot_service_ikpy),
):
    """IKPy 백엔드의 로봇 상태를 조회합니다."""
    return robot_service_ikpy.get_status()


@router.get("/status/{backend}", response_model=Dict[str, Any])
def get_robot_status_by_backend(
    backend: RobotBackend,
    robot_service_ikpy: RobotServiceIkpy = Depends(get_robot_service_ikpy),
):
    """백엔드별 로봇 상태를 조회합니다."""
    if backend != "ikpy":
        raise HTTPException(status_code=400, detail="Only 'ikpy' backend is available.")
    return robot_service_ikpy.get_status()


@router.get("/urdf", response_model=Dict[str, Any])
def get_robot_urdf_info(
    variant: Optional[str] = Query(None, description="확인할 URDF variant (없으면 기본값)"),
    robot_service: RobotServiceIkpy = Depends(get_robot_service),
):
    """기본 백엔드에 대한 URDF 정보를 반환합니다."""
    return _build_urdf_response(robot_service, variant)


@router.get("/urdf/ikpy", response_model=Dict[str, Any])
def get_robot_urdf_info_ikpy(
    variant: Optional[str] = Query(None, description="확인할 URDF variant (없으면 기본값)"),
    robot_service_ikpy: RobotServiceIkpy = Depends(get_robot_service_ikpy),
):
    """IKPy 백엔드에 대한 URDF 정보를 반환합니다."""
    return _build_urdf_response(robot_service_ikpy, variant)


@router.post("/ik", response_model=RobotIkResponse)
async def solve_robot_ik(
    request: RobotIkRequest,
    robot_service: RobotServiceIkpy = Depends(get_robot_service),
):
    _validate_pose_targets_height(request)
    return await robot_service.solve_ik(request)


@router.post("/ik/downward", response_model=RobotIkResponse)
async def solve_robot_ik_downward(
    request: RobotIkDownwardRequest,
    robot_service: RobotServiceIkpy = Depends(get_robot_service),
):
    base_request = _build_downward_request(request)
    return await _solve_ik(robot_service, base_request)


@router.post("/ik/ikpy/downward", response_model=RobotIkResponse)
async def solve_robot_ik_downward_ikpy(
    request: RobotIkDownwardRequest,
    robot_service_ikpy: RobotServiceIkpy = Depends(get_robot_service_ikpy),
):
    base_request = _build_downward_request_for_ikpy(request)
    return await _solve_ik(robot_service_ikpy, base_request)
