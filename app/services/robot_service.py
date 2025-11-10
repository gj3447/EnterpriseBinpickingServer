"""Robot 서비스 백엔드 선택 및 호환성 모듈."""

from typing import Literal, Type

from app.services.robot_service_base import RobotServiceError
from app.services.robot_service_ikpy import RobotServiceIkpy
from app.services.robot_service_pinocchio import RobotService as RobotServicePinocchio

RobotBackend = Literal["pinocchio", "ikpy"]

DEFAULT_BACKEND: RobotBackend = "pinocchio"


def get_robot_service_class(backend: RobotBackend = DEFAULT_BACKEND):
    if backend == "pinocchio":
        return RobotServicePinocchio
    if backend == "ikpy":
        return RobotServiceIkpy
    raise ValueError(f"Unsupported robot service backend: {backend}")


# 기존 import 경로(`from app.services.robot_service import RobotService`)를
# 유지하기 위해 핀노치오 구현을 기본값으로 노출합니다.
RobotService: Type[RobotServicePinocchio] = RobotServicePinocchio


__all__ = [
    "RobotService",
    "RobotServicePinocchio",
    "RobotServiceIkpy",
    "RobotServiceError",
    "get_robot_service_class",
    "DEFAULT_BACKEND",
    "RobotBackend",
]

