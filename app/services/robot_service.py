"""Robot 서비스 백엔드 선택 및 호환성 모듈."""

from typing import Literal, Type

from app.services.robot_service_base import RobotServiceError
from app.services.robot_service_ikpy import RobotServiceIkpy

RobotBackend = Literal["ikpy"]

DEFAULT_BACKEND: RobotBackend = "ikpy"


def get_robot_service_class(backend: RobotBackend = DEFAULT_BACKEND):
    if backend == "ikpy":
        return RobotServiceIkpy
    raise ValueError(f"Unsupported robot service backend: {backend}")


# 기존 import 경로(`from app.services.robot_service import RobotService`)를
# 유지하기 위해 ikpy 구현을 기본값으로 노출합니다.
RobotService: Type[RobotServiceIkpy] = RobotServiceIkpy


__all__ = [
    "RobotService",
    "RobotServiceIkpy",
    "RobotServiceError",
    "get_robot_service_class",
    "DEFAULT_BACKEND",
    "RobotBackend",
]

