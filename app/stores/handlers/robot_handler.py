import threading
from typing import Optional, Dict, Any
from app.core.logging import logger


class RobotHandler:
    """
    로봇 URDF 객체를 관리하는 핸들러입니다.
    URDF 파싱 라이브러리가 반환하는 원본 객체를 그대로 저장합니다.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._urdf_object: Optional[Any] = None  # URDF 파싱 라이브러리의 객체
        self._robot_name: Optional[str] = None
        self._loaded_file_path: Optional[str] = None
        logger.info("RobotHandler initialized.")

    def set_urdf_object(self, urdf_object: Any, robot_name: str, file_path: str) -> None:
        """URDF 객체를 설정합니다."""
        with self._lock:
            self._urdf_object = urdf_object
            self._robot_name = robot_name
            self._loaded_file_path = file_path
            logger.info(f"URDF object loaded from {file_path}: robot_name={robot_name}")

    def get_urdf_object(self) -> Optional[Any]:
        """URDF 객체를 반환합니다. (IK 등의 계산에 직접 사용 가능)"""
        with self._lock:
            return self._urdf_object

    def get_robot_name(self) -> Optional[str]:
        """로봇 이름을 반환합니다."""
        with self._lock:
            return self._robot_name

    def get_status(self) -> Dict[str, Any]:
        """현재 상태를 반환합니다."""
        with self._lock:
            return {
                "loaded": self._urdf_object is not None,
                "file_path": self._loaded_file_path,
                "robot_name": self._robot_name,
                "object_type": type(self._urdf_object).__name__ if self._urdf_object else None
            }

    def clear(self) -> None:
        """저장된 URDF 데이터를 초기화합니다."""
        with self._lock:
            self._urdf_object = None
            self._robot_name = None
            self._loaded_file_path = None
            logger.info("RobotHandler cleared.")