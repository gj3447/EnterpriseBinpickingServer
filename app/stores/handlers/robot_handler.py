import threading
from typing import Optional, Dict, Any
from app.core.logging import logger


class RobotHandler:
    """URDF 파서가 반환한 로봇 객체를 variant 별로 관리합니다."""

    def __init__(self):
        self._lock = threading.RLock()
        self._urdf_objects: Dict[str, Any] = {}
        self._robot_names: Dict[str, str] = {}
        self._loaded_file_paths: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._last_joint_positions: Dict[str, list[float]] = {}
        self._default_variant: Optional[str] = None
        logger.info("RobotHandler initialized.")

    def set_urdf_object(
        self,
        variant: str,
        urdf_object: Any,
        robot_name: str,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        set_default: bool = False,
    ) -> None:
        """특정 variant의 URDF 객체를 저장합니다."""
        with self._lock:
            self._urdf_objects[variant] = urdf_object
            self._robot_names[variant] = robot_name
            self._loaded_file_paths[variant] = file_path
            self._metadata[variant] = metadata or {}
            logger.info("URDF object(%s) loaded from %s: robot_name=%s", variant, file_path, robot_name)

            if set_default or self._default_variant is None:
                self._default_variant = variant

    def get_default_variant(self) -> Optional[str]:
        with self._lock:
            return self._default_variant

    def set_default_variant(self, variant: str) -> None:
        with self._lock:
            if variant not in self._urdf_objects:
                raise ValueError(f"Variant '{variant}' is not loaded.")
            self._default_variant = variant
            logger.info("Default robot variant set to %s", variant)

    def list_variants(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                variant: {
                    "robot_name": self._robot_names.get(variant),
                    "file_path": self._loaded_file_paths.get(variant),
                    "metadata": dict(self._metadata.get(variant, {})),
                }
                for variant in self._urdf_objects
            }

    def get_urdf_object(self, variant: Optional[str] = None) -> Optional[Any]:
        with self._lock:
            key = variant or self._default_variant
            if key is None:
                return None
            return self._urdf_objects.get(key)

    def get_metadata(self, variant: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            key = variant or self._default_variant
            if key is None:
                return {}
            return dict(self._metadata.get(key, {}))

    def get_robot_name(self, variant: Optional[str] = None) -> Optional[str]:
        with self._lock:
            key = variant or self._default_variant
            if key is None:
                return None
            return self._robot_names.get(key)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            variants = self.list_variants()
            return {
                "loaded_variants": list(variants.keys()),
                "default_variant": self._default_variant,
                "variants": variants,
                "has_last_joint_positions": {
                    variant: variant in self._last_joint_positions
                    for variant in variants.keys()
                },
            }

    def clear(self) -> None:
        with self._lock:
            self._urdf_objects.clear()
            self._robot_names.clear()
            self._loaded_file_paths.clear()
            self._metadata.clear()
            self._last_joint_positions.clear()
            self._default_variant = None
            logger.info("RobotHandler cleared.")

    def set_last_joint_positions(self, variant: str, joint_positions: list[float]) -> None:
        with self._lock:
            self._last_joint_positions[variant] = list(joint_positions)

    def get_last_joint_positions(self, variant: Optional[str] = None) -> Optional[list[float]]:
        with self._lock:
            key = variant or self._default_variant
            if key is None:
                return None
            positions = self._last_joint_positions.get(key)
            return list(positions) if positions is not None else None