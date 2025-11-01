import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pinocchio as pin

from app.core.config import settings
from app.core.logging import logger
from app.schemas.robot import (
    RobotIkRequest,
    RobotIkResponse,
    IkCandidateResult,
)
from app.stores.application_store import ApplicationStore


class RobotServiceError(Exception):
    pass


class RobotService:
    """
    로봇 URDF 파일을 로드하고 관리하는 서비스입니다.
    """
    def __init__(self, store: ApplicationStore, urdf_path: Optional[str] = None):
        self.store = store
        self.urdf_path = urdf_path
        self._is_running = False
        logger.info(f"RobotService initialized with URDF path: {urdf_path}")

    async def start(self):
        """서비스를 시작하고 URDF 파일을 로드합니다."""
        if self._is_running:
            logger.warning("RobotService is already running")
            return
        
        logger.info("Starting RobotService...")
        self._is_running = True
        
        if self.urdf_path:
            logger.info(f"Attempting to load URDF from: {self.urdf_path}")
            success = await self.load_urdf(self.urdf_path)
            if success:
                logger.info("RobotService started successfully with URDF loaded")
            else:
                logger.error("RobotService started but URDF loading failed")
        else:
            logger.warning("No URDF path provided. Robot service started without loading URDF.")

    async def stop(self):
        """서비스를 중지합니다."""
        if not self._is_running:
            return
        
        self._is_running = False
        logger.info("RobotService stopped.")

    async def load_urdf(self, file_path: str) -> bool:
        """Pinocchio를 사용하여 URDF 파일을 로드합니다."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"URDF file not found: {file_path}")
                return False
            
            logger.info(f"Loading URDF with Pinocchio: {file_path}")
            try:
                # CPU 바운드(URDF 파싱)를 워커 스레드로 오프로딩
                robot_name, robot_object, metadata = await asyncio.to_thread(
                    self._parse_urdf_with_pinocchio, path
                )
            except ImportError as e:
                logger.error(f"❌ Pinocchio not available: {e}")
                logger.error("Please install Pinocchio in your environment")
                return False
            except Exception as e:
                logger.error(f"❌ Error parsing URDF with Pinocchio: {e}", exc_info=True)
                return False

            # Store에 저장 (이벤트 루프 스레드에서 수행)
            self.store.robot.set_urdf_object(robot_object, robot_name, str(path), metadata=metadata)

            # 저장 확인
            stored_object = self.store.robot.get_urdf_object()
            if stored_object is not None:
                logger.info(f"✅ Pinocchio model successfully stored in ApplicationStore")
                logger.info(f"   Stored object type: {type(stored_object).__name__}")
                return True
            else:
                logger.error("❌ Failed to store Pinocchio model in ApplicationStore")
                return False
                
        except Exception as e:
            logger.error(f"❌ Unexpected error loading URDF file: {e}", exc_info=True)
            return False

    def _parse_urdf_with_pinocchio(self, path: Path) -> Tuple[str, Dict[str, object], Dict[str, object]]:
        """워커 스레드에서 실행되어 URDF를 파싱하고 로봇 객체를 생성합니다."""
        logger.info("Pinocchio library found - starting URDF parsing...")

        model = pin.buildModelFromUrdf(str(path))
        data = model.createData()

        robot_name = model.name if model.name else path.stem

        logger.info(f"✅ Pinocchio model loaded successfully!")
        logger.info(f"   Robot name: {robot_name}")
        logger.info(f"   DOF (degrees of freedom): {model.nq}")
        logger.info(f"   Joints: {model.njoints}")
        logger.info(f"   Bodies: {model.nbodies}")

        metadata: Dict[str, object] = {}

        gripper_joint_name = settings.GRIPPER_JOINT_NAME
        has_gripper_joint = False
        gripper_joint_q_index = None
        gripper_joint_id = None

        if gripper_joint_name:
            try:
                joint_id = model.getJointId(gripper_joint_name)
                joint = model.joints[joint_id]
                if joint.nq == 1 and "Prismatic" in joint.shortname():
                    has_gripper_joint = True
                    gripper_joint_id = joint_id
                    gripper_joint_q_index = joint.idx_q
                else:
                    logger.warning(
                        "Configured gripper joint '%s' is not prismatic (nq=%s, type=%s)",
                        gripper_joint_name,
                        joint.nq,
                        joint.shortname(),
                    )
            except (KeyError, ValueError):
                logger.warning(
                    "Configured gripper joint '%s' not found in model.",
                    gripper_joint_name,
                )

        metadata.update(
            {
                "has_gripper_joint": has_gripper_joint,
                "gripper_joint_name": gripper_joint_name if has_gripper_joint else None,
                "gripper_joint_q_index": gripper_joint_q_index,
                "gripper_joint_id": gripper_joint_id,
            }
        )

        robot_object = {
            'model': model,
            'data': data,
            'urdf_path': str(path),
            'library': 'pinocchio',
            'robot_name': robot_name,
            'dof': model.nq,
            'joint_names': [model.names[i] for i in range(1, model.njoints)],  # 0번은 'universe'
            'joint_limits': {
                'lower': model.lowerPositionLimit.tolist(),
                'upper': model.upperPositionLimit.tolist(),
                'velocity': model.velocityLimit.tolist()
            },
            'has_gripper_joint': has_gripper_joint,
            'gripper_joint_name': gripper_joint_name if has_gripper_joint else None,
            'gripper_joint_q_index': gripper_joint_q_index,
        }

        return robot_name, robot_object, metadata

    def get_robot_object(self):
        """저장된 로봇 객체를 반환합니다."""
        return self.store.robot.get_urdf_object()

    def get_status(self) -> dict:
        """서비스 상태를 반환합니다."""
        return {
            "is_running": self._is_running,
            "robot_status": self.store.robot.get_status()
        }

    def get_metadata(self) -> dict:
        return self.store.robot.get_metadata()

    async def solve_ik(self, ik_request: RobotIkRequest) -> RobotIkResponse:
        return await asyncio.to_thread(self._solve_ik_sync, ik_request)

    # --- 내부 구현 ---

    def _solve_ik_sync(self, ik_request: RobotIkRequest) -> RobotIkResponse:
        robot_object = self.store.robot.get_urdf_object()
        if not robot_object or robot_object.get('library') != 'pinocchio':
            raise RobotServiceError("Pinocchio robot model is not loaded.")

        model: pin.Model = robot_object['model']
        has_gripper_joint = bool(robot_object.get('has_gripper_joint', False))
        gripper_joint_name = robot_object.get('gripper_joint_name')
        gripper_q_index = robot_object.get('gripper_joint_q_index')

        if not ik_request.pose_targets:
            raise RobotServiceError("pose_targets must contain at least one target pose.")

        try:
            frame_id = model.getFrameId(ik_request.target_frame)
        except ValueError as exc:
            raise RobotServiceError(f"Unknown target frame: {ik_request.target_frame}") from exc

        transform_matrix = self._build_coordinate_transform(ik_request)

        grip_offsets = ik_request.grip_offsets or [0.0]
        mode = ik_request.mode

        use_joint = False
        mode_effective = mode
        if mode == "fixed":
            use_joint = False
        elif mode == "prismatic":
            if has_gripper_joint:
                use_joint = True
            else:
                logger.warning("Requested prismatic mode but gripper joint unavailable. Falling back to offset mode.")
                mode_effective = "fixed"
        else:  # auto
            use_joint = has_gripper_joint
            mode_effective = "prismatic" if has_gripper_joint else "fixed"

        if use_joint and gripper_q_index is None:
            raise RobotServiceError(
                "Gripper joint index is not available even though joint usage was requested."
            )

        initial_q = pin.neutral(model)
        if ik_request.initial_joint_positions is not None:
            if len(ik_request.initial_joint_positions) != model.nq:
                raise RobotServiceError(
                    f"initial_joint_positions length {len(ik_request.initial_joint_positions)} does not match model.nq={model.nq}."
                )
            initial_q = np.array(ik_request.initial_joint_positions, dtype=float)

        pose_results: List[IkCandidateResult] = []
        best_result: IkCandidateResult | None = None

        for pose_index, pose_target in enumerate(ik_request.pose_targets):
            base_pose = self._build_target_pose(pose_target, transform_matrix)

            for grip_offset in grip_offsets:
                q_candidate = initial_q.copy()
                local_mode_used = mode_effective
                if use_joint:
                    q_candidate[int(gripper_q_index)] = grip_offset

                data = model.createData()
                current_error = float('inf')
                iterations = 0

                for iteration in range(ik_request.max_iterations):
                    pin.forwardKinematics(model, data, q_candidate)
                    pin.updateFramePlacements(model, data)

                    current_pose = data.oMf[frame_id]

                    target_pose = base_pose
                    if not use_joint:
                        offset_transform = pin.SE3(np.eye(3), np.array([0.0, 0.0, grip_offset]))
                        target_pose = base_pose * offset_transform

                    error_vec = pin.log(current_pose.inverse() * target_pose).vector
                    current_error = float(np.linalg.norm(error_vec))
                    iterations = iteration + 1

                    if current_error <= ik_request.tolerance:
                        break

                    J = pin.computeFrameJacobian(
                        model,
                        data,
                        q_candidate,
                        frame_id,
                        pin.ReferenceFrame.LOCAL,
                    )

                    try:
                        dq, *_ = np.linalg.lstsq(J, -error_vec, rcond=None)
                    except np.linalg.LinAlgError as exc:
                        raise RobotServiceError("Singular Jacobian encountered during IK solving.") from exc

                    dq = ik_request.damping * dq

                    try:
                        q_candidate = pin.integrate(model, q_candidate, dq)
                    except Exception as exc:  # pragma: no cover
                        raise RobotServiceError("Integration failure during IK solving.") from exc

                    if use_joint:
                        q_candidate[int(gripper_q_index)] = grip_offset

                result = IkCandidateResult(
                    pose_index=pose_index,
                    grip_offset=float(grip_offset),
                    error=current_error,
                    iterations=iterations,
                    mode_used="prismatic_joint" if use_joint else "offset",
                    coordinate_mode_used=ik_request.coordinate_mode,
                    joint_positions=q_candidate.tolist(),
                )

                pose_results.append(result)

                if best_result is None or result.error < best_result.error:
                    best_result = result

        if best_result is None:
            raise RobotServiceError("IK solver did not evaluate any candidates.")

        return RobotIkResponse(
            best=best_result,
            candidates=pose_results,
            mode=mode_effective,
            has_gripper_joint=has_gripper_joint,
            gripper_joint_name=gripper_joint_name,
        )

    # --- Helper methods ---

    def _build_target_pose(self, pose_target, axes_matrix: Optional[np.ndarray]) -> pin.SE3:
        translation = np.array(pose_target.translation, dtype=float)
        if translation.shape != (3,):
            raise RobotServiceError("translation must be length 3.")

        rotation_matrix = self._quat_to_matrix(pose_target.rotation_quaternion)

        if axes_matrix is not None:
            translation = axes_matrix @ translation
            rotation_matrix = axes_matrix @ rotation_matrix

        return pin.SE3(rotation_matrix, translation)

    def _build_coordinate_transform(self, ik_request: RobotIkRequest) -> Optional[np.ndarray]:
        if ik_request.coordinate_mode == "base":
            return None

        if ik_request.custom_axes is None:
            raise RobotServiceError("custom_axes must be provided when coordinate_mode='custom'.")

        up_vec = self._axis_to_vector(ik_request.custom_axes.up)
        forward_vec = self._axis_to_vector(ik_request.custom_axes.forward)

        if np.isclose(abs(np.dot(up_vec, forward_vec)), 1.0, atol=1e-6):
            raise RobotServiceError("up axis and forward axis cannot be parallel.")

        up_vec = up_vec / np.linalg.norm(up_vec)
        forward_vec = forward_vec / np.linalg.norm(forward_vec)
        right_vec = np.cross(forward_vec, up_vec)
        right_vec = right_vec / np.linalg.norm(right_vec)
        forward_vec = np.cross(up_vec, right_vec)
        forward_vec = forward_vec / np.linalg.norm(forward_vec)

        rotation_axes = np.column_stack((forward_vec, right_vec, up_vec))
        return rotation_axes

    def _axis_to_vector(self, axis_label: str) -> np.ndarray:
        axis_map = {
            "x": np.array([1.0, 0.0, 0.0]),
            "-x": np.array([-1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "-y": np.array([0.0, -1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
            "-z": np.array([0.0, 0.0, -1.0]),
        }
        try:
            return axis_map[axis_label.lower()]
        except KeyError as exc:
            raise RobotServiceError(f"Unsupported axis label: {axis_label}") from exc

    def _quat_to_matrix(self, quaternion: List[float]) -> np.ndarray:
        if len(quaternion) != 4:
            raise RobotServiceError("rotation_quaternion must contain 4 values [x, y, z, w].")
        x, y, z, w = quaternion
        quat = pin.Quaternion(w, x, y, z)
        quat.normalize()
        return quat.toRotationMatrix()
