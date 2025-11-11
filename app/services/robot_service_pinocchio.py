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
from app.services.robot_service_base import RobotServiceError
from app.stores.application_store import ApplicationStore


class RobotService:
    """Pinocchio 기반 로봇 URDF 파일 로드 및 IK 서비스."""

    def __init__(
        self,
        store: ApplicationStore,
        fixed_urdf_path: Optional[str] = None,
        prismatic_urdf_path: Optional[str] = None,
    ):
        self.store = store
        self.fixed_urdf_path = fixed_urdf_path
        self.prismatic_urdf_path = prismatic_urdf_path
        self._is_running = False
        logger.info(
            "RobotService initialized with fixed=%s, prismatic=%s",
            fixed_urdf_path,
            prismatic_urdf_path,
        )

    async def start(self):
        """서비스를 시작하고 URDF 파일을 로드합니다."""
        if self._is_running:
            logger.warning("RobotService is already running")
            return

        logger.info("Starting RobotService...")
        self._is_running = True

        any_loaded = False

        if self.fixed_urdf_path:
            logger.info("Attempting to load fixed URDF from: %s", self.fixed_urdf_path)
            success_fixed = await self.load_urdf_variant("fixed", self.fixed_urdf_path, set_default=True)
            any_loaded = any_loaded or success_fixed
        else:
            logger.warning("No fixed URDF path provided. Skipping fixed variant load.")

        if self.prismatic_urdf_path:
            logger.info("Attempting to load prismatic URDF from: %s", self.prismatic_urdf_path)
            success_prismatic = await self.load_urdf_variant("prismatic", self.prismatic_urdf_path)
            any_loaded = any_loaded or success_prismatic
        else:
            logger.info("No prismatic URDF path provided. Skipping prismatic variant load.")

        if any_loaded:
            self._configure_default_variant()
            logger.info(
                "RobotService started with variants=%s (default=%s)",
                list(self.store.robot.list_variants().keys()),
                self.store.robot.get_default_variant(),
            )
        else:
            logger.error("RobotService started but no URDF variants could be loaded.")

    async def stop(self):
        """서비스를 중지합니다."""
        if not self._is_running:
            return

        self._is_running = False
        logger.info("RobotService stopped.")

    async def load_urdf_variant(self, variant: str, file_path: str, set_default: bool = False) -> bool:
        """Pinocchio를 사용하여 특정 variant의 URDF 파일을 로드합니다."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error("URDF file not found for variant %s: %s", variant, file_path)
                return False

            logger.info("Loading %s URDF with Pinocchio: %s", variant, file_path)
            try:
                robot_name, robot_object, metadata = await asyncio.to_thread(
                    self._parse_urdf_with_pinocchio,
                    path,
                    variant,
                    settings.GRIPPER_JOINT_NAME if variant == "prismatic" else None,
                )
            except ImportError as e:
                logger.error("❌ Pinocchio not available while loading %s: %s", variant, e)
                logger.error("Please install Pinocchio in your environment")
                return False
            except Exception as e:
                logger.error("❌ Error parsing %s URDF with Pinocchio: %s", variant, e, exc_info=True)
                return False

            self.store.robot.set_urdf_object(
                variant,
                robot_object,
                robot_name,
                str(path),
                metadata=metadata,
                set_default=set_default,
            )
            return True

        except Exception as e:
            logger.error("❌ Unexpected error loading %s URDF file: %s", variant, e, exc_info=True)
            return False

    def _configure_default_variant(self) -> None:
        """설정에 따라 기본 variant를 결정합니다."""
        variants = self.store.robot.list_variants()
        if not variants:
            logger.error("No URDF variants available to configure default.")
            return

        mode = settings.ROBOT_URDF_MODE
        preference: List[str]
        if mode == "fixed":
            preference = ["fixed", "prismatic"]
        elif mode == "prismatic":
            preference = ["prismatic", "fixed"]
        else:  # auto
            preference = ["prismatic", "fixed"]

        for candidate in preference:
            if candidate in variants:
                self.store.robot.set_default_variant(candidate)
                return

        # Fallback: first available variant
        self.store.robot.set_default_variant(next(iter(variants.keys())))

    def _parse_urdf_with_pinocchio(
        self,
        path: Path,
        variant: str,
        gripper_joint_name: Optional[str],
    ) -> Tuple[str, Dict[str, object], Dict[str, object]]:
        """워커 스레드에서 실행되어 URDF를 파싱하고 로봇 객체를 생성합니다."""
        logger.info("Pinocchio library found - starting URDF parsing for %s...", variant)

        model = pin.buildModelFromUrdf(str(path))
        data = model.createData()

        robot_name = model.name if model.name else path.stem

        logger.info("✅ Pinocchio model loaded successfully (%s)!", variant)
        logger.info("   Robot name: %s", robot_name)
        logger.info("   DOF (degrees of freedom): %s", model.nq)
        logger.info("   Joints: %s", model.njoints)
        logger.info("   Bodies: %s", model.nbodies)

        metadata: Dict[str, object] = {"variant": variant}

        has_gripper_joint = False
        gripper_joint_q_index = None
        gripper_joint_id = None

        if gripper_joint_name:
            try:
                joint_id = model.getJointId(gripper_joint_name)
                joint = model.joints[joint_id]
                joint_name = joint.shortname()
                if joint.nq == 1 and ("Prismatic" in joint_name or joint_name.startswith("JointModelP")):
                    has_gripper_joint = True
                    gripper_joint_id = joint_id
                    gripper_joint_q_index = joint.idx_q
                else:
                    logger.warning(
                        "Configured gripper joint '%s' is not prismatic (variant=%s, nq=%s, type=%s)",
                        gripper_joint_name,
                        variant,
                        joint.nq,
                        joint_name,
                    )
            except (KeyError, ValueError):
                logger.warning(
                    "Configured gripper joint '%s' not found in model for variant %s.",
                    gripper_joint_name,
                    variant,
                )

        approx_workspace_radius = max(
            (np.linalg.norm(frame.placement.translation) for frame in model.frames),
            default=0.0,
        )

        metadata.update(
            {
                "has_gripper_joint": has_gripper_joint,
                "gripper_joint_name": gripper_joint_name if has_gripper_joint else None,
                "gripper_joint_q_index": gripper_joint_q_index,
                "gripper_joint_id": gripper_joint_id,
                "configured_urdf_mode": settings.ROBOT_URDF_MODE,
                "configured_urdf_variant": variant,
                "configured_urdf_path": str(path),
                "configured_gripper_joint_name": gripper_joint_name,
                "approx_workspace_radius": approx_workspace_radius,
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
            'approx_workspace_radius': approx_workspace_radius,
            'has_gripper_joint': has_gripper_joint,
            'gripper_joint_name': gripper_joint_name if has_gripper_joint else None,
            'gripper_joint_q_index': gripper_joint_q_index,
        }

        return robot_name, robot_object, metadata

    def get_robot_object(self, variant: Optional[str] = None):
        """저장된 로봇 객체를 variant 기준으로 반환합니다."""
        return self.store.robot.get_urdf_object(variant)

    def get_status(self) -> dict:
        """서비스 상태를 반환합니다."""
        return {
            "is_running": self._is_running,
            "robot_status": self.store.robot.get_status()
        }

    def get_metadata(self, variant: Optional[str] = None) -> dict:
        return self.store.robot.get_metadata(variant)

    async def solve_ik(self, ik_request: RobotIkRequest) -> RobotIkResponse:
        return await asyncio.to_thread(self._solve_ik_sync, ik_request)

    # --- 내부 구현 ---

    def _solve_ik_sync(self, ik_request: RobotIkRequest) -> RobotIkResponse:
        variant = ik_request.urdf_variant or self.store.robot.get_default_variant()
        if variant is None:
            raise RobotServiceError("No robot variant is currently loaded.")

        pose_summaries = [
            {
                "translation": pose.translation,
                "rotation_quaternion": pose.rotation_quaternion,
            }
            for pose in ik_request.pose_targets
        ]

        logger.info(
            "IK request received | variant={} target_frame={} mode={} coordinate_mode={} poses={} grip_offsets={} initial_q={}",
            variant,
            ik_request.target_frame,
            ik_request.mode,
            ik_request.coordinate_mode,
            pose_summaries,
            ik_request.grip_offsets,
            "yes" if ik_request.initial_joint_positions is not None else "no",
        )

        robot_object = self.store.robot.get_urdf_object(variant)
        if not robot_object or robot_object.get('library') != 'pinocchio':
            raise RobotServiceError(f"Pinocchio robot model is not loaded for variant '{variant}'.")

        model: pin.Model = robot_object['model']
        nv = model.nv
        identity_nv = np.eye(nv)
        has_gripper_joint = bool(robot_object.get('has_gripper_joint', False))
        gripper_joint_name = robot_object.get('gripper_joint_name')
        gripper_q_index = robot_object.get('gripper_joint_q_index')
        joint_limits = robot_object.get('joint_limits', {})
        supports_joint_jacobians = hasattr(pin, "computeJointJacobians")
        approx_workspace_radius = float(robot_object.get('approx_workspace_radius') or 0.0)
        lower_limits_raw = joint_limits.get('lower')
        upper_limits_raw = joint_limits.get('upper')
        limit_arrays_valid = (
            isinstance(lower_limits_raw, (list, tuple, np.ndarray))
            and isinstance(upper_limits_raw, (list, tuple, np.ndarray))
            and len(lower_limits_raw) == model.nq
            and len(upper_limits_raw) == model.nq
        )
        if limit_arrays_valid:
            lower_limits = np.array(lower_limits_raw, dtype=float)
            upper_limits = np.array(upper_limits_raw, dtype=float)
            finite_lower_mask = np.isfinite(lower_limits)
            finite_upper_mask = np.isfinite(upper_limits)
        else:
            lower_limits = upper_limits = None
            finite_lower_mask = finite_upper_mask = None

        if not ik_request.pose_targets:
            raise RobotServiceError("pose_targets must contain at least one target pose.")

        try:
            frame_id = model.getFrameId(ik_request.target_frame)
        except ValueError as exc:
            raise RobotServiceError(f"Unknown target frame: {ik_request.target_frame}") from exc

        transform_matrix = self._build_coordinate_transform(ik_request)
        logger.debug(
            "Pinocchio IK transform computed | variant=%s coordinate_mode=%s transform=%s",
            variant,
            ik_request.coordinate_mode,
            transform_matrix.tolist() if transform_matrix is not None else None,
        )

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
        base_lambda = float(np.clip(ik_request.damping, 1e-6, 1e2))
        lambda_min = 1e-6
        lambda_max = 1e2
        lambda_increase_factor = 2.0
        lambda_decrease_factor = 0.7
        rotation_weight = 0.45
        error_weights = np.array([1.0, 1.0, 1.0, rotation_weight, rotation_weight, rotation_weight], dtype=float)
        workspace_margin = 0.05

        for pose_index, pose_target in enumerate(ik_request.pose_targets):
            base_pose = self._build_target_pose(pose_target, transform_matrix)
            logger.debug(
                "Pinocchio IK target built | variant=%s pose_index=%s translation=%s",
                variant,
                pose_index,
                base_pose.translation.tolist(),
            )

            if (
                approx_workspace_radius > 0.0
                and np.linalg.norm(base_pose.translation) > approx_workspace_radius + workspace_margin
            ):
                logger.warning(
                    "Requested pose is beyond approximate workspace radius | radius=%.3f translation=%s",
                    approx_workspace_radius,
                    base_pose.translation.tolist(),
                )
                continue

            for grip_offset in grip_offsets:
                seed_guesses = self._generate_initial_seeds(
                    model,
                    initial_q,
                    variant,
                    pose_index,
                    grip_offset,
                    use_joint,
                    gripper_q_index,
                    limit_arrays_valid,
                    lower_limits,
                    upper_limits,
                    finite_lower_mask,
                    finite_upper_mask,
                )
                logger.debug(
                    "Pinocchio IK seeds prepared | variant=%s pose_index=%s grip_offset=%.3f seeds=%s",
                    variant,
                    pose_index,
                    grip_offset,
                    len(seed_guesses),
                )

                for seed_index, seed_q in enumerate(seed_guesses):
                    q_candidate = seed_q.copy()
                    data = model.createData()
                    current_error = float('inf')
                    iterations = 0
                    adaptive_lambda = base_lambda
                    previous_error: float | None = None

                    for iteration in range(ik_request.max_iterations):
                        pin.forwardKinematics(model, data, q_candidate)
                        if supports_joint_jacobians:
                            pin.computeJointJacobians(model, data)
                        pin.updateFramePlacements(model, data)

                        current_pose = data.oMf[frame_id]

                        target_pose = base_pose
                        if not use_joint:
                            offset_transform = pin.SE3(np.eye(3), np.array([0.0, 0.0, grip_offset]))
                            target_pose = base_pose * offset_transform

                        raw_error_vec = pin.log(current_pose.inverse() * target_pose).vector
                        current_error = float(np.linalg.norm(raw_error_vec))
                        weighted_error_vec = raw_error_vec * error_weights
                        iterations = iteration + 1

                        if current_error <= ik_request.tolerance:
                            previous_error = current_error
                            break

                        increase_lambda = (
                            previous_error is not None
                            and current_error > previous_error * 1.05
                        )
                        decrease_lambda = (
                            previous_error is not None
                            and current_error < previous_error
                        )

                        try:
                            J = pin.getFrameJacobian(
                                model,
                                data,
                                frame_id,
                                pin.ReferenceFrame.LOCAL,
                            )
                        except AttributeError:
                            J = pin.computeFrameJacobian(
                                model,
                                data,
                                q_candidate,
                                frame_id,
                                pin.ReferenceFrame.LOCAL,
                            )

                        weighted_J = J * error_weights[:, None]
                        JT = weighted_J.T
                        damping_matrix = (adaptive_lambda * adaptive_lambda) * identity_nv
                        normal_matrix = JT @ weighted_J + damping_matrix
                        rhs = JT @ (-weighted_error_vec)

                        try:
                            dq = np.linalg.solve(normal_matrix, rhs)
                        except np.linalg.LinAlgError:
                            dq, *_ = np.linalg.lstsq(weighted_J, -weighted_error_vec, rcond=None)

                        try:
                            q_candidate = pin.integrate(model, q_candidate, dq)
                        except Exception as exc:  # pragma: no cover
                            raise RobotServiceError("Integration failure during IK solving.") from exc

                        q_candidate = self._enforce_joint_limits(
                            q_candidate,
                            limit_arrays_valid,
                            lower_limits,
                            upper_limits,
                            finite_lower_mask,
                            finite_upper_mask,
                        )

                        if use_joint and gripper_q_index is not None:
                            q_candidate[int(gripper_q_index)] = grip_offset

                        if increase_lambda:
                            adaptive_lambda = min(adaptive_lambda * lambda_increase_factor, lambda_max)
                        elif decrease_lambda:
                            adaptive_lambda = max(adaptive_lambda * lambda_decrease_factor, lambda_min)

                        previous_error = current_error

                    if current_error > ik_request.tolerance:
                        logger.debug(
                            "IK candidate did not converge within tolerance | variant=%s pose_index=%s grip_offset=%.3f seed_index=%s final_error=%.6f iterations=%s",
                            variant,
                            pose_index,
                            grip_offset,
                            seed_index,
                            current_error,
                            iterations,
                        )

                    joint_delta = float(np.linalg.norm(q_candidate - initial_q))
                    result = IkCandidateResult(
                        pose_index=pose_index,
                        grip_offset=float(grip_offset),
                        error=current_error,
                        iterations=iterations,
                        joint_distance=joint_delta,
                        mode_used="prismatic_joint" if use_joint else "offset",
                        coordinate_mode_used=ik_request.coordinate_mode,
                        urdf_variant=variant,
                        joint_positions=q_candidate.tolist(),
                    )

                    pose_results.append(result)

                    if best_result is None or result.error < best_result.error:
                        best_result = result
                        logger.debug(
                            "Pinocchio IK best candidate updated | variant=%s pose_index=%s grip_offset=%.3f error=%.6f",
                            variant,
                            pose_index,
                            grip_offset,
                            result.error,
                        )

        if best_result is None:
            raise RobotServiceError("IK solver did not evaluate any candidates.")

        self.store.robot.set_last_joint_positions(variant, best_result.joint_positions)

        logger.info(
            "IK result | variant={} best_error={:.6f} iterations={} pose_index={} grip_offset={:.3f} joint_positions={}",
            variant,
            best_result.error,
            best_result.iterations,
            best_result.pose_index,
            best_result.grip_offset,
            best_result.joint_positions,
        )

        return RobotIkResponse(
            best=best_result,
            candidates=pose_results,
            mode=mode_effective,
            urdf_variant_used=variant,
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

    def _enforce_joint_limits(
        self,
        q: np.ndarray,
        limit_arrays_valid: bool,
        lower_limits: Optional[np.ndarray],
        upper_limits: Optional[np.ndarray],
        finite_lower_mask: Optional[np.ndarray],
        finite_upper_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        if not limit_arrays_valid:
            return q

        if finite_lower_mask is not None and finite_lower_mask.any():
            q = np.where(finite_lower_mask, np.maximum(q, lower_limits), q)
        if finite_upper_mask is not None and finite_upper_mask.any():
            q = np.where(finite_upper_mask, np.minimum(q, upper_limits), q)
        return q

    def _generate_initial_seeds(
        self,
        model: pin.Model,
        initial_q: np.ndarray,
        variant: str,
        pose_index: int,
        grip_offset: float,
        use_joint: bool,
        gripper_q_index: Optional[int],
        limit_arrays_valid: bool,
        lower_limits: Optional[np.ndarray],
        upper_limits: Optional[np.ndarray],
        finite_lower_mask: Optional[np.ndarray],
        finite_upper_mask: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        seeds: List[np.ndarray] = []

        def add_seed(seed_q: np.ndarray) -> None:
            candidate = seed_q.copy()
            if use_joint and gripper_q_index is not None:
                candidate[int(gripper_q_index)] = grip_offset
            candidate = self._enforce_joint_limits(
                candidate,
                limit_arrays_valid,
                lower_limits,
                upper_limits,
                finite_lower_mask,
                finite_upper_mask,
            )
            for existing in seeds:
                if np.allclose(existing, candidate, atol=1e-6):
                    return
            seeds.append(candidate)

        add_seed(np.array(initial_q, dtype=float))

        last_known = self.store.robot.get_last_joint_positions(variant)
        if last_known is not None:
            add_seed(np.array(last_known, dtype=float))

        add_seed(pin.neutral(model))

        rng_seed = hash((variant, pose_index, float(grip_offset))) & 0xFFFFFFFFFFFF
        rng = np.random.default_rng(rng_seed)
        base_pool = [seed.copy() for seed in seeds[:2]]  # perturb 첫 두 시드

        for base in base_pool:
            try:
                perturb = rng.normal(scale=0.05, size=model.nv)
                noisy = pin.integrate(model, base, perturb)
                add_seed(noisy)
            except Exception:
                continue

        if not seeds:
            add_seed(pin.neutral(model))

        return seeds


