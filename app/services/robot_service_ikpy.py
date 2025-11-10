import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ikpy.chain import Chain
from ikpy.urdf.URDF import get_chain_from_joints

from app.core.config import settings
from app.core.logging import logger
from app.schemas.robot import IkCandidateResult, RobotIkRequest, RobotIkResponse
from app.services.robot_service_base import RobotServiceError
from app.stores.application_store import ApplicationStore


class RobotServiceIkpy:
    """ikpy 기반으로 URDF를 로드하고 역기구학을 계산하는 서비스."""

    def __init__(
        self,
        store: ApplicationStore,
        fixed_urdf_path: Optional[str] = None,
        prismatic_urdf_path: Optional[str] = None,
    ) -> None:
        self.store = store
        self.fixed_urdf_path = fixed_urdf_path
        self.prismatic_urdf_path = prismatic_urdf_path
        self._is_running = False
        logger.info(
            "RobotServiceIkpy initialized with fixed=%s, prismatic=%s",
            fixed_urdf_path,
            prismatic_urdf_path,
        )

    async def start(self) -> None:
        if self._is_running:
            logger.warning("RobotServiceIkpy is already running")
            return

        logger.info("Starting RobotServiceIkpy...")
        self._is_running = True

        any_loaded = False

        if self.fixed_urdf_path:
            logger.info("Attempting to load fixed URDF with ikpy: %s", self.fixed_urdf_path)
            success_fixed = await self.load_urdf_variant("fixed", self.fixed_urdf_path, set_default=True)
            any_loaded = any_loaded or success_fixed
        else:
            logger.warning("No fixed URDF path provided for ikpy. Skipping fixed variant load.")

        if self.prismatic_urdf_path:
            logger.info("Attempting to load prismatic URDF with ikpy: %s", self.prismatic_urdf_path)
            success_prismatic = await self.load_urdf_variant("prismatic", self.prismatic_urdf_path)
            any_loaded = any_loaded or success_prismatic
        else:
            logger.info("No prismatic URDF path provided for ikpy. Skipping prismatic variant load.")

        if any_loaded:
            self._configure_default_variant()
            logger.info(
                "RobotServiceIkpy started with variants=%s (default=%s)",
                list(self.store.robot.list_variants().keys()),
                self.store.robot.get_default_variant(),
            )
        else:
            logger.error("RobotServiceIkpy started but no URDF variants could be loaded.")

    async def stop(self) -> None:
        if not self._is_running:
            return

        self._is_running = False
        logger.info("RobotServiceIkpy stopped.")

    async def load_urdf_variant(self, variant: str, file_path: str, set_default: bool = False) -> bool:
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error("URDF file not found for variant %s: %s", variant, file_path)
                return False

            effective_path = Path(settings.ROBOT_URDF_PATH_FIXED)
            if not effective_path.exists():
                logger.warning(
                    "Configured fixed URDF path not found for ikpy override. Falling back to provided path. path=%s",
                    settings.ROBOT_URDF_PATH_FIXED,
                )
                effective_path = path

            logger.info(
                "Loading %s URDF with ikpy: %s (effective: %s)",
                variant,
                file_path,
                effective_path,
            )

            try:
                robot_name, robot_object, metadata = await asyncio.to_thread(
                    self._parse_urdf_with_ikpy,
                    effective_path,
                    variant,
                    original_path=path,
                )
            except ImportError as exc:
                logger.error("❌ ikpy not available while loading %s: %s", variant, exc)
                logger.error("Please install ikpy in your environment")
                return False
            except Exception as exc:
                logger.error("❌ Error parsing %s URDF with ikpy: %s", variant, exc, exc_info=True)
                return False

            self.store.robot.set_urdf_object(
                variant,
                robot_object,
                robot_name,
                str(effective_path),
                metadata=metadata,
                set_default=set_default,
            )
            return True

        except Exception as exc:
            logger.error("❌ Unexpected error loading %s URDF with ikpy: %s", variant, exc, exc_info=True)
            return False

    def _configure_default_variant(self) -> None:
        variants = self.store.robot.list_variants()
        if not variants:
            logger.error("No URDF variants available to configure default for ikpy service.")
            return

        mode = settings.ROBOT_URDF_MODE
        preference: List[str]
        if mode == "fixed":
            preference = ["fixed", "prismatic"]
        elif mode == "prismatic":
            preference = ["prismatic", "fixed"]
        else:
            preference = ["prismatic", "fixed"]

        for candidate in preference:
            if candidate in variants:
                self.store.robot.set_default_variant(candidate)
                return

        self.store.robot.set_default_variant(next(iter(variants.keys())))

    def _parse_urdf_with_ikpy(
        self,
        path: Path,
        variant: str,
        original_path: Optional[Path] = None,
    ) -> Tuple[str, Dict[str, object], Dict[str, object]]:
        logger.info("ikpy library found - starting URDF parsing for %s...", variant)

        active_joint_names: List[str] = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]

        try:
            base_elements = get_chain_from_joints(str(path), active_joint_names)
        except ValueError as exc:
            logger.warning("Failed to derive base elements for ikpy chain: %s. Falling back to default parsing.", exc)
            base_elements = None

        if base_elements is None:
            chain = Chain.from_urdf_file(str(path))
        else:
            active_mask = [False] + [True] * len(active_joint_names)
            chain = Chain.from_urdf_file(
                str(path),
                base_elements=base_elements,
                active_links_mask=active_mask,
            )

        robot_name = chain.name or path.stem

        active_indices = [idx for idx, active in enumerate(chain.active_links_mask) if active]
        joint_names = [chain.links[idx].name for idx in active_indices]

        lower_limits: List[float] = []
        upper_limits: List[float] = []
        for idx in active_indices:
            bounds = getattr(chain.links[idx], "bounds", None)
            if bounds is None:
                lower_limits.append(float("-inf"))
                upper_limits.append(float("inf"))
            else:
                lower_limits.append(bounds[0] if bounds[0] is not None else float("-inf"))
                upper_limits.append(bounds[1] if bounds[1] is not None else float("inf"))

        translation_vectors: List[np.ndarray] = []
        for link in chain.links:
            vector = getattr(link, "translation_vector", None)
            if vector is not None:
                translation_vectors.append(np.array(vector, dtype=float))

        approx_workspace_radius = float(np.sum(np.linalg.norm(vec) for vec in translation_vectors))

        neutral_configuration = np.zeros(len(chain.links), dtype=float)

        metadata: Dict[str, object] = {
            "variant": variant,
            "configured_urdf_mode": settings.ROBOT_URDF_MODE,
            "configured_urdf_variant": variant,
            "configured_urdf_path": str(path),
            "approx_workspace_radius": approx_workspace_radius,
            "active_link_indices": active_indices,
            "has_gripper_joint": False,
            "ikpy_override_used": original_path is not None and original_path != path,
            "original_urdf_path": str(original_path) if original_path else str(path),
            "ikpy_end_effector_frame": settings.IKPY_END_EFFECTOR_FRAME,
            "ikpy_gripper_length": settings.IKPY_GRIPPER_LENGTH,
        }

        robot_object = {
            "chain": chain,
            "library": "ikpy",
            "urdf_path": str(path),
            "robot_name": robot_name,
            "dof": len(active_indices),
            "joint_names": joint_names,
            "joint_limits": {
                "lower": lower_limits,
                "upper": upper_limits,
            },
            "approx_workspace_radius": approx_workspace_radius,
            "neutral_configuration": neutral_configuration.tolist(),
            "active_link_indices": active_indices,
            "end_effector_frame": settings.IKPY_END_EFFECTOR_FRAME,
        }

        return robot_name, robot_object, metadata

    def get_robot_object(self, variant: Optional[str] = None) -> Optional[Dict[str, object]]:
        return self.store.robot.get_urdf_object(variant)

    def get_status(self) -> Dict[str, object]:
        return {
            "is_running": self._is_running,
            "robot_status": self.store.robot.get_status(),
        }

    def get_metadata(self, variant: Optional[str] = None) -> Dict[str, object]:
        return self.store.robot.get_metadata(variant)

    async def solve_ik(self, ik_request: RobotIkRequest) -> RobotIkResponse:
        return await asyncio.to_thread(self._solve_ik_sync, ik_request)

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
            "ikpy IK request | variant={} coordinate_mode={} poses={} grip_offsets={} initial_q={}",
            variant,
            ik_request.coordinate_mode,
            pose_summaries,
            ik_request.grip_offsets,
            "yes" if ik_request.initial_joint_positions is not None else "no",
        )

        robot_object = self.store.robot.get_urdf_object(variant)
        if not robot_object or robot_object.get("library") != "ikpy":
            raise RobotServiceError(f"ikpy robot model is not loaded for variant '{variant}'.")

        chain: Chain = robot_object["chain"]
        active_indices: List[int] = robot_object["active_link_indices"]
        neutral_config = np.array(robot_object["neutral_configuration"], dtype=float)
        joint_limits = robot_object.get("joint_limits", {})

        lower_limits = np.array(joint_limits.get("lower", []), dtype=float)
        upper_limits = np.array(joint_limits.get("upper", []), dtype=float)
        limit_arrays_valid = lower_limits.size == len(active_indices) == upper_limits.size

        approx_workspace_radius = float(robot_object.get("approx_workspace_radius") or 0.0)

        if not ik_request.pose_targets:
            raise RobotServiceError("pose_targets must contain at least one target pose.")

        transform_matrix = self._build_coordinate_transform(ik_request)
        logger.debug(
            "ikpy IK transform matrix computed | variant={} coordinate_mode={} transform={}",
            variant,
            ik_request.coordinate_mode,
            transform_matrix.tolist() if transform_matrix is not None else None,
        )

        grip_offsets = ik_request.grip_offsets or [0.0]
        mode_effective = "fixed"

        initial_full = neutral_config.copy()
        if ik_request.initial_joint_positions is not None:
            initial_full = self._coerce_initial_positions(
                np.array(ik_request.initial_joint_positions, dtype=float),
                chain,
                active_indices,
                neutral_config,
            )
        else:
            last_known = self.store.robot.get_last_joint_positions(variant)
            if last_known is not None:
                initial_full = self._expand_active_to_full(
                    np.array(last_known, dtype=float),
                    active_indices,
                    neutral_config,
                )

        pose_results: List[IkCandidateResult] = []
        best_result: Optional[IkCandidateResult] = None

        workspace_margin = 0.05

        for pose_index, pose_target in enumerate(ik_request.pose_targets):
            logger.info(
                "ikpy IK evaluating pose | variant={} pose_index={} raw_translation={}",
                variant,
                pose_index,
                pose_target.translation,
            )
            target_matrix = self._build_target_matrix(pose_target, transform_matrix)
            logger.debug(
                "ikpy IK target built | variant={} pose_index={} translation={}",
                variant,
                pose_index,
                target_matrix[:3, 3].tolist(),
            )

            target_distance = float(np.linalg.norm(target_matrix[:3, 3]))
            outside_workspace = (
                approx_workspace_radius > 0.0
                and target_distance > approx_workspace_radius + workspace_margin
            )
            if outside_workspace:
                logger.warning(
                    "Requested pose is beyond approximate workspace radius | radius={:.3f} + margin={:.3f} distance={:.3f} translation={}",
                    approx_workspace_radius,
                    workspace_margin,
                    target_distance,
                    target_matrix[:3, 3].tolist(),
                )

            for grip_offset in grip_offsets:
                offset_matrix = np.eye(4)
                offset_matrix[:3, 3] = np.array([0.0, 0.0, grip_offset], dtype=float)
                adjusted_target = target_matrix @ offset_matrix

                seeds = self._generate_initial_seeds(
                    chain,
                    initial_full,
                    variant,
                    pose_index,
                    grip_offset,
                    active_indices,
                    neutral_config,
                )
                logger.info(
                    "ikpy IK seeds prepared | variant={} pose_index={} grip_offset={:.3f} seeds={}",
                    variant,
                    pose_index,
                    grip_offset,
                    len(seeds),
                )

                pose_attempted = False
                pose_candidates_found = False

                if not seeds:
                    logger.warning(
                        "ikpy IK generated no seeds | variant={} pose_index={} grip_offset={:.3f}",
                        variant,
                        pose_index,
                        grip_offset,
                    )
                    continue

                for seed_index, seed in enumerate(seeds):
                    pose_attempted = True
                    try:
                        solution = chain.inverse_kinematics(
                            target_position=adjusted_target[:3, 3],
                            target_orientation=adjusted_target[:3, :3],
                            orientation_mode="all",
                            max_iter=ik_request.max_iterations,
                            initial_position=seed,
                        )
                    except Exception as exc:
                        logger.info(
                            "ikpy inverse_kinematics failed | variant={} pose_index={} grip_offset={:.3f} seed_index={} error={!r}",
                            variant,
                            pose_index,
                            grip_offset,
                            seed_index,
                            exc,
                        )
                        continue

                    if np.any(np.isnan(solution)):
                        logger.info(
                            "ikpy IK solution contained NaN | variant={} pose_index={} grip_offset={:.3f} seed_index={}",
                            variant,
                            pose_index,
                            grip_offset,
                            seed_index,
                        )
                        continue

                    solution = self._enforce_joint_limits(
                        solution,
                        chain,
                        active_indices,
                        limit_arrays_valid,
                        lower_limits,
                        upper_limits,
                    )

                    error = self._compute_pose_error(chain, solution, adjusted_target)

                    active_solution = self._extract_active_positions(solution, active_indices)

                    result = IkCandidateResult(
                        pose_index=pose_index,
                        grip_offset=float(grip_offset),
                        error=float(error),
                        iterations=0,
                        mode_used="offset",
                        coordinate_mode_used=ik_request.coordinate_mode,
                        urdf_variant=variant,
                        joint_positions=active_solution.tolist(),
                    )

                    pose_candidates_found = True
                    pose_results.append(result)

                    if best_result is None or result.error < best_result.error:
                        best_result = result
                        logger.debug(
                            "ikpy IK best candidate updated | variant={} pose_index={} grip_offset={:.3f} error={:.6f} joints={}",
                            variant,
                            result.pose_index,
                            result.grip_offset,
                            result.error,
                            result.joint_positions,
                        )

                if not pose_attempted:
                    logger.warning(
                        "ikpy IK seeds failed before evaluation | variant={} pose_index={} grip_offset={:.3f} seeds={}",
                        variant,
                        pose_index,
                        grip_offset,
                        len(seeds),
                    )
                elif not pose_candidates_found:
                    logger.warning(
                        "ikpy IK evaluated seeds but found no valid solution | variant={} pose_index={} grip_offset={:.3f}",
                        variant,
                        pose_index,
                        grip_offset,
                    )

        if best_result is None:
            logger.error(
                "ikpy IK solver exhausted all candidates without success | "
                f"variant={variant} pose_count={len(ik_request.pose_targets)} "
                f"grip_offsets={grip_offsets} workspace_radius={approx_workspace_radius:.3f}"
            )
            raise RobotServiceError("ikpy IK solver did not evaluate any candidates.")

        self.store.robot.set_last_joint_positions(variant, best_result.joint_positions)

        logger.info(
            "ikpy IK result | variant={} best_error={:.6f} pose_index={} grip_offset={:.3f} joints={}",
            variant,
            best_result.error,
            best_result.pose_index,
            best_result.grip_offset,
            best_result.joint_positions,
        )

        return RobotIkResponse(
            best=best_result,
            candidates=pose_results,
            mode=mode_effective,
            urdf_variant_used=variant,
            has_gripper_joint=False,
            gripper_joint_name=None,
        )

    # --- Helper methods ---

    def _coerce_initial_positions(
        self,
        initial: np.ndarray,
        chain: Chain,
        active_indices: List[int],
        neutral: np.ndarray,
    ) -> np.ndarray:
        if initial.size == len(chain.links):
            return initial
        if initial.size == len(active_indices):
            return self._expand_active_to_full(initial, active_indices, neutral)
        raise RobotServiceError(
            f"initial_joint_positions length {initial.size} does not match ikpy chain lengths ({len(chain.links)} or {len(active_indices)})."
        )

    def _expand_active_to_full(
        self,
        active_values: np.ndarray,
        active_indices: List[int],
        base: np.ndarray,
    ) -> np.ndarray:
        full = base.copy()
        for value, idx in zip(active_values, active_indices):
            full[idx] = value
        return full

    def _extract_active_positions(
        self,
        full_solution: np.ndarray,
        active_indices: List[int],
    ) -> np.ndarray:
        return np.array([full_solution[idx] for idx in active_indices], dtype=float)

    def _generate_initial_seeds(
        self,
        chain: Chain,
        initial_full: np.ndarray,
        variant: str,
        pose_index: int,
        grip_offset: float,
        active_indices: List[int],
        neutral: np.ndarray,
    ) -> List[np.ndarray]:
        seeds: List[np.ndarray] = []

        def add_seed(candidate: np.ndarray) -> None:
            for existing in seeds:
                if np.allclose(existing, candidate, atol=1e-6):
                    return
            seeds.append(candidate)

        add_seed(initial_full.copy())

        last_active = self.store.robot.get_last_joint_positions(variant)
        if last_active is not None:
            expanded = self._expand_active_to_full(
                np.array(last_active, dtype=float),
                active_indices,
                neutral,
            )
            add_seed(expanded)

        add_seed(neutral.copy())

        rng_seed = hash((variant, pose_index, float(grip_offset))) & 0xFFFFFFFFFFFF
        rng = np.random.default_rng(rng_seed)

        base_pool = seeds[:2]
        for base in base_pool:
            try:
                perturb = rng.normal(scale=0.05, size=len(chain.links))
                noisy = base + perturb
                add_seed(noisy)
            except Exception:
                continue

        return seeds

    def _enforce_joint_limits(
        self,
        solution: np.ndarray,
        chain: Chain,
        active_indices: List[int],
        limit_arrays_valid: bool,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
    ) -> np.ndarray:
        if not limit_arrays_valid:
            return solution

        corrected = solution.copy()
        for local_idx, link_idx in enumerate(active_indices):
            lower = lower_limits[local_idx]
            upper = upper_limits[local_idx]
            corrected[link_idx] = float(np.clip(corrected[link_idx], lower, upper))
        return corrected

    def _compute_pose_error(
        self,
        chain: Chain,
        configuration: np.ndarray,
        target_matrix: np.ndarray,
    ) -> float:
        current_matrix = chain.forward_kinematics(configuration)
        translation_error = np.linalg.norm(current_matrix[:3, 3] - target_matrix[:3, 3])
        rotation_delta = target_matrix[:3, :3].T @ current_matrix[:3, :3]
        trace_value = np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
        orientation_error = float(np.arccos(trace_value))
        return float(np.sqrt(translation_error ** 2 + orientation_error ** 2))

    def _build_target_matrix(
        self,
        pose_target,
        axes_matrix: Optional[np.ndarray],
    ) -> np.ndarray:
        translation = np.array(pose_target.translation, dtype=float)
        if translation.shape != (3,):
            raise RobotServiceError("translation must be length 3.")

        rotation_matrix = self._quat_to_matrix(pose_target.rotation_quaternion)

        if axes_matrix is not None:
            translation = axes_matrix @ translation
            rotation_matrix = axes_matrix @ rotation_matrix

        target_matrix = np.eye(4)
        target_matrix[:3, :3] = rotation_matrix
        target_matrix[:3, 3] = translation
        return target_matrix

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
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=float,
        )


__all__ = ["RobotServiceIkpy"]

