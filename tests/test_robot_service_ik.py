import pytest


pytest.skip("Pinocchio backend disabled; legacy pinocchio IK tests skipped.", allow_module_level=True)


import sys
import types

import numpy as np


def _install_dummy_pinocchio():
    if "pinocchio" in sys.modules:
        return sys.modules["pinocchio"]

    dummy = types.ModuleType("pinocchio")

    class _DummyMotion:
        def __init__(self, vector: np.ndarray):
            self.vector = vector

    class _DummySE3:
        def __init__(self, rotation, translation):
            self.rotation = np.array(rotation, dtype=float)
            self.translation = np.array(translation, dtype=float)

        def __mul__(self, other: "_DummySE3") -> "_DummySE3":
            rotation = self.rotation @ other.rotation
            translation = self.rotation @ other.translation + self.translation
            return _DummySE3(rotation, translation)

        def inverse(self) -> "_DummySE3":
            rotation = self.rotation.T
            translation = -rotation @ self.translation
            return _DummySE3(rotation, translation)

    class _DummyQuaternion:
        def __init__(self, w: float, x: float, y: float, z: float):
            self._array = np.array([w, x, y, z], dtype=float)

        def normalize(self):
            norm = np.linalg.norm(self._array)
            if norm == 0.0:
                return
            self._array /= norm

        def toRotationMatrix(self):
            return np.eye(3)

    class _DummyReferenceFrame:
        LOCAL = "LOCAL"

    def _dummy_neutral(model):
        return np.zeros(model.nq)

    def _dummy_forward_kinematics(model, data, q):
        translation = np.array([0.0, 0.0, q[0]], dtype=float)
        data.oMf[model.frame_ids["tool"]] = _DummySE3(np.eye(3), translation)

    def _dummy_update_frame_placements(model, data):
        return None

    def _dummy_compute_jacobian(model, data, q, frame_id, reference_frame):
        jacobian = np.zeros((6, model.nq), dtype=float)
        jacobian[2, 0] = -1.0
        return jacobian

    def _dummy_integrate(model, q, dq):
        return q + dq

    def _dummy_log(se3: _DummySE3):
        vector = np.zeros(6, dtype=float)
        vector[:3] = se3.translation
        return _DummyMotion(vector)

    dummy.SE3 = _DummySE3
    dummy.Quaternion = _DummyQuaternion
    dummy.ReferenceFrame = _DummyReferenceFrame
    dummy.Model = type("Model", (), {})
    dummy.neutral = _dummy_neutral
    dummy.forwardKinematics = _dummy_forward_kinematics
    dummy.updateFramePlacements = _dummy_update_frame_placements
    dummy.computeFrameJacobian = _dummy_compute_jacobian
    dummy.integrate = _dummy_integrate
    dummy.log = _dummy_log

    sys.modules["pinocchio"] = dummy
    return dummy


_install_dummy_pinocchio()

from app.schemas.robot import RobotIkRequest, PoseTarget, CoordinateAxes
from app.services.robot_service import RobotService
from app.stores.handlers.robot_handler import RobotHandler


class DummyModel:
    def __init__(self):
        self.nq = 2
        self.nv = 2
        self.frame_ids = {"tool": 0}
        self.lowerPositionLimit = np.array([-10.0, -10.0], dtype=float)
        self.upperPositionLimit = np.array([10.0, 10.0], dtype=float)
        self.velocityLimit = np.array([1.0, 1.0], dtype=float)
        self.njoints = 3
        self.nbodies = 2
        self.names = ["universe", "joint0", "joint1"]

    def createData(self):
        return DummyData(self)

    def getFrameId(self, name: str) -> int:
        return self.frame_ids[name]

    def getJointId(self, name: str) -> int:
        if name == "gripper_joint":
            return 2
        raise ValueError(name)

    @property
    def joints(self):
        class _Joint:
            def __init__(self, idx_q):
                self.nq = 1
                self.idx_q = idx_q

            def shortname(self):  # pragma: no cover
                return "PrismaticJoint"

        return [None, _Joint(0), _Joint(1)]


class DummyData:
    def __init__(self, model: DummyModel):
        dummy_se3 = sys.modules["pinocchio"].SE3
        self.oMf = [dummy_se3(np.eye(3), np.zeros(3)) for _ in range(len(model.frame_ids))]


_dummy_pinocchio = sys.modules["pinocchio"]


def _dummy_build_model_from_urdf(_path: str):  # pragma: no cover
    return DummyModel()


if not hasattr(_dummy_pinocchio, "buildModelFromUrdf"):
    _dummy_pinocchio.buildModelFromUrdf = _dummy_build_model_from_urdf


class DummyStore:
    def __init__(self):
        self.robot = RobotHandler()


def _create_robot_service(has_gripper_joint: bool) -> RobotService:
    model = DummyModel()
    data = model.createData()
    metadata = {
        "has_gripper_joint": has_gripper_joint,
        "gripper_joint_name": "gripper_joint" if has_gripper_joint else None,
        "gripper_joint_q_index": 1 if has_gripper_joint else None,
        "configured_urdf_variant": "prismatic" if has_gripper_joint else "fixed",
    }
    robot_object = {
        "model": model,
        "data": data,
        "library": "ikpy",
        "robot_name": "dummy",
        "urdf_path": "dummy.urdf",
        "dof": model.nq,
        "joint_names": ["joint0", "joint1"],
        "joint_limits": {
            "lower": model.lowerPositionLimit.tolist(),
            "upper": model.upperPositionLimit.tolist(),
            "velocity": model.velocityLimit.tolist(),
        },
        "has_gripper_joint": has_gripper_joint,
        "gripper_joint_name": metadata["gripper_joint_name"],
        "gripper_joint_q_index": metadata["gripper_joint_q_index"],
    }

    store = DummyStore()
    variant_name = "prismatic" if has_gripper_joint else "fixed"
    store.robot.set_urdf_object(
        variant_name,
        robot_object,
        "dummy",
        "dummy.urdf",
        metadata=metadata,
        set_default=True,
    )

    return RobotService(store, fixed_urdf_path=None, prismatic_urdf_path=None)


def test_solve_ik_in_base_coordinate_without_prismatic_joint():
    service = _create_robot_service(has_gripper_joint=False)

    request = RobotIkRequest(
        target_frame="tool",
        pose_targets=[PoseTarget(translation=[0.0, 0.0, 1.0], rotation_quaternion=[0.0, 0.0, 0.0, 1.0])],
        grip_offsets=[0.0, 0.2],
        mode="auto",
        coordinate_mode="base",
        max_iterations=40,
        tolerance=1e-6,
    )

    response = service._solve_ik_sync(request)

    assert response.mode == "fixed"
    assert response.urdf_variant_used == "fixed"
    assert response.best.mode_used == "offset"
    assert response.best.coordinate_mode_used == "base"
    assert response.best.urdf_variant == "fixed"
    assert response.best.error <= 1e-5
    assert response.best.joint_distance >= 0.0


def test_solve_ik_with_custom_coordinate_axes():
    service = _create_robot_service(has_gripper_joint=False)

    request = RobotIkRequest(
        target_frame="tool",
        pose_targets=[PoseTarget(translation=[0.0, 0.5, 0.0], rotation_quaternion=[0.0, 0.0, 0.0, 1.0])],
        grip_offsets=[0.0],
        coordinate_mode="custom",
        custom_axes=CoordinateAxes(up="y", forward="x"),
        mode="fixed",
        max_iterations=15,
        tolerance=1e-6,
    )

    response = service._solve_ik_sync(request)

    assert response.best.coordinate_mode_used == "custom"
    assert response.best.urdf_variant == "fixed"
    assert pytest.approx(response.best.joint_positions[0], abs=1e-5) == 0.5
    assert response.best.joint_distance >= 0.0


def test_solve_ik_uses_prismatic_joint_when_available():
    service = _create_robot_service(has_gripper_joint=True)

    request = RobotIkRequest(
        target_frame="tool",
        pose_targets=[PoseTarget(translation=[0.0, 0.0, 1.0], rotation_quaternion=[0.0, 0.0, 0.0, 1.0])],
        grip_offsets=[0.15],
        mode="auto",
        coordinate_mode="base",
        max_iterations=40,
        tolerance=1e-6,
    )

    response = service._solve_ik_sync(request)

    assert response.mode == "prismatic"
    assert response.urdf_variant_used == "prismatic"
    assert response.best.mode_used == "prismatic_joint"
    assert response.best.urdf_variant == "prismatic"
    assert pytest.approx(response.best.joint_positions[1], abs=1e-6) == 0.15
    assert pytest.approx(response.best.joint_positions[0], abs=1e-5) == 1.0
    assert response.best.error <= 1e-5
    assert response.best.joint_distance >= 0.0

