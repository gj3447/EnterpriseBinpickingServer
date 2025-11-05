import sys
import types
from contextlib import asynccontextmanager

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings


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

    class _DummyModel:
        def __init__(self):
            self.nq = 2
            self.njoints = 3
            self.nbodies = 2
            self.frame_ids = {"tool": 0}
            self.names = ["universe", "joint0", "joint1"]
            self.lowerPositionLimit = np.array([-10.0, -10.0], dtype=float)
            self.upperPositionLimit = np.array([10.0, 10.0], dtype=float)
            self.velocityLimit = np.array([1.0, 1.0], dtype=float)

        def createData(self):
            return _DummyData(self)

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

                    def shortname():
                        return "JointModelP"

                    self.shortname = shortname
                    self.idx_q = idx_q

            return [None, _Joint(0), _Joint(1)]

    class _DummyData:
        def __init__(self, model: _DummyModel):
            dummy_se3 = _DummySE3
            self.oMf = [dummy_se3(np.eye(3), np.zeros(3)) for _ in range(len(model.frame_ids))]

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

    def _dummy_build_model_from_urdf(path: str):
        return _DummyModel()

    dummy.SE3 = _DummySE3
    dummy.Quaternion = _DummyQuaternion
    dummy.ReferenceFrame = _DummyReferenceFrame
    dummy.neutral = _dummy_neutral
    dummy.forwardKinematics = _dummy_forward_kinematics
    dummy.updateFramePlacements = _dummy_update_frame_placements
    dummy.computeFrameJacobian = _dummy_compute_jacobian
    dummy.integrate = _dummy_integrate
    dummy.log = _dummy_log
    dummy.buildModelFromUrdf = _dummy_build_model_from_urdf
    dummy.Model = _DummyModel

    sys.modules["pinocchio"] = dummy
    return dummy


_install_dummy_pinocchio()

import app.dependencies as deps
from app.main import app
from app.schemas.robot import RobotIkResponse, IkCandidateResult


def _prepare_robot_store(has_gripper_joint: bool) -> None:
    model = sys.modules["pinocchio"].buildModelFromUrdf("dummy.urdf")
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
        "library": "pinocchio",
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

    store = deps.get_store()
    store.robot.clear()
    variant = "prismatic" if has_gripper_joint else "fixed"
    store.robot.set_urdf_object(
        variant,
        robot_object,
        "dummy",
        "dummy.urdf",
        metadata=metadata,
        set_default=True,
    )


@asynccontextmanager
async def _dummy_lifespan(_app):
    yield


def _get_client():
    app.router.lifespan_context = _dummy_lifespan
    return TestClient(app)


class _CaptureRobotService:
    def __init__(self):
        self.last_request = None

    async def solve_ik(self, request):
        self.last_request = request
        dummy_result = IkCandidateResult(
            pose_index=0,
            grip_offset=0.0,
            error=0.0,
            iterations=1,
            mode_used="offset",
            coordinate_mode_used=request.coordinate_mode,
            urdf_variant=request.urdf_variant or "fixed",
            joint_positions=[0.0],
        )
        return RobotIkResponse(
            best=dummy_result,
            candidates=[dummy_result],
            mode="fixed",
            urdf_variant_used=request.urdf_variant or "fixed",
            has_gripper_joint=False,
            gripper_joint_name=None,
        )


def test_robot_status_endpoint():
    _prepare_robot_store(has_gripper_joint=True)

    with _get_client() as client:
        response = client.get("/api/robot/status")
        assert response.status_code == 200
        body = response.json()
        assert "prismatic" in body["robot_status"]["loaded_variants"]
        assert body["robot_status"]["default_variant"] == "prismatic"


def test_robot_ik_endpoint_with_custom_axes():
    _prepare_robot_store(has_gripper_joint=False)

    payload = {
        "target_frame": "tool",
        "pose_targets": [
            {
                "translation": [0.0, 0.4, 0.0],
                "rotation_quaternion": [0.0, 0.0, 0.0, 1.0],
            }
        ],
        "grip_offsets": [0.0],
        "mode": "fixed",
        "coordinate_mode": "custom",
        "custom_axes": {"up": "y", "forward": "x"},
        "max_iterations": 30,
        "tolerance": 1e-6,
        "urdf_variant": "fixed",
    }

    with _get_client() as client:
        response = client.post("/api/robot/ik", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "fixed"
        assert body["urdf_variant_used"] == "fixed"
        assert body["best"]["coordinate_mode_used"] == "custom"
        assert body["best"]["urdf_variant"] == "fixed"
        assert pytest.approx(body["best"]["joint_positions"][0], abs=1e-5) == 0.4


def test_robot_ik_endpoint_prismatic_mode():
    _prepare_robot_store(has_gripper_joint=True)

    payload = {
        "target_frame": "tool",
        "pose_targets": [
            {
                "translation": [0.0, 0.0, 0.3],
                "rotation_quaternion": [0.0, 0.0, 0.0, 1.0],
            }
        ],
        "grip_offsets": [0.12],
        "mode": "auto",
        "coordinate_mode": "base",
        "max_iterations": 30,
        "tolerance": 1e-6,
        "urdf_variant": "prismatic",
    }

    with _get_client() as client:
        response = client.post("/api/robot/ik", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "prismatic"
        assert body["urdf_variant_used"] == "prismatic"
        assert body["best"]["mode_used"] == "prismatic_joint"
        assert body["best"]["urdf_variant"] == "prismatic"
        assert pytest.approx(body["best"]["joint_positions"][1], abs=1e-6) == 0.12
        assert pytest.approx(body["best"]["joint_positions"][0], abs=1e-5) == 0.3


def test_robot_ik_downward_endpoint_sets_fixed_orientation():
    capture_service = _CaptureRobotService()
    app.dependency_overrides[deps.get_robot_service] = lambda: capture_service

    payload = {
        "target_frame": "tool",
        "translations": [[0.1, -0.2, 0.3], [0.0, 0.0, 0.4]],
        "mode": "fixed",
        "coordinate_mode": "base",
    }

    try:
        with _get_client() as client:
            response = client.post("/api/robot/ik/downward", json=payload)
            assert response.status_code == 200
            body = response.json()
            assert body["mode"] == "fixed"
            assert body["urdf_variant_used"] == "fixed"
    finally:
        app.dependency_overrides.pop(deps.get_robot_service, None)

    assert capture_service.last_request is not None
    pose_targets = capture_service.last_request.pose_targets
    assert len(pose_targets) == 2
    assert pose_targets[0].translation == [0.1, -0.2, 0.3]
    assert pose_targets[1].translation == [0.0, 0.0, 0.4]
    expected_quat = [1.0, 0.0, 0.0, 0.0]
    assert pose_targets[0].rotation_quaternion == expected_quat
    assert pose_targets[1].rotation_quaternion == expected_quat


def test_robot_ik_downward_ikpy_endpoint_custom_axes_hover():
    capture_service = _CaptureRobotService()
    app.dependency_overrides[deps.get_robot_service_ikpy] = lambda: capture_service

    payload = {
        "target_frame": "tool",
        "translations": [[0.2, -0.1, 0.35]],
        "hover_height": 0.04,
        "grip_offsets": [0.02],
        "mode": "auto",
        "coordinate_mode": "custom",
        "custom_axes": {"up": "y", "forward": "x"},
        "urdf_variant": "fixed",
    }

    try:
        with _get_client() as client:
            response = client.post("/api/robot/ik/ikpy/downward", json=payload)
            assert response.status_code == 200
            body = response.json()
            assert body["mode"] == "fixed"
            assert body["urdf_variant_used"] == "fixed"
            assert body["best"]["coordinate_mode_used"] == "custom"
    finally:
        app.dependency_overrides.pop(deps.get_robot_service_ikpy, None)

    assert capture_service.last_request is not None
    request = capture_service.last_request
    assert request.coordinate_mode == "custom"
    assert request.custom_axes is not None
    assert request.custom_axes.up == "y"
    assert request.mode == "auto"
    assert request.urdf_variant == "fixed"
    assert request.target_frame == settings.IKPY_END_EFFECTOR_FRAME

    pose_targets = request.pose_targets
    assert len(pose_targets) == 1
    expected_translation = [0.2, -0.06, 0.35]
    assert pose_targets[0].translation == pytest.approx(expected_translation, abs=1e-8)
    assert pose_targets[0].rotation_quaternion == [1.0, 0.0, 0.0, 0.0]

