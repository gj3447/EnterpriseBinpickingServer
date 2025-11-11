import numpy as np

from app.services.robot_service_ikpy import RobotServiceIkpy


def _make_service() -> RobotServiceIkpy:
    return RobotServiceIkpy(store=None, fixed_urdf_path=None, prismatic_urdf_path=None)


def test_apply_grip_offset_identity_rotation():
    service = _make_service()
    target = np.eye(4)
    target[:3, 3] = np.array([0.0, 0.0, 0.5])

    adjusted = service._apply_grip_offset(target, 0.1)

    assert np.allclose(adjusted[:3, 3], np.array([0.0, 0.0, 0.4]))
    # Rotation should remain unchanged
    assert np.allclose(adjusted[:3, :3], target[:3, :3])


def test_apply_grip_offset_respects_local_orientation():
    service = _make_service()

    theta = np.pi / 2  # 90 degrees about Y-axis
    rotation = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ],
        dtype=float,
    )

    target = np.eye(4)
    target[:3, :3] = rotation
    target[:3, 3] = np.array([0.5, 0.0, 0.0])

    adjusted = service._apply_grip_offset(target, 0.2)

    expected_translation = target[:3, 3] - rotation[:, 2] * 0.2
    assert np.allclose(adjusted[:3, 3], expected_translation)
    assert np.allclose(adjusted[:3, :3], rotation)

