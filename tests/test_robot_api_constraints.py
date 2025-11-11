from tests.test_robot_service_ik import _install_dummy_pinocchio

_install_dummy_pinocchio()

from app.api.v1.endpoints.robot import _validate_pose_targets_height
from app.schemas.robot import PoseTarget, RobotIkRequest


def test_validate_pose_targets_height_allows_ground_plane():
    request = RobotIkRequest(
        target_frame="tool",
        pose_targets=[PoseTarget(translation=[0.0, 0.0, 0.0], rotation_quaternion=[0.0, 0.0, 0.0, 1.0])],
        coordinate_mode="base",
    )

    # Should not modify when z == IK_MIN_Z (default 0.0)
    _validate_pose_targets_height(request)
    assert request.pose_targets[0].translation[2] == 0.0


def test_validate_pose_targets_height_rejects_below_ground():
    request = RobotIkRequest(
        target_frame="tool",
        pose_targets=[PoseTarget(translation=[0.0, 0.0, -0.01], rotation_quaternion=[0.0, 0.0, 0.0, 1.0])],
        coordinate_mode="base",
    )

    _validate_pose_targets_height(request)
    assert request.pose_targets[0].translation[2] == 0.0

