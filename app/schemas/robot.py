from __future__ import annotations

from typing import List, Literal, Optional

import math
from pydantic import BaseModel, Field, field_validator


class CoordinateAxes(BaseModel):
    up: str = Field(..., description="위쪽을 가리키는 축 (예: 'z', '-y')")
    forward: str = Field(..., description="전방을 가리키는 축 (예: 'x', 'z')")


class PoseTarget(BaseModel):
    translation: List[float] = Field(..., min_length=3, max_length=3)
    rotation_quaternion: List[float] = Field(..., min_length=4, max_length=4)

    @field_validator("rotation_quaternion")
    @classmethod
    def _validate_quaternion(cls, value: List[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in value))
        if norm == 0:
            raise ValueError("rotation_quaternion must not be zero-length.")
        return value


class RobotIkRequest(BaseModel):
    target_frame: str = Field(..., description="IK를 계산할 말단 프레임 이름")
    pose_targets: List[PoseTarget] = Field(..., min_length=1)
    grip_offsets: Optional[List[float]] = Field(default_factory=lambda: [0.0])
    mode: Literal["auto", "fixed", "prismatic"] = Field("auto")
    coordinate_mode: Literal["base", "custom"] = Field("base")
    urdf_variant: Optional[Literal["fixed", "prismatic"]] = Field(
        default=None,
        description="IK 계산에 사용할 URDF variant (None이면 기본값 사용)",
    )
    custom_axes: Optional[CoordinateAxes] = None
    initial_joint_positions: Optional[List[float]] = Field(default=None)
    max_iterations: int = Field(200, ge=1, le=500)
    tolerance: float = Field(1e-4, gt=0.0)
    damping: float = Field(0.6, gt=0.0)

    @field_validator("grip_offsets")
    @classmethod
    def _validate_offsets(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is None:
            return None
        if len(value) == 0:
            raise ValueError("grip_offsets must contain at least one value when provided.")
        return value

    @field_validator("custom_axes")
    @classmethod
    def _require_axes_for_custom(cls, value: Optional[CoordinateAxes], info):
        coordinate_mode = info.data.get("coordinate_mode", "base")
        if coordinate_mode == "custom" and value is None:
            raise ValueError("custom_axes must be provided when coordinate_mode is 'custom'.")
        return value


class RobotIkDownwardRequest(BaseModel):
    target_frame: str = Field(..., description="IK를 계산할 말단 프레임 이름")
    translations: List[List[float]] = Field(..., min_length=1, description="위치만 지정된 목표 포인트 목록")
    hover_height: float = Field(0.0, ge=0.0, description="목표 지점 위에 유지할 높이 (미터 단위)")
    grip_offsets: Optional[List[float]] = Field(default_factory=lambda: [0.0])
    mode: Literal["auto", "fixed", "prismatic"] = Field("auto")
    coordinate_mode: Literal["base", "custom"] = Field("base")
    urdf_variant: Optional[Literal["fixed", "prismatic"]] = Field(
        default=None,
        description="IK 계산에 사용할 URDF variant (None이면 기본값 사용)",
    )
    custom_axes: Optional[CoordinateAxes] = None
    initial_joint_positions: Optional[List[float]] = Field(default=None)
    max_iterations: int = Field(200, ge=1, le=500)
    tolerance: float = Field(1e-4, gt=0.0)
    damping: float = Field(0.6, gt=0.0)

    @field_validator("translations")
    @classmethod
    def _validate_translations(cls, value: List[List[float]]) -> List[List[float]]:
        if not value:
            raise ValueError("translations must contain at least one target.")
        for idx, vec in enumerate(value):
            if len(vec) != 3:
                raise ValueError(f"translation at index {idx} must be length 3.")
        return value

    @field_validator("grip_offsets")
    @classmethod
    def _validate_offsets(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is None:
            return None
        if len(value) == 0:
            raise ValueError("grip_offsets must contain at least one value when provided.")
        return value

    @field_validator("custom_axes")
    @classmethod
    def _require_axes_for_custom(cls, value: Optional[CoordinateAxes], info):
        coordinate_mode = info.data.get("coordinate_mode", "base")
        if coordinate_mode == "custom" and value is None:
            raise ValueError("custom_axes must be provided when coordinate_mode is 'custom'.")
        return value


class IkCandidateResult(BaseModel):
    pose_index: int
    grip_offset: float
    error: float
    iterations: int
    joint_distance: float
    mode_used: Literal["prismatic_joint", "offset"]
    coordinate_mode_used: Literal["base", "custom"]
    urdf_variant: Literal["fixed", "prismatic"]
    joint_positions: List[float]


class RobotIkResponse(BaseModel):
    best: IkCandidateResult
    candidates: List[IkCandidateResult]
    mode: Literal["fixed", "prismatic"]
    urdf_variant_used: Literal["fixed", "prismatic"]
    has_gripper_joint: bool
    gripper_joint_name: Optional[str]

