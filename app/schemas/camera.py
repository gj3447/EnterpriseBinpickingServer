"""
카메라 API 응답을 위한 DTO(Data Transfer Objects)를 정의합니다.
Pydantic의 BaseModel을 사용하여 데이터의 유효성을 검사하고,
타입 힌트와 자동 완성을 지원받습니다.
"""
from pydantic import BaseModel
from typing import List

# --- /camera/status 응답 모델 ---

class DeviceInfo(BaseModel):
    name: str
    serial_number: str
    usb_type: str
    firmware_version: str

class StreamConfig(BaseModel):
    width: int
    height: int
    fps: int

class CameraStatus(BaseModel):
    device_info: DeviceInfo
    last_update: float
    active_stream_config: StreamConfig


# --- /camera/calibration 응답 모델 ---

class Intrinsics(BaseModel):
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str
    coeffs: List[float]

class Extrinsics(BaseModel):
    rotation: List[float]
    translation: List[float]

class CameraCalibration(BaseModel):
    depth_intrinsics: Intrinsics
    color_intrinsics: Intrinsics
    depth_to_color_extrinsics: Extrinsics
