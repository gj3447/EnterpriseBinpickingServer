from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field
from pathlib import Path
from typing import Optional, Literal

env_path = Path("app") / "config" / ".env"

class AppSettings(BaseSettings):
    """
    pydantic-settings를 사용하여 환경 변수 및 .env 파일로부터 설정을 관리합니다.
    """
    # --- General ---
    LOG_LEVEL: str = Field(
        "INFO",
        description="전체 애플리케이션 로그 레벨 (예: DEBUG, INFO, WARNING)",
    )

    # --- Camera Settings ---
    CAMERA_API_BASE_URL: str = "192.168.0.197:51000"
    API_SYNC_INTERVAL_SECONDS: int = 300
    CAMERA_WS_CONNECT_TIMEOUT_SECONDS: float = Field(10.0, description="카메라 WebSocket 연결 타임아웃 (초)")
    CAMERA_WS_RECV_TIMEOUT_SECONDS: float = Field(3.0, description="카메라 WebSocket 수신 타임아웃 (초)")
    CAMERA_WS_RECONNECT_DELAY_SECONDS: float = Field(5.0, description="재연결 시도 간격 (초)")
    CAMERA_WS_PING_INTERVAL_SECONDS: Optional[float] = Field(20.0, description="ping 프레임 전송 주기 (초, 0 또는 음수면 비활성화)")
    CAMERA_WS_PING_TIMEOUT_SECONDS: Optional[float] = Field(20.0, description="ping 응답 대기 타임아웃 (초)")
    CAMERA_WS_CLOSE_TIMEOUT_SECONDS: float = Field(5.0, description="연결 종료 시 대기 시간 (초)")

    # --- Robot / IK Settings ---
    ROBOT_URDF_PATH_FIXED: str = Field(
        "app/static/urdf/dsr_description2/urdf/a0509.urdf",
        description="기본(고정) URDF 파일 경로",
    )
    ROBOT_URDF_PATH_PRISMATIC: Optional[str] = Field(
        default=None,
        description="프리스매틱 그리퍼가 포함된 URDF 파일 경로 (선택)",
    )
    ROBOT_URDF_MODE: Literal["auto", "fixed", "prismatic"] = Field(
        "auto",
        description="어떤 URDF를 사용할지 결정 (`fixed`, `prismatic`, `auto`)",
    )
    GRIPPER_JOINT_NAME: Optional[str] = Field(default=None, description="프리스매틱 그리퍼 관절 이름 (없으면 None)")
    ROBOT_IK_BACKEND: Literal["ikpy"] = Field(
        "ikpy",
        description="IK 계산에 사용할 라이브러리 백엔드",
    )
    IKPY_END_EFFECTOR_FRAME: str = Field(
        "link_6",
        description="ikpy 백엔드가 사용할 말단 프레임 이름 (그리퍼 없이 사용 시 flange 프레임)",
    )
    IKPY_GRIPPER_LENGTH: float = Field(
        0.12,
        gt=0.0,
        description="ikpy 백엔드에서 그리퍼 길이를 보정하기 위해 사용할 추가 높이 (미터)",
    )
    IK_JOINT_DISTANCE_WEIGHT: float = Field(
        0.1,
        ge=0.0,
        description="ikpy IK에서 관절 변위 비용에 적용할 가중치",
    )
    IK_MIN_Z: float = Field(
        0.0,
        description="IK 목표 및 결과에서 허용하는 최소 Z 높이 (ground plane)",
    )
    
    # --- Stream Mode Settings ---
    COLOR_STREAM_MODE: str = Field("jpeg", description="Color 스트림 모드: 'jpeg' 또는 'raw'")
    DEPTH_STREAM_MODE: str = Field("raw", description="Depth 스트림 모드: 'jpeg' 또는 'raw'")

    # --- Physical Asset Settings ---
    BOARD_WIDTH_M: float = 0.8
    BOARD_HEIGHT_M: float = 0.7
    ARUCO_SINGLE_MARKER_SIZE_M: float = Field(0.05, description="개별 ArUco 마커의 한 변 길이 (미터 단위)")
    ARUCO_TARGET_STREAM_IDS: list[str] = Field(["color_raw"], description="ArUco 감지를 수행할 이미지 스트림 ID 목록")
    
    # --- Pointcloud Settings ---
    MAX_POINTCLOUD_DEPTH_M: float = Field(3.0, description="포인트클라우드 생성 시 최대 깊이 (미터 단위)")
    POINTCLOUD_DOWNSAMPLE_FACTOR: int = Field(4, description="포인트클라우드 다운샘플링 팩터 (4 = 4x4 다운샘플링)")

    @computed_field(return_type=str)
    @property
    def robot_urdf_path(self) -> str:
        """환경 설정에 따라 사용할 URDF 경로를 반환합니다."""
        mode = self.ROBOT_URDF_MODE
        prismatic_path = self.ROBOT_URDF_PATH_PRISMATIC

        if mode == "fixed":
            return self.ROBOT_URDF_PATH_FIXED

        if mode == "prismatic":
            return prismatic_path or self.ROBOT_URDF_PATH_FIXED

        # auto
        if prismatic_path:
            return prismatic_path
        return self.ROBOT_URDF_PATH_FIXED

    @computed_field(return_type=str)
    @property
    def robot_urdf_variant(self) -> str:
        """선택된 URDF 변형(fixed/prismatic/fallback)을 표시합니다."""
        mode = self.ROBOT_URDF_MODE
        prismatic_path = self.ROBOT_URDF_PATH_PRISMATIC

        if mode == "fixed":
            return "fixed"

        if mode == "prismatic":
            return "prismatic" if prismatic_path else "fixed_fallback"

        # auto
        return "prismatic" if prismatic_path else "fixed"


    # pydantic-settings 설정
    model_config = SettingsConfigDict(env_file=env_path, env_file_encoding='utf-8')

settings = AppSettings()
