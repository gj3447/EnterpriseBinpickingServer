from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field
from pathlib import Path

env_path = Path(".") / "config" / ".env"

class AppSettings(BaseSettings):
    """
    pydantic-settings를 사용하여 환경 변수 및 .env 파일로부터 설정을 관리합니다.
    """
    # --- Camera Settings ---
    CAMERA_API_BASE_URL: str = "192.168.0.197:51000"
    API_SYNC_INTERVAL_SECONDS: int = 300
    
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


    # pydantic-settings 설정
    model_config = SettingsConfigDict(env_file=env_path, env_file_encoding='utf-8')

settings = AppSettings()
