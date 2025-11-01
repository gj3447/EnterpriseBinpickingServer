"""ArUco 서비스 설정을 위한 Pydantic 모델"""
from pydantic import BaseModel, Field
from typing import Optional

class TemporalFilterConfig(BaseModel):
    """시간적 필터링 설정"""
    enabled: bool = True
    board_alpha: float = Field(0.8, ge=0.0, le=1.0)
    marker_alpha: float = Field(0.7, ge=0.0, le=1.0)
    timeout_seconds: float = Field(1.0, gt=0)

class DepthQualityConfig(BaseModel):
    """깊이 품질 검증 설정"""
    enabled: bool = True
    max_relative_std: float = Field(0.05, gt=0)

class DepthSamplingConfig(BaseModel):
    """깊이 샘플링 설정"""
    window_size_small: int = Field(3, ge=1)
    window_size_large: int = Field(5, ge=1)
    small_marker_threshold_m: float = Field(0.05, gt=0)
    use_interior_sampling: bool = Field(True, description="마커 내부 영역 사용 여부")
    interior_offset_ratio: float = Field(0.2, ge=0.1, le=0.4, description="코너에서 중심으로의 오프셋 비율")
    min_valid_samples: int = Field(3, ge=1, le=4, description="최소 유효 샘플 개수")
    quality_check: DepthQualityConfig

class RansacConfig(BaseModel):
    """RANSAC 설정"""
    enabled: bool = True
    inlier_threshold_m: float = Field(0.01, gt=0)
    inlier_threshold_small_m: float = Field(0.005, gt=0)
    max_iterations: int = Field(100, ge=1)
    min_inlier_ratio: float = Field(0.6, ge=0.0, le=1.0)

class HybridModeConfig(BaseModel):
    """하이브리드 포즈 추정 설정"""
    enabled: bool = Field(True, description="하이브리드 모드 활성화")
    depth_weight: float = Field(0.7, ge=0.0, le=1.0, description="깊이 기반 추정의 가중치")
    min_depth_consistency: float = Field(0.8, ge=0.0, le=1.0, description="최소 깊이 일관성 비율")
    scale_correction_threshold: float = Field(0.1, gt=0, description="스케일 보정 임계값")
    use_weighted_average: bool = Field(True, description="가중 평균 사용 여부")

class PoseEstimationConfig(BaseModel):
    """포즈 추정 설정"""
    min_points_for_board: int = Field(6, ge=3)
    use_ippe_for_markers: bool = True
    corner_refinement: Optional[str] = "CORNER_REFINE_SUBPIX"
    hybrid_mode: HybridModeConfig

class DebugConfig(BaseModel):
    """디버깅 설정"""
    log_metrics: bool = False
    log_filter_resets: bool = True

class ArucoConfig(BaseModel):
    """ArUco 서비스 전체 설정"""
    dictionary: str = "DICT_4X4_250"
    marker_size_m: float = Field(0.05, gt=0)
    board_config_file: str = "aruco_place.csv"
    temporal_filter: TemporalFilterConfig
    depth_sampling: DepthSamplingConfig
    ransac: RansacConfig
    pose_estimation: PoseEstimationConfig
    debug: DebugConfig
