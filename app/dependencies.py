"""
애플리케이션의 의존성(dependency)을 생성하고 관리합니다.
"""
import json
from typing import Optional
from app.schemas.aruco import Pose
from app.core.logging import logger

from app.stores.application_store import ApplicationStore
from app.services.camera_service import CameraService
from app.services.aruco_service import ArucoService
from app.services.image_service import ImageService
from app.services.frame_sync_service import FrameSyncService  # 임포트 추가
from app.websockets.connection_manager import ConnectionManager
from app.websockets.streaming_service import StreamingService
from app.core.event_bus import EventBus


# --- 설정 파일 경로 ---
ARUCO_BOARD_CONFIG_PATH = "app/config/aruco_place.csv"
ROBOT_POSITION_CONFIG_PATH = "app/config/robot_position.json"

def _load_robot_pose_from_config(path: str) -> Optional[Pose]:
    try:
        with open(path, 'r') as f: data = json.load(f)
        return Pose(translation=[data['x'], data['y'], data['z']], orientation_quaternion=[0.0, 0.0, 0.0, 1.0])
    except Exception as e:
        logger.error(f"Could not load robot position from {path}: {e}")
        return None

# --- 단일 인스턴스 생성 (의존성 순서에 주의) ---

# 1. 의존성이 없는 기본 서비스들
_store = ApplicationStore()
_event_bus = EventBus()
_connection_manager = ConnectionManager()

# 의존성 연결: EventBus가 Store의 EventHandler를 사용하도록 설정
_event_bus.set_event_handler(_store.events)



# 2. 기본 서비스에 의존하는 서비스들
_frame_sync_service = FrameSyncService(event_bus=_event_bus, tolerance_ms=150) # FrameSyncService 인스턴스 생성
_image_service = ImageService(store=_store, event_bus=_event_bus)
_camera_service = CameraService(store=_store, event_bus=_event_bus)
_robot_pose = _load_robot_pose_from_config(ROBOT_POSITION_CONFIG_PATH)
_aruco_service = ArucoService(store=_store, event_bus=_event_bus, board_config_path=ARUCO_BOARD_CONFIG_PATH, robot_pose=_robot_pose)

# 3. 여러 서비스에 의존하는 최종 서비스
_streaming_service = StreamingService(
    connection_manager=_connection_manager,
    store=_store,
    event_bus=_event_bus
)


# --- 의존성 공급자(Provider) 함수 ---
def get_store() -> ApplicationStore: return _store
def get_event_bus() -> EventBus: return _event_bus
def get_camera_service() -> CameraService: return _camera_service
def get_aruco_service() -> ArucoService: return _aruco_service
def get_image_service() -> ImageService: return _image_service
def get_frame_sync_service() -> FrameSyncService: return _frame_sync_service  # 공급자 함수 추가
def get_connection_manager() -> ConnectionManager: return _connection_manager
def get_streaming_service() -> StreamingService: return _streaming_service
