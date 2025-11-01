from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from app.websockets.connection_manager import ConnectionManager
from app.dependencies import get_connection_manager
from app.core.logging import logger

router = APIRouter()

async def _websocket_handler(websocket: WebSocket, stream_id: str, manager: ConnectionManager):
    """
    웹소켓 연결을 처리하는 공통 핸들러 함수입니다.
    - 클라이언트가 접속하면 ConnectionManager에 등록합니다.
    - 연결이 끊어지면 자동으로 등록을 해제합니다.
    """
    await manager.connect(stream_id, websocket)
    try:
        # 연결을 유지하고 클라이언트로부터의 메시지를 수신 대기합니다.
        # 이 루프는 연결이 끊어지면 WebSocketDisconnect 예외를 발생시킵니다.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"Client at {websocket.client} disconnected from stream '{stream_id}'.")
    finally:
        # 예외 발생 여부와 관계없이 항상 연결 해제 처리
        await manager.disconnect(stream_id, websocket)

# --- Image Streams ---

@router.websocket("/ws/color_jpg")
async def ws_color_jpg(
    websocket: WebSocket,
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """실시간 컬러(JPEG) 이미지 스트림을 구독합니다."""
    await _websocket_handler(websocket, "color_jpg", manager)

@router.websocket("/ws/depth_jpg")
async def ws_depth_jpg(
    websocket: WebSocket,
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """실시간 깊이(JPEG) 이미지 스트림을 구독합니다."""
    await _websocket_handler(websocket, "depth_jpg", manager)

@router.websocket("/ws/aruco_debug_jpg")
async def ws_aruco_debug_jpg(
    websocket: WebSocket,
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """ArUco 마커와 보드 정보가 그려진 디버그 이미지 스트림을 구독합니다."""
    await _websocket_handler(websocket, "aruco_debug_jpg", manager)

@router.websocket("/ws/board_perspective_jpg")
async def ws_board_perspective_jpg(
    websocket: WebSocket,
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """원근 보정된 보드 이미지 스트림을 구독합니다."""
    await _websocket_handler(websocket, "board_perspective_jpg", manager)

# --- Transform Streams ---

@router.websocket("/ws/transforms_camera")
async def ws_transforms_camera(
    websocket: WebSocket,
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """'카메라' 좌표계를 기준으로 모든 객체의 Pose 정보를 구독합니다."""
    await _websocket_handler(websocket, "transforms_camera", manager)

@router.websocket("/ws/transforms_board")
async def ws_transforms_board(
    websocket: WebSocket,
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """'보드' 좌표계를 기준으로 모든 객체의 Pose 정보를 구독합니다."""
    await _websocket_handler(websocket, "transforms_board", manager)

@router.websocket("/ws/transforms_robot")
async def ws_transforms_robot(
    websocket: WebSocket,
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """'로봇' 좌표계를 기준으로 모든 객체의 Pose 정보를 구독합니다."""
    await _websocket_handler(websocket, "transforms_robot", manager)

# --- Pointcloud Stream ---

@router.websocket("/ws/pointcloud")
async def ws_pointcloud(
    websocket: WebSocket,
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """실시간 3D 포인트클라우드 데이터를 구독합니다."""
    await _websocket_handler(websocket, "pointcloud", manager)