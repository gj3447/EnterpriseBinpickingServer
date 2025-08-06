from typing import Dict, Set
import asyncio
from fastapi import WebSocket
from starlette.websockets import WebSocketState

from app.core.logging import logger

class ConnectionManager:
    """
    웹소켓 연결을 중앙에서 관리하는 클래스입니다.
    - 스트림 ID별로 클라이언트 구독을 관리합니다.
    - 불안정한 클라이언트가 전체 스트림에 영향을 주지 않도록 안정적인 브로드캐스팅을 보장합니다.
    """
    def __init__(self):
        self.subscriptions: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, stream_id: str, websocket: WebSocket):
        """클라이언트를 특정 스트림의 구독자로 추가합니다."""
        await websocket.accept()
        async with self._lock:
            if stream_id not in self.subscriptions:
                self.subscriptions[stream_id] = set()
            self.subscriptions[stream_id].add(websocket)
            logger.info(f"Client connected to stream '{stream_id}'. Total subscribers: {len(self.subscriptions[stream_id])}")

    async def disconnect(self, stream_id: str, websocket: WebSocket):
        """클라이언트 연결을 해제하고 구독자 목록에서 제거합니다."""
        async with self._lock:
            if stream_id in self.subscriptions and websocket in self.subscriptions[stream_id]:
                self.subscriptions[stream_id].remove(websocket)
                if not self.subscriptions[stream_id]:
                    del self.subscriptions[stream_id]
                logger.info(f"Client disconnected from stream '{stream_id}'.")

    def has_subscribers(self, stream_id: str) -> bool:
        """특정 스트림에 활성 구독자가 있는지 확인합니다."""
        return stream_id in self.subscriptions and len(self.subscriptions[stream_id]) > 0

    async def broadcast_bytes(self, stream_id: str, data: bytes):
        """특정 스트림의 모든 구독자에게 bytes 데이터를 전송합니다."""
        async with self._lock:
            if stream_id in self.subscriptions:
                # 브로드캐스팅 중 연결이 끊어지는 경우를 대비해 구독자 목록 복사
                subscribers = list(self.subscriptions[stream_id])
                tasks = [self._send_bytes_safely(websocket, data) for websocket in subscribers]
                await asyncio.gather(*tasks)

    async def broadcast_text(self, stream_id: str, data: str):
        """특정 스트림의 모든 구독자에게 text 데이터를 전송합니다."""
        async with self._lock:
            if stream_id in self.subscriptions:
                subscribers = list(self.subscriptions[stream_id])
                tasks = [self._send_text_safely(websocket, data) for websocket in subscribers]
                await asyncio.gather(*tasks)

    async def _send_bytes_safely(self, websocket: WebSocket, data: bytes):
        """
        단일 클라이언트에게 bytes를 전송하고, 실패 시 자동으로 연결을 해제합니다.
        """
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_bytes(data)
            except Exception as e:
                logger.warning(f"Failed to send bytes to client {websocket.client}: {e}. Disconnecting.")
                await self._handle_failed_connection(websocket)

    async def _send_text_safely(self, websocket: WebSocket, data: str):
        """
        단일 클라이언트에게 text를 전송하고, 실패 시 자동으로 연결을 해제합니다.
        """
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_text(data)
            except Exception as e:
                logger.warning(f"Failed to send text to client {websocket.client}: {e}. Disconnecting.")
                await self._handle_failed_connection(websocket)

    async def _handle_failed_connection(self, websocket: WebSocket):
        """
        전송에 실패한 클라이언트를 모든 구독 목록에서 제거합니다.
        """
        async with self._lock:
            for stream_id in list(self.subscriptions.keys()):
                if websocket in self.subscriptions[stream_id]:
                    self.subscriptions[stream_id].remove(websocket)
                    if not self.subscriptions[stream_id]:
                        del self.subscriptions[stream_id]
                    logger.info(f"Removed failed client from stream '{stream_id}'.")
