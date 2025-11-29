import asyncio
from typing import Any, Coroutine

from app.core.event_bus import EventBus
from app.core.logging import logger
from app.core.event_type import EventType
from app.stores.application_store import ApplicationStore
from app.websockets.connection_manager import ConnectionManager
from app.schemas.events import (
    SystemTransformsUpdatePayload, WsColorImageUpdatePayload, WsDepthImageUpdatePayload,
    WsDebugImageUpdatePayload, WsPerspectiveImageUpdatePayload, WsPointcloudUpdatePayload
)

class StreamingService:
    """
    Subscribes to specific WebSocket events and broadcasts the corresponding
    data to clients connected to the relevant stream.
    """
    def __init__(
        self,
        connection_manager: ConnectionManager,
        store: ApplicationStore,
        event_bus: EventBus,
    ):
        self.manager = connection_manager
        self.store = store
        self.event_bus = event_bus
        self._is_running = False

    def start_listening(self):
        if self._is_running: return
        self._is_running = True
        asyncio.create_task(self.subscribe_to_events())
        logger.info("StreamingService started and subscribed to WebSocket events.")

    async def subscribe_to_events(self):
        """Subscribes to all relevant WebSocket events."""
        await self.event_bus.subscribe(EventType.WS_COLOR_IMAGE_UPDATE.value, self.handle_ws_color_image_update)
        await self.event_bus.subscribe(EventType.WS_DEPTH_IMAGE_UPDATE.value, self.handle_ws_depth_image_update)
        await self.event_bus.subscribe(EventType.WS_DEBUG_IMAGE_UPDATE.value, self.handle_ws_debug_image_update)
        await self.event_bus.subscribe(EventType.WS_PERSPECTIVE_IMAGE_UPDATE.value, self.handle_ws_perspective_image_update)
        await self.event_bus.subscribe(EventType.WS_POINTCLOUD_UPDATE.value, self.handle_ws_pointcloud_update)
        await self.event_bus.subscribe(EventType.SYSTEM_TRANSFORMS_UPDATE.value, self.handle_system_transforms_update)

    def stop_listening(self):
        if not self._is_running: return
        self._is_running = False
        asyncio.create_task(self.unsubscribe_from_events())
        logger.info("StreamingService stopped and unsubscribed from events.")

    async def unsubscribe_from_events(self):
        """Unsubscribes from all WebSocket events."""
        await self.event_bus.unsubscribe(EventType.WS_COLOR_IMAGE_UPDATE.value, self.handle_ws_color_image_update)
        await self.event_bus.unsubscribe(EventType.WS_DEPTH_IMAGE_UPDATE.value, self.handle_ws_depth_image_update)
        await self.event_bus.unsubscribe(EventType.WS_DEBUG_IMAGE_UPDATE.value, self.handle_ws_debug_image_update)
        await self.event_bus.unsubscribe(EventType.WS_PERSPECTIVE_IMAGE_UPDATE.value, self.handle_ws_perspective_image_update)
        await self.event_bus.unsubscribe(EventType.WS_POINTCLOUD_UPDATE.value, self.handle_ws_pointcloud_update)
        await self.event_bus.unsubscribe(EventType.SYSTEM_TRANSFORMS_UPDATE.value, self.handle_system_transforms_update)

    # --- Event Handlers ---

    async def handle_ws_color_image_update(self, event_name: str, payload: WsColorImageUpdatePayload):
        stream_id = "color_jpg"
        if self.manager.has_subscribers(stream_id):
            self._schedule_broadcast(
                stream_id,
                self.manager.broadcast_bytes(stream_id, payload.jpeg_data),
            )

    async def handle_ws_depth_image_update(self, event_name: str, payload: WsDepthImageUpdatePayload):
        stream_id = "depth_jpg"
        if self.manager.has_subscribers(stream_id):
            self._schedule_broadcast(
                stream_id,
                self.manager.broadcast_bytes(stream_id, payload.jpeg_data),
            )

    async def handle_ws_debug_image_update(self, event_name: str, payload: WsDebugImageUpdatePayload):
        stream_id = "aruco_debug_jpg"
        if self.manager.has_subscribers(stream_id):
            self._schedule_broadcast(
                stream_id,
                self.manager.broadcast_bytes(stream_id, payload.jpeg_data),
            )

    async def handle_ws_perspective_image_update(self, event_name: str, payload: WsPerspectiveImageUpdatePayload):
        stream_id = "board_perspective_jpg"
        if self.manager.has_subscribers(stream_id):
            self._schedule_broadcast(
                stream_id,
                self.manager.broadcast_bytes(stream_id, payload.jpeg_data),
            )
    
    async def handle_ws_pointcloud_update(self, event_name: str, payload: WsPointcloudUpdatePayload):
        stream_id = "pointcloud"
        if self.manager.has_subscribers(stream_id):
            # 포인트클라우드 데이터는 JSON으로 전송
            json_data = payload.model_dump_json()
            self._schedule_broadcast(
                stream_id,
                self.manager.broadcast_text(stream_id, json_data),
            )

    async def handle_system_transforms_update(self, event_name: str, payload: SystemTransformsUpdatePayload):
        """Broadcasts transform snapshots to subscribers of each frame."""
        scheduled = False
        for snapshot in payload.snapshots:
            stream_id = f"transforms_{snapshot.frame}"
            if self.manager.has_subscribers(stream_id):
                self._schedule_broadcast(
                    stream_id,
                    self.manager.broadcast_text(stream_id, snapshot.model_dump_json()),
                )
                scheduled = True
        if scheduled:
            logger.debug("Scheduled transform snapshot broadcasts.")

    def _schedule_broadcast(self, stream_id: str, coro: Coroutine[Any, Any, None]) -> None:
        """브로드캐스트 코루틴을 백그라운드 태스크로 실행하고 예외를 로깅합니다."""
        task = asyncio.create_task(coro)

        def _callback(finished_task: asyncio.Task, sid: str = stream_id) -> None:
            try:
                finished_task.result()
            except asyncio.CancelledError:
                logger.debug("Broadcast task cancelled for stream '%s'.", sid)
            except Exception as exc:  # pragma: no cover
                logger.warning("Broadcast task failed for stream '%s': %s", sid, exc, exc_info=True)

        task.add_done_callback(_callback)
