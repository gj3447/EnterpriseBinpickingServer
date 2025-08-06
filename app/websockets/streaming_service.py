import asyncio
import json
from typing import Dict, Any

from app.core.event_bus import EventBus
from app.core.logging import logger
from app.core.event_type import EventType
from app.services.aruco_service import ArucoService
from app.stores.application_store import ApplicationStore
from app.websockets.connection_manager import ConnectionManager
from app.schemas.events import (
    SystemTransformsUpdatePayload, WsColorImageUpdatePayload, WsDepthImageUpdatePayload,
    WsDebugImageUpdatePayload, WsPerspectiveImageUpdatePayload
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
        await self.event_bus.unsubscribe(EventType.SYSTEM_TRANSFORMS_UPDATE.value, self.handle_system_transforms_update)

    # --- Event Handlers ---

    async def handle_ws_color_image_update(self, event_name: str, payload: WsColorImageUpdatePayload):
        stream_id = "color_jpg"
        if self.manager.has_subscribers(stream_id):
            await self.manager.broadcast_bytes(stream_id, payload.jpeg_data)

    async def handle_ws_depth_image_update(self, event_name: str, payload: WsDepthImageUpdatePayload):
        stream_id = "depth_jpg"
        if self.manager.has_subscribers(stream_id):
            await self.manager.broadcast_bytes(stream_id, payload.jpeg_data)

    async def handle_ws_debug_image_update(self, event_name: str, payload: WsDebugImageUpdatePayload):
        stream_id = "aruco_debug_jpg"
        if self.manager.has_subscribers(stream_id):
            await self.manager.broadcast_bytes(stream_id, payload.jpeg_data)

    async def handle_ws_perspective_image_update(self, event_name: str, payload: WsPerspectiveImageUpdatePayload):
        stream_id = "board_perspective_jpg"
        if self.manager.has_subscribers(stream_id):
            await self.manager.broadcast_bytes(stream_id, payload.jpeg_data)

    async def handle_system_transforms_update(self, event_name: str, payload: SystemTransformsUpdatePayload):
        """Broadcasts transform snapshots to subscribers of each frame."""
        tasks = []
        for snapshot in payload.snapshots:
            stream_id = f"transforms_{snapshot.frame}"
            if self.manager.has_subscribers(stream_id):
                tasks.append(
                    self.manager.broadcast_text(stream_id, snapshot.model_dump_json())
                )
        if tasks:
            await asyncio.gather(*tasks)
            logger.debug("Broadcasted transform snapshots.")
