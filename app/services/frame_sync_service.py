import asyncio
from typing import Optional, Any

from app.core.event_bus import EventBus
from app.core.event_type import EventType
from app.core.logging import logger
from app.schemas.events import ColorImageReceivedPayload, DepthImageReceivedPayload, SyncFrameReadyPayload

class FrameSyncService:
    """
    Listens for individual color and depth image events and publishes a
    'SYNC_FRAME_READY' event only when a time-synchronized pair is available.
    This implementation is optimized to hold locks for the shortest possible duration.
    """
    def __init__(self, event_bus: EventBus, tolerance_ms: float = 150.0):
        self.event_bus = event_bus
        self._is_running = False
        self._tolerance_sec = tolerance_ms / 1000.0
        
        self._latest_color: Optional[ColorImageReceivedPayload] = None
        self._latest_depth: Optional[DepthImageReceivedPayload] = None
        self._lock = asyncio.Lock()
        
        # 성능 측정용
        self._synced_pairs = 0
        self._color_received = 0
        self._depth_received = 0
        self._last_log_time = 0

    async def start(self):
        if self._is_running: return
        self._is_running = True
        await self.event_bus.subscribe(EventType.COLOR_IMAGE_RECEIVED.value, self.handle_color_image)
        await self.event_bus.subscribe(EventType.DEPTH_IMAGE_RECEIVED.value, self.handle_depth_image)
        logger.info("FrameSyncService started and subscribed to image events.")

    async def stop(self):
        if not self._is_running: return
        self._is_running = False
        await self.event_bus.unsubscribe(EventType.COLOR_IMAGE_RECEIVED.value, self.handle_color_image)
        await self.event_bus.unsubscribe(EventType.DEPTH_IMAGE_RECEIVED.value, self.handle_depth_image)
        logger.info("FrameSyncService stopped.")

    async def handle_color_image(self, event_name: str, payload: ColorImageReceivedPayload):
        self._color_received += 1
        payload_to_publish = None
        async with self._lock:
            self._latest_color = payload
            payload_to_publish = self._get_synced_payload_and_clear_state()
            
            # 성능 측정 카운터 리셋 (1초마다)
            current_time = asyncio.get_event_loop().time()
            if current_time - self._last_log_time >= 1.0:
                self._synced_pairs = 0
                self._color_received = 0
                self._depth_received = 0
                self._last_log_time = current_time
        
        if payload_to_publish:
            await self.event_bus.publish(EventType.SYNC_FRAME_READY.value, payload_to_publish)

    async def handle_depth_image(self, event_name: str, payload: DepthImageReceivedPayload):
        self._depth_received += 1
        payload_to_publish = None
        async with self._lock:
            self._latest_depth = payload
            payload_to_publish = self._get_synced_payload_and_clear_state()

        if payload_to_publish:
            await self.event_bus.publish(EventType.SYNC_FRAME_READY.value, payload_to_publish)

    def _get_synced_payload_and_clear_state(self) -> Optional[SyncFrameReadyPayload]:
        """
        Atomically checks for a synchronized pair. If found, it clears the
        internal state and returns the payload to be published.
        This method MUST be called within a lock.
        """
        if not (self._latest_color and self._latest_depth):
            return None

        if abs(self._latest_color.timestamp - self._latest_depth.timestamp) <= self._tolerance_sec:
            # Pair found, create the payload using the latest of the two timestamps
            sync_payload = SyncFrameReadyPayload(
                timestamp=max(self._latest_color.timestamp, self._latest_depth.timestamp),
                color_image_data=self._latest_color.image_data,
                depth_image_data=self._latest_depth.image_data
            )
            
            self._synced_pairs += 1
            
            # Clear state for next pair
            self._latest_color = None
            self._latest_depth = None
            return sync_payload
        else:
            # Timestamps are too far apart, discard the older frame to allow a new pair to form
            if self._latest_color.timestamp < self._latest_depth.timestamp:
                self._latest_color = None
            else:
                self._latest_depth = None
            return None
