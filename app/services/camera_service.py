import asyncio
import httpx
import websockets
import numpy as np
import time
from typing import Optional, List
from pydantic import ValidationError

from app.stores.application_store import ApplicationStore
from app.core.config import settings
from app.schemas.camera import CameraStatus, CameraCalibration
from app.schemas.events import (
    ColorImageReceivedPayload, DepthImageReceivedPayload,
    ColorJpegReceivedPayload, DepthJpegReceivedPayload
)
from app.core.event_bus import EventBus
from app.core.logging import logger
from app.core.event_type import EventType

class CameraService:
    def __init__(self, store: ApplicationStore, event_bus: EventBus):
        self.store = store
        self.event_bus = event_bus
        self.base_url = settings.CAMERA_API_BASE_URL
        self.api_client = httpx.AsyncClient(base_url=f"http://{self.base_url}")
        self._is_running = False
        self.tasks: List[asyncio.Task] = []

    async def _periodic_api_sync(self):
        # ... (이전과 동일)
        interval = settings.API_SYNC_INTERVAL_SECONDS
        while self._is_running:
            await asyncio.sleep(interval)
            try:
                logger.info(f"Performing periodic API sync (interval: {interval}s)...")
                await self.sync_with_api()
            except Exception as e:
                logger.error(f"An error occurred during periodic API sync: {e}")
    
    async def start(self):
        if not await self.sync_with_api():
            logger.error("Initial API sync failed. Aborting service start.")
            return
        self._is_running = True
        
        self.tasks.append(asyncio.create_task(self._periodic_api_sync()))
        logger.info(f"Periodic API sync scheduled every {settings.API_SYNC_INTERVAL_SECONDS} seconds.")

        config = self.store.device.get_config()
        if not config:
            logger.error("Failed to get device config from store. Aborting websocket listeners.")
            return
            
        width, height = config.width, config.height
        
        # Color 스트림 시작 (설정에 따라 raw 또는 jpeg)
        if settings.COLOR_STREAM_MODE == "jpeg":
            self.tasks.append(asyncio.create_task(self._listen_to_jpeg_stream("color_jpeg", "color")))
        else:
            self.tasks.append(asyncio.create_task(self._listen_to_stream("color_raw", width, height, 3)))
        
        # Depth 스트림 시작 (설정에 따라 raw 또는 jpeg)
        if settings.DEPTH_STREAM_MODE == "jpeg":
            self.tasks.append(asyncio.create_task(self._listen_to_jpeg_stream("depth_jpeg", "depth")))
        else:
            self.tasks.append(asyncio.create_task(self._listen_to_stream("depth_raw", width, height, 2)))
            
        logger.info(f"Websocket listeners started - Color: {settings.COLOR_STREAM_MODE}, Depth: {settings.DEPTH_STREAM_MODE}")

    async def sync_with_api(self) -> bool:
        # ... (이전과 동일)
        try:
            logger.info("Attempting to sync with camera API...")
            status_res = await self.api_client.get("/camera/status")
            calib_res = await self.api_client.get("/camera/calibration")
            status_res.raise_for_status()
            calib_res.raise_for_status()
            camera_status = CameraStatus(**status_res.json())
            camera_calibration = CameraCalibration(**calib_res.json())
            self.store.device.update_status(camera_status)
            self.store.calibration.set_data(camera_calibration)
            logger.info("Successfully synced with camera API and validated data.")
            return True
        except (httpx.RequestError, ValidationError) as e:
            logger.error(f"Failed to sync with camera API: {e}")
            return False

    async def _listen_to_stream(self, stream_name: str, width: int, height: int, bytes_per_pixel: int):
        uri = f"ws://{self.base_url}/ws/{stream_name}"
        bytes_per_frame = width * height * bytes_per_pixel
        frames_received = 0
        last_log_time = time.time()
        
        while self._is_running:
            try:
                async with websockets.connect(uri, max_size=None) as websocket:
                    logger.info(f"Connected to {stream_name} websocket.")
                    while self._is_running:
                        # WebSocket 수신 시간 측정
                        recv_start = time.time()
                        message = await websocket.recv()
                        recv_time = time.time() - recv_start
                        
                        if isinstance(message, bytes) and len(message) == bytes_per_frame:
                            dtype = np.uint16 if "depth" in stream_name else np.uint8
                            shape = (height, width) if bytes_per_pixel < 3 else (height, width, 3)
                            image = np.frombuffer(message, dtype=dtype).reshape(shape)
                            
                            current_timestamp = time.time()
                            frames_received += 1
                        
                            # 성능 측정은 계속하지만 로깅은 제거
                            if current_timestamp - last_log_time >= 1.0:
                                frames_received = 0
                                last_log_time = current_timestamp
                            
                            if "color" in stream_name:
                                self.store.camera_raw.update_color_image(image, current_timestamp)
                                payload = ColorImageReceivedPayload(timestamp=current_timestamp, image_data=image)
                                await self.event_bus.publish(EventType.COLOR_IMAGE_RECEIVED.value, payload)
                            elif "depth" in stream_name:
                                self.store.camera_raw.update_depth_image(image, current_timestamp)
                                payload = DepthImageReceivedPayload(timestamp=current_timestamp, image_data=image)
                                await self.event_bus.publish(EventType.DEPTH_IMAGE_RECEIVED.value, payload)
            except Exception as e:
                logger.warning(f"Websocket connection error for {stream_name}: {e}. Retrying in 5s...")
                await asyncio.sleep(5)
        
    async def _listen_to_jpeg_stream(self, stream_name: str, stream_type: str):
        """JPEG 스트림을 수신하는 WebSocket 리스너"""
        uri = f"ws://{self.base_url}/ws/{stream_name}"
        frames_received = 0
        last_log_time = time.time()
        
        while self._is_running:
            try:
                async with websockets.connect(uri, max_size=None) as websocket:
                    logger.info(f"Connected to {stream_name} JPEG websocket.")
                    while self._is_running:
                        # WebSocket 수신
                        message = await websocket.recv()
                        
                        if isinstance(message, bytes):
                            current_timestamp = time.time()
                            frames_received += 1
                            
                            # 성능 측정은 계속하지만 로깅은 제거
                            if current_timestamp - last_log_time >= 1.0:
                                frames_received = 0
                                last_log_time = current_timestamp
                            
                            if stream_type == "color":
                                # Store에 JPEG 저장
                                self.store.images.update_color_jpeg(message, current_timestamp)
                                # JPEG 이벤트 발행
                                payload = ColorJpegReceivedPayload(timestamp=current_timestamp, jpeg_data=message)
                                await self.event_bus.publish(EventType.COLOR_JPEG_RECEIVED.value, payload)
                            elif stream_type == "depth":
                                # Store에 JPEG 저장
                                self.store.images.update_depth_jpeg(message, current_timestamp)
                                # JPEG 이벤트 발행
                                payload = DepthJpegReceivedPayload(timestamp=current_timestamp, jpeg_data=message)
                                await self.event_bus.publish(EventType.DEPTH_JPEG_RECEIVED.value, payload)
            except Exception as e:
                logger.warning(f"Websocket connection error for {stream_name}: {e}. Retrying in 5s...")
                await asyncio.sleep(5)
    
    async def stop(self):
        self._is_running = False
        for task in self.tasks:
            task.cancel()
        await self.api_client.aclose()
        logger.info("CameraService stopping...")
