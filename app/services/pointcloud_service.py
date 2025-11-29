import asyncio
import time
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2

from app.schemas.events import SyncFrameReadyPayload, WsPointcloudUpdatePayload
from app.schemas.camera import CameraCalibration, Intrinsics
from app.core.config import settings
from app.core.event_bus import EventBus
from app.core.logging import logger
from app.core.event_type import EventType
from app.stores.application_store import ApplicationStore


class PointcloudService:
    """
    동기화된 컬러/깊이 이미지에서 포인트클라우드를 생성하고 관리하는 서비스입니다.
    """
    
    def __init__(self, store: ApplicationStore, event_bus: EventBus):
        """PointcloudService를 초기화합니다."""
        self.store = store
        self.event_bus = event_bus
        self._is_running = False
        
        # 최신 프레임만 처리하기 위한 coalesce 버퍼
        self._latest_sync_frame: Optional[Tuple[SyncFrameReadyPayload, float]] = None
        self._processing = False
        
        # 포인트클라우드 생성 설정
        self._max_depth_m = settings.MAX_POINTCLOUD_DEPTH_M  # 최대 깊이 (미터)
        self._downsample_factor = settings.POINTCLOUD_DOWNSAMPLE_FACTOR  # 다운샘플링 팩터
        
    async def start(self):
        """서비스를 시작하고 이벤트 구독을 등록합니다."""
        if self._is_running:
            return
        self._is_running = True
        
        # 동기화된 프레임 이벤트 구독
        await self.event_bus.subscribe(EventType.SYNC_FRAME_READY.value, self.handle_sync_frame)
        logger.info("PointcloudService started and subscribed to SYNC_FRAME_READY events.")
        
    async def stop(self):
        """서비스를 중지하고 이벤트 구독을 해제합니다."""
        if not self._is_running:
            return
        self._is_running = False
        
        await self.event_bus.unsubscribe(EventType.SYNC_FRAME_READY.value, self.handle_sync_frame)
        logger.info("PointcloudService stopped and unsubscribed from events.")
        
    async def handle_sync_frame(self, event_name: str, payload: SyncFrameReadyPayload):
        """SYNC_FRAME_READY 이벤트를 처리하여 포인트클라우드를 생성합니다."""
        try:
            # 최신 프레임만 유지하고 처리 루프가 없으면 시작
            self._latest_sync_frame = (payload, payload.timestamp)
            if not self._processing:
                self._processing = True
                asyncio.create_task(self._process_pointcloud_loop())
        except Exception as e:
            logger.error(f"Error in handle_sync_frame: {e}", exc_info=True)
            
    async def _process_pointcloud_loop(self):
        """최신 프레임만 처리하는 루프."""
        try:
            while True:
                snapshot = self._latest_sync_frame
                # 읽은 즉시 비워서 다음 주기 동안 최신값만 남도록 함
                self._latest_sync_frame = None
                if not snapshot:
                    break
                    
                payload, timestamp = snapshot
                try:
                    await self._process_and_publish_pointcloud(payload, timestamp)
                except Exception as e:
                    logger.error(f"Error while processing pointcloud: {e}", exc_info=True)
        finally:
            self._processing = False
            # 종료 직후 새 프레임이 들어온 경우 재기동
            if self._latest_sync_frame is not None and not self._processing:
                self._processing = True
                asyncio.create_task(self._process_pointcloud_loop())
                
    async def _process_and_publish_pointcloud(self, payload: SyncFrameReadyPayload, timestamp: float):
        """포인트클라우드를 생성하고 발행합니다."""
        # 칼리브레이션 데이터 가져오기
        calibration = self.store.calibration.get_data()
        if calibration is None:
            logger.warning("Missing calibration data for generating pointcloud.")
            return
            
        # CPU 바운드 작업(포인트클라우드 생성)을 스레드로 오프로딩
        start = time.perf_counter()
        pointcloud_data = await asyncio.to_thread(
            self._generate_pointcloud,
            payload.color_image_data,
            payload.depth_image_data,
            calibration
        )
        duration_ms = (time.perf_counter() - start) * 1000.0
        
        if pointcloud_data is None:
            return
            
        points, colors = pointcloud_data
        
        # Store에 포인트클라우드 저장
        self.store.pointcloud.update_pointcloud(points, colors, timestamp)
        
        # WebSocket 이벤트 발행 (다운샘플링된 버전)
        downsampled_points, downsampled_colors = self.store.pointcloud.get_downsampled_pointcloud(max_points=50000)
        if downsampled_points is not None:
            ws_payload = WsPointcloudUpdatePayload(
                timestamp=timestamp,
                points=downsampled_points.tolist(),
                colors=downsampled_colors.tolist() if downsampled_colors is not None else None
            )
            await self.event_bus.publish(EventType.WS_POINTCLOUD_UPDATE.value, ws_payload)
        
        logger.debug(
            "Pointcloud generated | points=%d duration=%.1fms downsampled=%s",
            len(points),
            duration_ms,
            "yes" if downsampled_points is not None else "no",
        )
        
    def _generate_pointcloud(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        calibration: CameraCalibration
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """컬러와 깊이 이미지에서 포인트클라우드를 생성합니다."""
        try:
            # 깊이 이미지가 uint16 형식인지 확인
            if depth_image.dtype != np.uint16:
                logger.error(f"Unexpected depth image dtype: {depth_image.dtype}")
                return None
                
            # 이미지 크기 확인
            h, w = depth_image.shape
            color_h, color_w = color_image.shape[:2]
            
            # 깊이와 컬러 이미지 크기가 다른 경우 리사이즈
            if (h, w) != (color_h, color_w):
                # 깊이 이미지를 컬러 이미지 크기로 리사이즈
                depth_image = cv2.resize(depth_image, (color_w, color_h), interpolation=cv2.INTER_NEAREST)
                h, w = color_h, color_w
                
            # 다운샘플링 적용
            if self._downsample_factor > 1:
                h_ds = h // self._downsample_factor
                w_ds = w // self._downsample_factor
                depth_downsampled = cv2.resize(depth_image, (w_ds, h_ds), interpolation=cv2.INTER_NEAREST)
                color_downsampled = cv2.resize(color_image, (w_ds, h_ds), interpolation=cv2.INTER_LINEAR)
            else:
                h_ds, w_ds = h, w
                depth_downsampled = depth_image
                color_downsampled = color_image
                
            # 카메라 내부 파라미터
            depth_intrinsics = calibration.depth_intrinsics
            fx = depth_intrinsics.fx / self._downsample_factor
            fy = depth_intrinsics.fy / self._downsample_factor
            cx = depth_intrinsics.ppx / self._downsample_factor
            cy = depth_intrinsics.ppy / self._downsample_factor
            
            # 픽셀 좌표 그리드 생성
            xx, yy = np.meshgrid(np.arange(w_ds), np.arange(h_ds))
            
            # 깊이 값을 미터 단위로 변환 (RealSense는 일반적으로 mm 단위)
            z = depth_downsampled.astype(np.float32) / 1000.0
            
            # 유효한 깊이 값만 선택 (0이 아니고 최대 깊이 이내)
            valid_mask = (z > 0) & (z < self._max_depth_m)
            
            # 3D 좌표 계산
            x = (xx - cx) * z / fx
            y = (yy - cy) * z / fy
            
            # 유효한 포인트만 추출
            points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=-1)
            
            # 컬러 값 추출 (BGR -> RGB 변환)
            colors_bgr = color_downsampled[valid_mask]
            colors_rgb = cv2.cvtColor(colors_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
            
            # 왜곡 보정이 필요한 경우 적용
            if calibration.depth_intrinsics.coeffs and any(c != 0 for c in calibration.depth_intrinsics.coeffs):
                points = self._undistort_points(points, depth_intrinsics)
                
            return points, colors_rgb
            
        except Exception as e:
            logger.error(f"Failed to generate pointcloud: {e}", exc_info=True)
            return None
            
    def _undistort_points(self, points: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
        """왜곡된 3D 포인트를 보정합니다."""
        if points is None or points.size == 0:
            return points

        coeffs = list(intrinsics.coeffs or [])
        if not any(coeffs):
            return points

        # OpenCV undistortPoints는 최소 4개 계수를 기대하므로 5개까지 패딩
        while len(coeffs) < 5:
            coeffs.append(0.0)

        fx = float(intrinsics.fx)
        fy = float(intrinsics.fy)
        cx = float(intrinsics.ppx)
        cy = float(intrinsics.ppy)

        z_values = points[:, 2]
        valid_mask = z_values > 0
        if not np.any(valid_mask):
            return points

        # 현재 3D 포인트를 다시 픽셀 좌표(u, v)로 투영
        uv = np.zeros((valid_mask.sum(), 1, 2), dtype=np.float32)
        valid_points = points[valid_mask]
        normalized_x = valid_points[:, 0] / valid_points[:, 2]
        normalized_y = valid_points[:, 1] / valid_points[:, 2]
        uv[:, 0, 0] = normalized_x * fx + cx
        uv[:, 0, 1] = normalized_y * fy + cy

        camera_matrix = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        dist_coeffs = np.array(coeffs, dtype=np.float32)

        try:
            undistorted_uv = cv2.undistortPoints(
                uv,
                camera_matrix,
                dist_coeffs,
                P=camera_matrix,
            )
        except cv2.error:
            # 실패하면 원본 포인트를 그대로 반환
            return points

        # 보정된 픽셀 좌표를 다시 3D로 역투영
        corrected_points = points.copy()
        corrected_u = undistorted_uv[:, 0, 0]
        corrected_v = undistorted_uv[:, 0, 1]
        corrected_points[valid_mask, 0] = (corrected_u - cx) * z_values[valid_mask] / fx
        corrected_points[valid_mask, 1] = (corrected_v - cy) * z_values[valid_mask] / fy

        return corrected_points
