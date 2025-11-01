import asyncio
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from app.schemas.camera import CameraCalibration
from app.schemas.aruco import Pose, DetectedMarker
from app.schemas.events import (
    ColorImageReceivedPayload, DepthImageReceivedPayload, ArucoUpdatePayload,
    ColorJpegReceivedPayload, DepthJpegReceivedPayload,
    WsColorImageUpdatePayload, WsDepthImageUpdatePayload,
    WsDebugImageUpdatePayload, WsPerspectiveImageUpdatePayload
)
from app.core.config import settings
from app.core.event_bus import EventBus
from app.core.logging import logger
from app.core.event_type import EventType
from app.stores.application_store import ApplicationStore

class ImageService:
    """
    이미지 처리 유틸리티와 JPEG 변환 및 저장을 담당하는 서비스입니다.
    """
    def __init__(self, store: ApplicationStore, event_bus: EventBus):
        """ImageService를 초기화합니다."""
        self.store = store
        self.event_bus = event_bus
        self._is_running = False
        # 최신 프레임만 처리하기 위한 간단한 coalesce 버퍼와 처리 상태
        self._latest_images: Dict[str, Optional[Tuple[np.ndarray, float]]] = {"color_raw": None, "depth_raw": None}
        self._processing_streams: set[str] = set()

    async def start(self):
        """서비스를 시작하고 이벤트 구독을 등록합니다."""
        if self._is_running:
            return
        self._is_running = True
        
        # Raw 이미지 이벤트 구독
        await self.event_bus.subscribe(EventType.COLOR_IMAGE_RECEIVED.value, self.handle_color_image)
        await self.event_bus.subscribe(EventType.DEPTH_IMAGE_RECEIVED.value, self.handle_depth_image)
        
        # JPEG 이미지 이벤트 구독 (설정에 따라)
        if settings.COLOR_STREAM_MODE == "jpeg":
            await self.event_bus.subscribe(EventType.COLOR_JPEG_RECEIVED.value, self.handle_color_jpeg)
        if settings.DEPTH_STREAM_MODE == "jpeg":
            await self.event_bus.subscribe(EventType.DEPTH_JPEG_RECEIVED.value, self.handle_depth_jpeg)
            
        await self.event_bus.subscribe(EventType.ARUCO_UPDATE.value, self.handle_aruco_update)
        logger.info(f"ImageService started - Color: {settings.COLOR_STREAM_MODE}, Depth: {settings.DEPTH_STREAM_MODE}")

    async def stop(self):
        """서비스를 중지하고 이벤트 구독을 해제합니다."""
        if not self._is_running:
            return
        self._is_running = False
        
        # Raw 이미지 이벤트 구독 해제
        await self.event_bus.unsubscribe(EventType.COLOR_IMAGE_RECEIVED.value, self.handle_color_image)
        await self.event_bus.unsubscribe(EventType.DEPTH_IMAGE_RECEIVED.value, self.handle_depth_image)
        
        # JPEG 이미지 이벤트 구독 해제 (설정에 따라)
        if settings.COLOR_STREAM_MODE == "jpeg":
            await self.event_bus.unsubscribe(EventType.COLOR_JPEG_RECEIVED.value, self.handle_color_jpeg)
        if settings.DEPTH_STREAM_MODE == "jpeg":
            await self.event_bus.unsubscribe(EventType.DEPTH_JPEG_RECEIVED.value, self.handle_depth_jpeg)
            
        await self.event_bus.unsubscribe(EventType.ARUCO_UPDATE.value, self.handle_aruco_update)
        logger.info("ImageService stopped and unsubscribed from events.")

    async def handle_color_image(self, event_name: str, payload: ColorImageReceivedPayload):
        """COLOR_IMAGE_RECEIVED 이벤트를 처리하여 JPEG으로 변환합니다."""
        try:
            # 최신 프레임만 유지하고 처리 루프가 없으면 시작
            self._latest_images["color_raw"] = (payload.image_data, payload.timestamp)
            if "color_raw" not in self._processing_streams:
                self._processing_streams.add("color_raw")
                asyncio.create_task(self._process_image_loop("color_raw"))
        except Exception as e:
            logger.error(f"Error in handle_color_image: {e}", exc_info=True)

    async def handle_depth_image(self, event_name: str, payload: DepthImageReceivedPayload):
        """DEPTH_IMAGE_RECEIVED 이벤트를 처리하여 JPEG으로 변환합니다."""
        try:
            # 최신 프레임만 유지하고 처리 루프가 없으면 시작
            self._latest_images["depth_raw"] = (payload.image_data, payload.timestamp)
            if "depth_raw" not in self._processing_streams:
                self._processing_streams.add("depth_raw")
                asyncio.create_task(self._process_image_loop("depth_raw"))
        except Exception as e:
            logger.error(f"Error in handle_depth_image: {e}", exc_info=True)
            
    async def _process_and_publish_jpeg(self, image_data: np.ndarray, timestamp: float, stream_id: str):
        # CPU 바운드(JPEG 인코딩)를 이벤트 루프 밖으로 오프로딩
        jpeg_data = await asyncio.to_thread(self.get_image_as_jpeg, image_data, stream_id)
        if not jpeg_data:
            return

        if stream_id == "color_raw":
            self.store.images.update_color_jpeg(jpeg_data, timestamp)
            payload = WsColorImageUpdatePayload(timestamp=timestamp, jpeg_data=jpeg_data)
            await self.event_bus.publish(EventType.WS_COLOR_IMAGE_UPDATE.value, payload)
        
        elif stream_id == "depth_raw":
            self.store.images.update_depth_jpeg(jpeg_data, timestamp)
            payload = WsDepthImageUpdatePayload(timestamp=timestamp, jpeg_data=jpeg_data)
            await self.event_bus.publish(EventType.WS_DEPTH_IMAGE_UPDATE.value, payload)
        
        logger.debug(f"Successfully converted and stored {stream_id} as JPEG.")

    async def _process_image_loop(self, stream_id: str):
        """각 스트림별 최신 프레임만 처리하는 루프."""
        try:
            while True:
                snapshot = self._latest_images.get(stream_id)
                # 읽은 즉시 비워서 다음 주기 동안 최신값만 남도록 함
                self._latest_images[stream_id] = None
                if not snapshot:
                    break
                image_data, timestamp = snapshot
                try:
                    await self._process_and_publish_jpeg(image_data, timestamp, stream_id)
                except Exception as e:
                    logger.error(f"Error while processing {stream_id} frame: {e}", exc_info=True)
        finally:
            self._processing_streams.discard(stream_id)
            # 종료 직후 새 프레임이 들어온 경우 재기동
            if self._latest_images.get(stream_id) is not None and stream_id not in self._processing_streams:
                self._processing_streams.add(stream_id)
                asyncio.create_task(self._process_image_loop(stream_id))
        
    async def handle_color_jpeg(self, event_name: str, payload: ColorJpegReceivedPayload):
        """COLOR_JPEG_RECEIVED 이벤트를 처리하여 BGR로 디코딩합니다."""
        try:
            # JPEG를 BGR로 디코딩 (CPU 바운드 작업이므로 스레드로 오프로딩)
            bgr_image = await asyncio.to_thread(self._decode_jpeg_to_bgr, payload.jpeg_data)
            if bgr_image is not None:
                # Store에 raw 이미지 저장
                self.store.camera_raw.update_color_image(bgr_image, payload.timestamp)
                # Raw 이미지 이벤트 발행 (ArUco 등이 처리할 수 있도록)
                raw_payload = ColorImageReceivedPayload(timestamp=payload.timestamp, image_data=bgr_image)
                await self.event_bus.publish(EventType.COLOR_IMAGE_RECEIVED.value, raw_payload)
        except Exception as e:
            logger.error(f"Error in handle_color_jpeg: {e}", exc_info=True)
    
    async def handle_depth_jpeg(self, event_name: str, payload: DepthJpegReceivedPayload):
        """DEPTH_JPEG_RECEIVED 이벤트를 처리하여 Z16으로 디코딩합니다."""
        try:
            # JPEG를 Z16으로 디코딩 (CPU 바운드 작업이므로 스레드로 오프로딩)
            z16_image = await asyncio.to_thread(self._decode_jpeg_to_z16, payload.jpeg_data)
            if z16_image is not None:
                # Store에 raw 이미지 저장
                self.store.camera_raw.update_depth_image(z16_image, payload.timestamp)
                # Raw 이미지 이벤트 발행 (ArUco 등이 처리할 수 있도록)
                raw_payload = DepthImageReceivedPayload(timestamp=payload.timestamp, image_data=z16_image)
                await self.event_bus.publish(EventType.DEPTH_IMAGE_RECEIVED.value, raw_payload)
        except Exception as e:
            logger.error(f"Error in handle_depth_jpeg: {e}", exc_info=True)
    
    def _decode_jpeg_to_bgr(self, jpeg_data: bytes) -> Optional[np.ndarray]:
        """JPEG 바이트를 BGR numpy 배열로 디코딩합니다."""
        try:
            # JPEG 바이트를 numpy 배열로 변환
            nparr = np.frombuffer(jpeg_data, np.uint8)
            # OpenCV로 디코딩 (BGR 형식)
            bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return bgr_image
        except Exception as e:
            logger.error(f"Failed to decode JPEG to BGR: {e}")
            return None
    
    def _decode_jpeg_to_z16(self, jpeg_data: bytes) -> Optional[np.ndarray]:
        """JPEG 바이트를 Z16 (uint16) numpy 배열로 디코딩합니다."""
        try:
            # JPEG 바이트를 numpy 배열로 변환
            nparr = np.frombuffer(jpeg_data, np.uint8)
            # OpenCV로 디코딩 (16비트 깊이)
            depth_image = cv2.imdecode(nparr, cv2.IMREAD_ANYDEPTH)
            # uint16으로 변환 (Z16 형식)
            if depth_image.dtype != np.uint16:
                depth_image = depth_image.astype(np.uint16)
            return depth_image
        except Exception as e:
            logger.error(f"Failed to decode JPEG to Z16: {e}")
            return None

    async def handle_aruco_update(self, event_name: str, payload: ArucoUpdatePayload):
        """ARUCO_UPDATE 이벤트를 처리하여 디버그 및 원근 보정 이미지를 생성하고 저장합니다."""
        try:
            calibration = self.store.calibration.get_data()
            image = payload.source_image_data

            if calibration is None:
                logger.warning("Missing calibration data for generating images from aruco update.")
                return

            # 1. ArUco 디버그 이미지 생성 및 발행 준비 (CPU 바운드 오프로딩)
            # 스토어 접근은 이벤트 루프 스레드에서 수행 후 값만 워커 스레드로 전달
            robot_pose_board = self.store.aruco.get_robot_pose_on_board()
            debug_image_jpeg = await asyncio.to_thread(
                self.get_aruco_debug_image_as_jpeg, image, payload, calibration, robot_pose_board
            )
            if debug_image_jpeg:
                self.store.images.update_aruco_debug_image(debug_image_jpeg, payload.timestamp)
                ws_payload = WsDebugImageUpdatePayload(timestamp=payload.timestamp, jpeg_data=debug_image_jpeg)
                await self.event_bus.publish(EventType.WS_DEBUG_IMAGE_UPDATE.value, ws_payload)
                logger.debug("Successfully generated and stored 'aruco_debug_jpg'.")

            # 2. ArUco 보드 기준 원근 보정 이미지 생성 및 발행 준비 (CPU 바운드 오프로딩)
            if payload.board_pose:
                perspective_jpeg = await asyncio.to_thread(self.get_board_perspective_corrected_image_as_jpeg, image, payload.board_pose, calibration)
                if perspective_jpeg:
                    self.store.images.update_board_perspective_image(perspective_jpeg, payload.timestamp)
                    ws_payload = WsPerspectiveImageUpdatePayload(timestamp=payload.timestamp, jpeg_data=perspective_jpeg)
                    await self.event_bus.publish(EventType.WS_PERSPECTIVE_IMAGE_UPDATE.value, ws_payload)
                    logger.debug("Successfully generated and stored 'board_perspective_jpg'.")

            # 3. 모든 준비된 이벤트를 한 번에 발행

        except Exception as e:
            logger.error(f"Error generating or storing images from aruco update: {e}", exc_info=True)


    def get_raw_image(self, image: np.ndarray) -> np.ndarray:
        """주어진 numpy 이미지를 아무 처리 없이 그대로 반환합니다."""
        return image

    def get_image_as_jpeg(self, image: np.ndarray, stream_id: str) -> Optional[bytes]:
        """주어진 numpy 이미지를 JPEG 형식의 바이트로 변환합니다."""
        if "depth" in stream_id:
            processed_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            processed_image = image
        success, jpeg_image = cv2.imencode(".jpg", processed_image)
        return jpeg_image.tobytes() if success else None

    def get_board_perspective_corrected_image_as_jpeg(
        self,
        image: np.ndarray,
        board_pose: Pose,
        calibration: CameraCalibration,
        width_px: int = 800
    ) -> Optional[bytes]:
        """주어진 이미지와 Pose를 사용해 원근 변환된 이미지를 생성합니다."""
        board_half_width = settings.BOARD_WIDTH_M / 2
        board_half_height = settings.BOARD_HEIGHT_M / 2
        board_corners_3d = np.array([
            [-board_half_width, -board_half_height, 0], [board_half_width, -board_half_height, 0],
            [board_half_width, board_half_height, 0], [-board_half_width, board_half_height, 0]
        ], dtype=np.float32)

        intrinsics = calibration.color_intrinsics
        cam_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]])
        dist_coeffs = np.array(intrinsics.coeffs)
        
        rvec = R.from_quat(board_pose.orientation_quaternion).as_rotvec()
        tvec = np.array(board_pose.translation)
        
        src_points, _ = cv2.projectPoints(board_corners_3d, rvec, tvec, cam_matrix, dist_coeffs)
        # getPerspectiveTransform은 (4,2) float32 입력을 기대하므로 형상 명시
        src_points = src_points.reshape(-1, 2).astype(np.float32)
        aspect_ratio = settings.BOARD_HEIGHT_M / settings.BOARD_WIDTH_M
        height_px = int(width_px * aspect_ratio)
        dst_points = np.array([[0, 0], [width_px - 1, 0], [width_px - 1, height_px - 1], [0, height_px - 1]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_image = cv2.warpPerspective(image, matrix, (width_px, height_px))
        success, jpeg_image = cv2.imencode(".jpg", warped_image)
        return jpeg_image.tobytes() if success else None

    def get_board_mask_image_as_jpeg(self, image: np.ndarray, board_pose: Pose, calibration: CameraCalibration) -> Optional[bytes]:
        # ... (로직은 get_board_perspective_corrected_image_as_jpeg와 유사)
        board_half_width = settings.BOARD_WIDTH_M / 2
        board_half_height = settings.BOARD_HEIGHT_M / 2
        board_corners_3d = np.array([
            [-board_half_width, -board_half_height, 0], [board_half_width, -board_half_height, 0],
            [board_half_width, board_half_height, 0], [-board_half_width, board_half_height, 0]
        ], dtype=np.float32)
        intrinsics = calibration.color_intrinsics
        cam_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]])
        dist_coeffs = np.array(intrinsics.coeffs)
        rvec = R.from_quat(board_pose.orientation_quaternion).as_rotvec()
        tvec = np.array(board_pose.translation)
        src_points, _ = cv2.projectPoints(board_corners_3d, rvec, tvec, cam_matrix, dist_coeffs)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(src_points)], 255)
        success, jpeg_image = cv2.imencode(".jpg", mask)
        return jpeg_image.tobytes() if success else None

    def get_single_marker_mask_image_as_jpeg(self, image: np.ndarray, marker: DetectedMarker, calibration: CameraCalibration) -> Optional[bytes]:
        marker_half_size = settings.ARUCO_SINGLE_MARKER_SIZE_M / 2
        marker_corners_3d = np.array([
            [-marker_half_size, -marker_half_size, 0], [marker_half_size, -marker_half_size, 0],
            [marker_half_size, marker_half_size, 0], [-marker_half_size, marker_half_size, 0]
        ], dtype=np.float32)
        intrinsics = calibration.color_intrinsics
        cam_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]])
        dist_coeffs = np.array(intrinsics.coeffs)
        rvec = R.from_quat(marker.pose.orientation_quaternion).as_rotvec()
        tvec = np.array(marker.pose.translation)
        corners_2d, _ = cv2.projectPoints(marker_corners_3d, rvec, tvec, cam_matrix, dist_coeffs)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(corners_2d)], 255)
        success, jpeg_image = cv2.imencode(".jpg", mask)
        return jpeg_image.tobytes() if success else None

    def get_aruco_debug_image_as_jpeg(
        self,
        image: np.ndarray,
        aruco_payload: ArucoUpdatePayload,
        calibration: CameraCalibration,
        robot_pose_board: Optional[Pose] = None,
    ) -> Optional[bytes]:
        debug_image = image.copy()
        board_pose = aruco_payload.board_pose
        
        if board_pose and calibration:
            intrinsics = calibration.color_intrinsics
            cam_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]])
            dist_coeffs = np.array(intrinsics.coeffs)
            board_tvec = np.array(board_pose.translation)
            board_rvec = R.from_quat(board_pose.orientation_quaternion).as_rotvec()

            self._draw_board_area(debug_image, board_rvec, board_tvec, cam_matrix, dist_coeffs)
            if robot_pose_board:
                self._draw_robot_position(debug_image, robot_pose_board, board_rvec, board_tvec, cam_matrix, dist_coeffs)
            
            # 외부 마커와 보드 마커를 모두 그립니다.
            all_markers = aruco_payload.board_markers + aruco_payload.external_markers
            for marker in all_markers:
                self._draw_marker(debug_image, marker)
                
            cv2.drawFrameAxes(debug_image, cam_matrix, dist_coeffs, board_rvec, board_tvec, 0.1)

        success, jpeg_image = cv2.imencode(".jpg", debug_image)
        return jpeg_image.tobytes() if success else None

    def _draw_board_area(self, image, rvec, tvec, cam_matrix, dist_coeffs):
        board_half_width = settings.BOARD_WIDTH_M / 2
        board_half_height = settings.BOARD_HEIGHT_M / 2
        board_corners_3d = np.array([
            [-board_half_width, -board_half_height, 0], [board_half_width, -board_half_height, 0],
            [board_half_width, board_half_height, 0], [-board_half_width, board_half_height, 0]
        ], dtype=np.float32)
        board_corners_2d, _ = cv2.projectPoints(board_corners_3d, rvec, tvec, cam_matrix, dist_coeffs)
        overlay = image.copy()
        cv2.fillPoly(overlay, [np.int32(board_corners_2d)], color=(255, 100, 100))
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, dst=image)

    def _draw_marker(self, image: np.ndarray, marker: DetectedMarker):
        """탐지된 마커의 corner 정보를 사용해 이미지에 직접 그립니다."""
        if not marker.corners:
            return
        
        # corner들을 int32 numpy 배열로 변환
        pts = np.array([[corner.x, corner.y] for corner in marker.corners], np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # 마커 영역을 반투명하게 채움
        overlay = image.copy()
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
        cv2.addWeighted(overlay, 0.2, image, 0.8, 0, dst=image)
        
        # 마커 ID 텍스트 추가
        text_pos = (int(pts[0][0][0]), int(pts[0][0][1]) - 10)
        cv2.putText(image, f"ID:{marker.id}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


    def _draw_robot_position(self, image, robot_pose, board_rvec, board_tvec, cam_matrix, dist_coeffs):
        robot_pos_3d = np.array(robot_pose.translation, dtype=np.float32)
        robot_pos_2d, _ = cv2.projectPoints(robot_pos_3d, board_rvec, board_tvec, cam_matrix, dist_coeffs)
        if robot_pos_2d is not None:
            pt = (int(robot_pos_2d[0][0][0]), int(robot_pos_2d[0][0][1]))
            cv2.circle(image, pt, 10, (0, 0, 255), -1)
            cv2.putText(image, "Robot", (pt[0] + 15, pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
