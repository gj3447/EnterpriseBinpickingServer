import asyncio
import cv2
import json
from pathlib import Path
import numpy as np
from typing import Optional, List, Set, Tuple, Dict, Any
from scipy.spatial.transform import Rotation as R

from app.stores.application_store import ApplicationStore
from app.schemas.aruco import Pose, DetectedMarker, Corner, DetectedMarkerPose
from app.schemas.aruco_config import ArucoConfig
from app.schemas.events import SyncFrameReadyPayload, ArucoUpdatePayload, SystemTransformsUpdatePayload
from app.schemas.transforms import SystemTransformSnapshot, SystemTransformSnapshotResponse
from app.core.config import settings
from app.core.event_bus import EventBus
from app.core.logging import logger
from app.core.event_type import EventType

class PoseTemporalFilter:
    """포즈의 시간적 안정화를 위한 필터"""
    def __init__(self, alpha: float = 0.8):
        """
        Args:
            alpha: 필터 강도 (0-1). 1에 가까울수록 이전 값에 더 의존
        """
        self.alpha = alpha
        self.prev_translation: Optional[np.ndarray] = None
        self.prev_quaternion: Optional[np.ndarray] = None
        
    def filter(self, pose: Pose) -> Pose:
        """새로운 포즈를 필터링하여 부드럽게 만듭니다."""
        translation = np.array(pose.translation)
        quaternion = np.array(pose.orientation_quaternion)
        
        if self.prev_translation is None or self.prev_quaternion is None:
            # 첫 프레임은 그대로 사용
            self.prev_translation = translation
            self.prev_quaternion = quaternion
            return pose
            
        # Translation: Exponential Moving Average
        filtered_translation = (
            self.alpha * self.prev_translation + 
            (1 - self.alpha) * translation
        )
        
        # Rotation: Quaternion SLERP (Spherical Linear Interpolation)
        filtered_quaternion = self._slerp_quaternion(
            self.prev_quaternion, 
            quaternion, 
            1 - self.alpha
        )
        
        # 업데이트
        self.prev_translation = filtered_translation
        self.prev_quaternion = filtered_quaternion
        
        return Pose(
            translation=filtered_translation.tolist(),
            orientation_quaternion=filtered_quaternion.tolist()
        )
    
    def _slerp_quaternion(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """두 quaternion 사이의 구면 선형 보간"""
        # 내적으로 각도 계산
        dot = np.dot(q1, q2)
        
        # 가장 가까운 경로 선택 (dot < 0이면 반대 방향)
        if dot < 0:
            q2 = -q2
            dot = -dot
            
        # 거의 같은 방향이면 선형 보간
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
            
        # SLERP 수행
        theta = np.arccos(np.clip(dot, -1, 1))
        sin_theta = np.sin(theta)
        
        # 0으로 나누기 방지
        if sin_theta < 0.001:
            return q1
            
        a = np.sin((1 - t) * theta) / sin_theta
        b = np.sin(t * theta) / sin_theta
        
        result = a * q1 + b * q2
        return result / np.linalg.norm(result)
    
    def reset(self):
        """필터 상태를 초기화합니다."""
        self.prev_translation = None
        self.prev_quaternion = None

class ArucoService:
    def __init__(self, store: ApplicationStore, event_bus: EventBus, board_config_path: str, 
                 robot_pose: Optional[Pose], config: Optional[ArucoConfig] = None):
        self.store = store
        self.event_bus = event_bus
        self._is_running = False
        self.config = config or self._get_default_config()
        self._frames_skipped_total = 0
        
        # ArUco 딕셔너리 설정
        dict_attr = getattr(cv2.aruco, self.config.dictionary, cv2.aruco.DICT_4X4_250)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # 코너 정밀화 설정
        if self.config.pose_estimation.corner_refinement:
            try:
                refine_method = getattr(cv2.aruco, self.config.pose_estimation.corner_refinement)
                self.aruco_params.cornerRefinementMethod = refine_method
            except Exception:
                pass
                
        # 최신 OpenCV 4.7+ API 사용
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.board_marker_ids: Set[int] = set()
        self.board = self._create_board_from_config(board_config_path)
        
        # _create_board_from_config에서 board_marker_ids가 채워진 후 Store에 저장
        self.store.aruco.set_board_marker_ids(self.board_marker_ids)
        logger.info(f"Stored {len(self.board_marker_ids)} board marker IDs in store: {sorted(self.board_marker_ids)}")
        
        if robot_pose:
            self.store.aruco.set_robot_pose(robot_pose)
        self.identity_pose = Pose(translation=[0,0,0], orientation_quaternion=[0,0,0,1])

        # 최신 프레임만 처리하기 위한 coalesce 버퍼와 처리 상태
        self._latest_sync: Optional[Tuple[np.ndarray, float]] = None
        self._sync_processing: bool = False
        
        # 시간적 필터링을 위한 포즈 필터 (설정에 따라 초기화)
        if self.config.temporal_filter.enabled:
            self._board_pose_filter = PoseTemporalFilter(alpha=self.config.temporal_filter.board_alpha)
            self._marker_pose_filters: Dict[int, PoseTemporalFilter] = {}
            self._marker_last_seen: Dict[int, float] = {}
            self._filter_timeout = self.config.temporal_filter.timeout_seconds
        else:
            self._board_pose_filter = None
            self._marker_pose_filters = None
            self._marker_last_seen = None
            self._filter_timeout = None

      
    def _get_default_config(self) -> ArucoConfig:
        """기본 ArUco 설정을 반환합니다."""
        return ArucoConfig(
            dictionary="DICT_4X4_250",
            marker_size_m=0.05,
            board_config_file="aruco_place.csv",
            temporal_filter={
                "enabled": True,
                "board_alpha": 0.8,
                "marker_alpha": 0.7,
                "timeout_seconds": 1.0
            },
            depth_sampling={
                "window_size_small": 3,
                "window_size_large": 5,
                "small_marker_threshold_m": 0.05,
                "quality_check": {
                    "enabled": True,
                    "max_relative_std": 0.05
                }
            },
            ransac={
                "enabled": True,
                "inlier_threshold_m": 0.01,
                "inlier_threshold_small_m": 0.005,
                "max_iterations": 100,
                "min_inlier_ratio": 0.6
            },
            pose_estimation={
                "min_points_for_board": 6,
                "use_ippe_for_markers": True,
                "corner_refinement": "CORNER_REFINE_SUBPIX"
            },
            debug={
                "log_metrics": False,
                "log_filter_resets": True
            }
        )
            
    async def start(self):
        """서비스를 시작하고 이벤트 구독을 등록합니다."""
        if self._is_running:
            return
        self._is_running = True
        await self.event_bus.subscribe(EventType.SYNC_FRAME_READY.value, self.handle_sync_frame)
        logger.info("ArucoService started and subscribed to SYNC_FRAME_READY events.")

    async def stop(self):
        """서비스를 중지하고 이벤트 구독을 해제합니다."""
        if not self._is_running:
            return
        self._is_running = False
        await self.event_bus.unsubscribe(EventType.SYNC_FRAME_READY.value, self.handle_sync_frame)
        logger.info("ArucoService stopped and unsubscribed from events.")

    async def handle_sync_frame(self, event_name: str, payload: SyncFrameReadyPayload):
        """SYNC_FRAME_READY 이벤트를 처리하는 콜백 함수입니다."""
        try:
            # 이전 프레임 정보가 있고 처리 중이면 스킵 카운트 증가
            if self._latest_sync is not None and hasattr(self, '_frames_skipped_total'):
                self._frames_skipped_total += 1
            
            # 최신 페이로드만 유지하고 처리 루프가 없으면 시작
            self._latest_sync = (payload.color_image_data, payload.depth_image_data, payload.timestamp)
            if not self._sync_processing:
                self._sync_processing = True
                asyncio.create_task(self._process_sync_loop())
        except Exception as e:
            logger.error(f"Error processing sync frame for Aruco detection: {e}", exc_info=True)
                
    def _create_board_from_config(self, path: str) -> Optional[cv2.aruco.Board]:
        try:
            obj_points, ids_list = [], []
            with open(path, 'r') as f:
                next(f); [self.board_marker_ids.add(int(parts[0])) or obj_points.append(np.array([[float(p) for p in parts[1:4]],[float(p) for p in parts[4:7]],[float(p) for p in parts[7:10]],[float(p) for p in parts[10:13]]], dtype=np.float32)) or ids_list.append([int(parts[0])]) for parts in (line.strip().split(',') for line in f)]
            board = cv2.aruco.Board(np.array(obj_points), self.aruco_dict, np.array(ids_list))
            logger.info(f"Successfully created ArUco board with {len(self.board_marker_ids)} markers from {path}")
            return board
        except Exception as e:
            logger.error(f"Failed to create ArUco board from {path}: {e}"); return None

    def _estimate_pose_single_markers(self, corners, marker_size, cam_matrix, dist_coeffs):
        """OpenCV 4.7+용 개별 마커 포즈 추정 (estimatePoseSingleMarkers 대체)"""
        rvecs = []
        tvecs = []
        
        # 마커 3D 좌표 정의 (정사각형, Z=0 평면)
        half_size = marker_size / 2.0
        object_points = np.array([
            [-half_size, half_size, 0],
            [half_size, half_size, 0], 
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)
        
        for corner in corners:
            success, rvec, tvec = cv2.solvePnP(
                object_points, 
                corner[0],  # corner는 (1, 4, 2) 형태
                cam_matrix, 
                dist_coeffs
            )
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
            else:
                # 실패 시 빈 벡터 추가 (기존 API와 호환)
                rvecs.append(np.zeros((3, 1), dtype=np.float32))
                tvecs.append(np.zeros((3, 1), dtype=np.float32))
        
        return np.array(rvecs), np.array(tvecs)

    def _estimate_pose_board(self, corners, ids, cam_matrix, dist_coeffs):
        """OpenCV 4.7+용 보드 포즈 추정 (estimatePoseBoard 대체)"""
        if self.board is None or len(corners) == 0:
            return None, None
        
        try:
            # 보드의 객체 포인트와 이미지 포인트 매칭
            obj_points = []
            img_points = []
            
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.board_marker_ids:
                    # 보드에서 해당 마커의 3D 좌표 찾기
                    marker_idx = list(self.board.getIds().flatten()).index(marker_id)
                    marker_obj_points = self.board.getObjPoints()[marker_idx]
                    
                    obj_points.extend(marker_obj_points)
                    img_points.extend(corners[i][0])
            
            if len(obj_points) >= 4:  # 최소 4개 포인트 필요
                obj_points = np.array(obj_points, dtype=np.float32)
                img_points = np.array(img_points, dtype=np.float32)
                
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, 
                    img_points, 
                    cam_matrix, 
                    dist_coeffs
                )
                
                if success:
                    return rvec, tvec
            
            return None, None
            
        except Exception as e:
            logger.error(f"Error in board pose estimation: {e}")
            return None, None

    async def detect_and_update_store(self, color_image: np.ndarray, depth_image: np.ndarray, timestamp: float):
        board_pose, board_markers, external_markers = None, [], []
        calib_data = self.store.calibration.get_data()
        if calib_data and self.board is not None:
            try:
                intrinsics = calib_data.color_intrinsics
                cam_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]])
                dist_coeffs = np.array(intrinsics.coeffs)

                # CPU 바운드(검출/포즈 추정)를 워커 스레드로 오프로딩
                board_pose, board_markers, external_markers = await asyncio.to_thread(
                    self._detect_markers_and_board_pose,
                    color_image,
                    depth_image,
                    cam_matrix,
                    dist_coeffs,
                    intrinsics.fx,
                    intrinsics.fy,
                    intrinsics.ppx,
                    intrinsics.ppy,
                    timestamp,
                )
            except Exception as e:
                logger.error(f"An error occurred during ArUco detection: {e}")
        self.store.aruco.update_detection_results(board_markers, external_markers, board_pose)

        # 분석 완료 후, 실시간 스트리밍 서비스를 위해 이벤트 발행
        aruco_payload = ArucoUpdatePayload(
            timestamp=timestamp,
            source_image_data=color_image,
            board_pose=board_pose,
            board_markers=board_markers,
            external_markers=external_markers
        )
        
        # 모든 좌표계에 대한 변환 스냅샷을 계산하여 함께 발행
        frames = ["camera", "board", "robot"]
        internal_snapshots = [self.get_transform_snapshot(frame) for frame in frames]
        
        # 외부용 응답 모델로 수동 변환 (코너 정보 제거)
        response_snapshots = []
        for snapshot in internal_snapshots:
            if snapshot:
                response_snapshots.append(
                    SystemTransformSnapshotResponse(
                        frame=snapshot.frame,
                        board_detected=snapshot.board_detected,
                        board=snapshot.board,
                        robot=snapshot.robot,
                        camera=snapshot.camera,
                        external_markers=[
                            DetectedMarkerPose(id=m.id, pose=m.pose) for m in snapshot.external_markers
                        ]
                    )
                )

        transform_payload = SystemTransformsUpdatePayload(
            timestamp=timestamp,
            snapshots=response_snapshots
        )

        publish_tasks = [
            self.event_bus.publish(EventType.ARUCO_UPDATE.value, aruco_payload),
            self.event_bus.publish(EventType.SYSTEM_TRANSFORMS_UPDATE.value, transform_payload)
        ]
        await asyncio.gather(*publish_tasks)

    async def _process_sync_loop(self):
        """최신 동기 프레임만 순차 처리하는 루프."""
        frames_processed = 0
        frames_skipped = 0
        last_log_time = asyncio.get_event_loop().time()
        
        try:
            while True:
                snapshot = self._latest_sync
                # 읽은 즉시 비워서 다음 주기 동안 최신값만 남도록 함
                self._latest_sync = None
                if not snapshot:
                    break
                    
                # 성능 로깅 (1초마다)
                current_time = asyncio.get_event_loop().time()
                if current_time - last_log_time >= 1.0:
                    if self.config.debug.log_metrics:
                        total_frames = frames_processed + self._frames_skipped_total
                        skip_rate = (self._frames_skipped_total / total_frames * 100) if total_frames > 0 else 0
                        logger.info(f"ArUco Performance - FPS: {frames_processed:.1f}, Skipped: {self._frames_skipped_total} ({skip_rate:.1f}%), Total Input: {total_frames}")
                    frames_processed = 0
                    self._frames_skipped_total = 0
                    last_log_time = current_time
                
                color_image, depth_image, timestamp = snapshot
                try:
                    # 처리 시간 측정
                    process_start = asyncio.get_event_loop().time()
                    await self.detect_and_update_store(color_image, depth_image, timestamp)
                    process_time = asyncio.get_event_loop().time() - process_start
                    
                    frames_processed += 1
                    
                    # 처리 시간이 너무 길면 경고
                    if self.config.debug.log_metrics and process_time > 0.1:  # 100ms 이상
                        logger.warning(f"ArUco processing took {process_time*1000:.1f}ms (target: 33ms for 30fps)")
                except Exception as e:
                    logger.error(f"Error in Aruco processing loop: {e}", exc_info=True)
        finally:
            self._sync_processing = False
            # 종료 직후 새 프레임이 들어온 경우 재기동
            if self._latest_sync is not None and not self._sync_processing:
                self._sync_processing = True
                asyncio.create_task(self._process_sync_loop())

    def _detect_markers_and_board_pose(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        cam_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        timestamp: float = None,
    ) -> Tuple[Optional[Pose], List[DetectedMarker], List[DetectedMarker]]:
        """동기 컨텍스트에서 실행되는 검출/포즈 추정 루틴. to_thread로 호출됩니다."""
        board_pose: Optional[Pose] = None
        board_markers: List[DetectedMarker] = []
        external_markers: List[DetectedMarker] = []

        # 스레드 안전을 위해 로컬 detector 사용 + 그레이스케일로 검출 안정화
        local_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        try:
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = color_image
        corners, ids, _ = local_detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            # 깊이 단위 정규화(Z16-mm → m 대응)
            depth_m = self._depth_to_meters(depth_image)

            # 개별 마커 포즈: 깊이 기반 3D-3D 정합 우선, 부족 시 PnP(IPPE) 폴백
            est_rvecs: List[np.ndarray] = []
            est_tvecs: List[np.ndarray] = []
            for i, marker_id in enumerate(ids.flatten()):
                corner_px = corners[i].reshape(4, 2)
                # 깊이 기반 추정 시도
                rvec, tvec = self._estimate_marker_pose_with_depth(
                    corner_px, depth_m, fx, fy, cx, cy, self.config.marker_size_m
                )
                if rvec is None or tvec is None:
                    # PnP(IPPE) 폴백
                    rvec, tvec = self._estimate_pose_single_marker_ippe(
                        corner_px, cam_matrix, dist_coeffs, self.config.marker_size_m
                    )
                est_rvecs.append(rvec if rvec is not None else np.zeros((3, 1), dtype=np.float32))
                est_tvecs.append(tvec if tvec is not None else np.zeros((3, 1), dtype=np.float32))

            rvecs = np.array(est_rvecs)
            tvecs = np.array(est_tvecs)

            all_markers = self._convert_to_marker_dtos(ids, corners, rvecs, tvecs)
            
            # 각 마커에 시간적 필터링 적용 (설정에 따라)
            filtered_markers = []
            current_time = timestamp if timestamp else 0
            detected_ids = set()
            
            for marker in all_markers:
                detected_ids.add(marker.id)
                
                # 마커별 필터 생성 또는 가져오기
                if self.config.temporal_filter.enabled and marker.id not in self._marker_pose_filters:
                    self._marker_pose_filters[marker.id] = PoseTemporalFilter(alpha=self.config.temporal_filter.marker_alpha)
                
                # 타임아웃 체크: 오래 안 보인 마커는 필터 리셋
                if self.config.temporal_filter.enabled and marker.id in self._marker_last_seen:
                    time_since_last = current_time - self._marker_last_seen[marker.id]
                    if time_since_last > self._filter_timeout:
                        self._marker_pose_filters[marker.id].reset()
                        if self.config.debug.log_filter_resets:
                            logger.debug(f"Reset filter for marker {marker.id} (timeout: {time_since_last:.2f}s)")
                
                # 필터링된 포즈 적용
                if self.config.temporal_filter.enabled and self._marker_pose_filters:
                    filtered_pose = self._marker_pose_filters[marker.id].filter(marker.pose)
                    filtered_marker = DetectedMarker(
                        id=marker.id,
                        pose=filtered_pose,
                        corners=marker.corners
                    )
                    # 마지막 감지 시간 업데이트
                    self._marker_last_seen[marker.id] = current_time
                else:
                    filtered_marker = marker
                    
                filtered_markers.append(filtered_marker)
            
            # 보드가 감지되지 않으면 보드 필터 리셋
            if self.config.temporal_filter.enabled and self._board_pose_filter:
                board_detected = any(mid in self.board_marker_ids for mid in detected_ids)
                if not board_detected and hasattr(self, '_last_board_detection_time'):
                    time_since_board = current_time - self._last_board_detection_time
                    if time_since_board > self._filter_timeout:
                        self._board_pose_filter.reset()
                        if self.config.debug.log_filter_resets:
                            logger.debug(f"Reset board pose filter (timeout: {time_since_board:.2f}s)")
                elif board_detected:
                    self._last_board_detection_time = current_time
            
            # 필터링된 마커들을 보드/외부로 분류
            for marker in filtered_markers:
                (board_markers if marker.id in self.board_marker_ids else external_markers).append(marker)

            # 보드 포즈: 하이브리드 모드 또는 기존 방식
            if self.config.pose_estimation.hybrid_mode.enabled:
                board_rvec, board_tvec = self._estimate_board_pose_hybrid(
                    ids, corners, depth_m, cam_matrix, dist_coeffs, fx, fy, cx, cy
                )
            else:
                # 기존 방식: 깊이 기반 3D-3D 정합 우선, 부족 시 PnP 폴백
                board_rvec, board_tvec = self._estimate_board_pose_with_depth(
                    ids, corners, depth_m, fx, fy, cx, cy
                )
                if board_rvec is None or board_tvec is None:
                    board_rvec, board_tvec = self._estimate_pose_board(corners, ids, cam_matrix, dist_coeffs)
                    
            if board_rvec is not None and board_tvec is not None:
                raw_board_pose = self._convert_to_pose_dto(board_rvec, board_tvec)
                # 시간적 필터링 적용 (설정에 따라)
                if self.config.temporal_filter.enabled and self._board_pose_filter:
                    board_pose = self._board_pose_filter.filter(raw_board_pose)
                else:
                    board_pose = raw_board_pose

        return board_pose, board_markers, external_markers

    def get_board_pose(self, frame: str) -> Optional[Pose]:
        # ... (이전과 동일)
        board_status = self.store.aruco.get_board_status()
        if not board_status.detected or not board_status.pose: return None
        if frame == "camera": return board_status.pose
        elif frame == "board": return self.identity_pose
        elif frame == "robot":
            pose_robot_cam = self.get_robot_pose("camera")
            if not pose_robot_cam: return None
            T_robot_cam = self._pose_to_transform(pose_robot_cam)
            T_cam_board = self._pose_to_transform(board_status.pose)
            T_robot_board = np.dot(T_robot_cam, T_cam_board)
            return self._transform_to_pose(T_robot_board)
        return None

    def get_robot_pose(self, frame: str) -> Optional[Pose]:
        robot_pose_board = self.store.aruco.get_robot_pose_on_board()
        if not robot_pose_board: return None
        if frame == "board": return robot_pose_board
        elif frame == "camera":
            board_status = self.store.aruco.get_board_status()
            if not board_status.detected or not board_status.pose: return None
            T_cam_board = self._pose_to_transform(board_status.pose)
            T_board_robot = self._pose_to_transform(robot_pose_board)
            T_cam_robot = np.dot(T_cam_board, T_board_robot)
            return self._transform_to_pose(T_cam_robot)
        elif frame == "robot":
            # 로봇 좌표계에서는 로봇이 원점이므로 identity pose 반환
            return self.identity_pose
        return None

    def get_camera_pose(self, frame: str) -> Optional[Pose]:
        if frame == "camera":
            # 카메라 좌표계에서는 카메라가 원점이므로 identity pose 반환
            return self.identity_pose
        elif frame == "board":
            board_status = self.store.aruco.get_board_status()
            if not board_status.detected or not board_status.pose: return None
            T_cam_board = self._pose_to_transform(board_status.pose)
            T_board_cam = np.linalg.inv(T_cam_board)
            return self._transform_to_pose(T_board_cam)
        elif frame == "robot":
            robot_pose_board = self.store.aruco.get_robot_pose_on_board()
            if not robot_pose_board: return None
            board_status = self.store.aruco.get_board_status()
            if not board_status.detected or not board_status.pose: return None
            T_cam_board = self._pose_to_transform(board_status.pose)
            T_board_robot = self._pose_to_transform(robot_pose_board)
            T_cam_robot = np.dot(T_cam_board, T_board_robot)
            T_robot_cam = np.linalg.inv(T_cam_robot)
            return self._transform_to_pose(T_robot_cam)
        return None

    def get_external_markers_pose(self, frame: str) -> List[DetectedMarker]:
        """외부 마커 목록을 지정된 좌표계 기준으로 변환하여 반환합니다."""
        external_markers = self.store.aruco.get_external_markers()
        if not external_markers: return []

        if frame == "camera":
            return external_markers

        # 변환에 필요한 상위 Pose 정보 가져오기
        board_pose_cam = self.store.aruco.get_board_status().pose
        robot_pose_board = self.store.aruco.get_robot_pose_on_board()
        if not board_pose_cam or not robot_pose_board: return []

        T_cam_board = self._pose_to_transform(board_pose_cam)

        if frame == "board":
            T_target_cam = np.linalg.inv(T_cam_board)
        elif frame == "robot":
            T_board_robot = self._pose_to_transform(robot_pose_board)
            T_cam_robot = np.dot(T_cam_board, T_board_robot)
            T_target_cam = np.linalg.inv(T_cam_robot)
        else:
            return []

        transformed_markers = []
        for marker in external_markers:
            T_cam_marker = self._pose_to_transform(marker.pose)
            T_target_marker = np.dot(T_target_cam, T_cam_marker)
            new_pose = self._transform_to_pose(T_target_marker)
            transformed_markers.append(DetectedMarker(id=marker.id, pose=new_pose, corners=marker.corners))
        
        return transformed_markers

    def get_transform_snapshot(self, frame: str) -> Optional[SystemTransformSnapshot]:
        """지정된 좌표계를 기준으로 시스템 전체의 변환 정보를 담은 스냅샷을 생성합니다."""
        # --- 1. 기본 데이터 로드 ---
        board_status = self.store.aruco.get_board_status()
        robot_pose_board = self.store.aruco.get_robot_pose_on_board()
        external_markers_cam = self.store.aruco.get_external_markers()

        if not board_status.detected or not board_status.pose or not robot_pose_board:
            logger.warning("Cannot calculate transform snapshot due to missing board or robot pose.")
            return SystemTransformSnapshot(frame=frame, board_detected=False)

        # --- 2. 기본 변환 행렬 계산 ---
        T_cam_board = self._pose_to_transform(board_status.pose)
        T_board_robot = self._pose_to_transform(robot_pose_board)
        T_cam_robot = np.dot(T_cam_board, T_board_robot)

        # --- 3. 목표 좌표계에 따른 변환 행렬 결정 ---
        T_target_cam = np.eye(4) # 기본값: camera frame
        if frame == "board":
            T_target_cam = np.linalg.inv(T_cam_board)
        elif frame == "robot":
            T_target_cam = np.linalg.inv(T_cam_robot)
        
        # --- 4. 각 객체의 Pose 계산 ---
        board_pose = self._transform_to_pose(np.dot(T_target_cam, T_cam_board))
        robot_pose = self._transform_to_pose(np.dot(T_target_cam, T_cam_robot))
        camera_pose = self._transform_to_pose(T_target_cam)
        
        transformed_markers = []
        for marker in external_markers_cam:
            T_cam_marker = self._pose_to_transform(marker.pose)
            T_target_marker = np.dot(T_target_cam, T_cam_marker)
            new_pose = self._transform_to_pose(T_target_marker)
            transformed_markers.append(DetectedMarker(id=marker.id, pose=new_pose, corners=marker.corners))

        return SystemTransformSnapshot(
            frame=frame,
            board_detected=True,
            board=board_pose,
            robot=robot_pose,
            camera=camera_pose,
            external_markers=transformed_markers
        )
        
    # ... (헬퍼 메서드들은 이전과 동일)
    def _pose_to_transform(self, pose: Pose) -> np.ndarray:
        r = R.from_quat(pose.orientation_quaternion); t = np.array(pose.translation)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = r.as_matrix()
        transform_matrix[:3, 3] = t
        return transform_matrix

    def _transform_to_pose(self, transform_matrix: np.ndarray) -> Pose:
        r = R.from_matrix(transform_matrix[:3, :3]); t = transform_matrix[:3, 3]
        return Pose(translation=t.tolist(), orientation_quaternion=r.as_quat().tolist())

    def _convert_to_pose_dto(self, rvec: np.ndarray, tvec: np.ndarray) -> Pose:
        rotation = R.from_rotvec(rvec.flatten()); quat = rotation.as_quat()
        return Pose(translation=tvec.flatten().tolist(), orientation_quaternion=quat.tolist())

    def _convert_to_marker_dtos(self, ids: np.ndarray, corners: List, rvecs: np.ndarray, tvecs: np.ndarray) -> List[DetectedMarker]:
        markers = []
        for i, marker_id in enumerate(ids.flatten()):
            pose_dto = self._convert_to_pose_dto(rvecs[i], tvecs[i])
            corner_dtos = [Corner(x=c[0], y=c[1]) for c in corners[i].reshape(4, 2)]
            markers.append(DetectedMarker(id=int(marker_id), pose=pose_dto, corners=corner_dtos))
        return markers

    # ===================== 깊이·정합 기반 보조 메서드 =====================
    def _depth_to_meters(self, depth_image: np.ndarray) -> np.ndarray:
        """깊이 배열을 미터 단위 float32로 정규화합니다."""
        depth = depth_image.astype(np.float32)
        # 값 범위를 보고 mm → m 자동 판별(경험적)
        if np.nanmax(depth) > 50.0:
            depth = depth / 1000.0
        return depth

    def _sample_depth_median(self, depth_m: np.ndarray, u: float, v: float, window: int = 3) -> Optional[float]:
        h, w = depth_m.shape[:2]
        half = window // 2
        u_i = int(round(u)); v_i = int(round(v))
        u0 = max(0, u_i - half); u1 = min(w, u_i + half + 1)
        v0 = max(0, v_i - half); v1 = min(h, v_i + half + 1)
        patch = depth_m[v0:v1, u0:u1]
        if patch.size == 0:
            return None
        vals = patch[np.isfinite(patch) & (patch > 0)]
        if vals.size == 0:
            return None
        
        # 깊이 품질 검증: 표준편차가 너무 크면 신뢰할 수 없음
        if self.config.depth_sampling.quality_check.enabled and vals.size > 1:
            depth_std = np.std(vals)
            depth_median = np.median(vals)
            # 상대 표준편차가 설정값 이상이면 노이즈가 심한 것으로 판단
            if depth_std / depth_median > self.config.depth_sampling.quality_check.max_relative_std:
                return None
        
        return float(np.median(vals))

    def _backproject(self, u: float, v: float, z_m: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
        x = (u - cx) * z_m / fx
        y = (v - cy) * z_m / fy
        return np.array([x, y, z_m], dtype=np.float32)
    
    def _sample_marker_interior_depth(self, depth_m: np.ndarray, corners_px: np.ndarray, offset_ratio: float = 0.2) -> Optional[List[float]]:
        """마커 내부 영역에서 깊이를 샘플링하여 엣지 노이즈를 회피합니다."""
        # 마커 중심 계산
        center = np.mean(corners_px, axis=0)
        
        # 각 코너를 중심 방향으로 offset_ratio만큼 이동
        interior_points = []
        for corner in corners_px:
            direction = center - corner
            interior_point = corner + direction * offset_ratio
            interior_points.append(interior_point)
        
        # 내부 포인트들의 깊이 샘플링
        depths = []
        window_size = (self.config.depth_sampling.window_size_small 
                       if self.config.marker_size_m <= self.config.depth_sampling.small_marker_threshold_m 
                       else self.config.depth_sampling.window_size_large)
                       
        for pt in interior_points:
            z = self._sample_depth_median(depth_m, pt[0], pt[1], window=window_size)
            if z is not None and z > 0:
                depths.append(z)
        
        # 최소 유효 샘플 개수 체크
        if len(depths) < self.config.depth_sampling.min_valid_samples:
            return None
            
        # 깊이 일관성 체크 (옵션)
        if self.config.depth_sampling.quality_check.enabled and len(depths) > 1:
            depth_std = np.std(depths)
            depth_mean = np.mean(depths)
            if depth_std / depth_mean > self.config.depth_sampling.quality_check.max_relative_std:
                return None
                
        return depths

    def _estimate_marker_pose_with_depth(
        self,
        corner_px: np.ndarray,
        depth_m: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        marker_size_m: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """깊이 기반 3D-3D 정합으로 단일 마커 포즈 추정. 실패 시 (None, None)."""
        # 마커 내부 영역에서 깊이 샘플링 (설정에 따라)
        if self.config.depth_sampling.use_interior_sampling:
            interior_depths = self._sample_marker_interior_depth(depth_m, corner_px, 
                                                                offset_ratio=self.config.depth_sampling.interior_offset_ratio)
        else:
            interior_depths = None
        if interior_depths is None:
            return None, None
            
        # 내부 포인트들의 평균 깊이 사용
        avg_depth = np.mean(interior_depths)
        
        # 코너들을 평균 깊이로 백프로젝션
        observed_points: List[np.ndarray] = []
        for (u, v) in corner_px:
            observed_points.append(self._backproject(float(u), float(v), avg_depth, fx, fy, cx, cy))
            
        if len(observed_points) < 4:
            return None, None

        # 마커 로컬 3D 모델 코너(Z=0)
        half = marker_size_m / 2.0
        model_points = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float32)

        obs = np.stack(observed_points, axis=0).astype(np.float32)
        R_cm, t_c = self._rigid_transform_3d(model_points, obs)
        if R_cm is None or t_c is None:
            return None, None
        rvec, _ = cv2.Rodrigues(R_cm.astype(np.float32))
        tvec = t_c.reshape(3, 1).astype(np.float32)
        return rvec, tvec

    def _estimate_pose_single_marker_ippe(
        self,
        corner_px: np.ndarray,
        cam_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        marker_size_m: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """IPPE 기반 PnP로 단일 마커 포즈를 추정합니다."""
        half_size = marker_size_m / 2.0
        object_points = np.array([
            [-half_size,  half_size, 0.0],
            [ half_size,  half_size, 0.0], 
            [ half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0]
        ], dtype=np.float32)
        image_points = corner_px.astype(np.float32)
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                cam_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if success:
                return rvec, tvec
        except Exception:
            pass
        return None, None

    def _estimate_board_pose_with_depth(
        self,
        ids: np.ndarray,
        corners: List[np.ndarray],
        depth_m: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """깊이 기반 3D-3D 정합으로 보드 포즈를 추정합니다. 실패 시 (None, None)."""
        if self.board is None:
            return None, None

        model_points_all: List[np.ndarray] = []
        observed_points_all: List[np.ndarray] = []
        try:
            board_ids = list(self.board.getIds().flatten())
            board_obj_points = self.board.getObjPoints()
        except Exception:
            return None, None

        for i, marker_id in enumerate(ids.flatten()):
            if int(marker_id) not in self.board_marker_ids:
                continue
            try:
                marker_idx = board_ids.index(int(marker_id))
            except ValueError:
                continue
            model_pts_marker = np.array(board_obj_points[marker_idx], dtype=np.float32).reshape(4, 3)
            corner_px = corners[i].reshape(4, 2)

            # 마커 내부 영역에서 깊이 샘플링 (설정에 따라)
            if self.config.depth_sampling.use_interior_sampling:
                interior_depths = self._sample_marker_interior_depth(depth_m, corner_px, 
                                                                    offset_ratio=self.config.depth_sampling.interior_offset_ratio)
            else:
                interior_depths = None
            if interior_depths is None or len(interior_depths) < self.config.depth_sampling.min_valid_samples:
                # 내부 샘플링 실패 시 기존 방식으로 폴백
                for k in range(4):
                    u, v = float(corner_px[k, 0]), float(corner_px[k, 1])
                    window_size = (self.config.depth_sampling.window_size_small 
                                   if self.config.marker_size_m <= self.config.depth_sampling.small_marker_threshold_m 
                                   else self.config.depth_sampling.window_size_large)
                    z = self._sample_depth_median(depth_m, u, v, window=window_size)
                    if z is None or z <= 0:
                        continue
                    observed = self._backproject(u, v, z, fx, fy, cx, cy)
                    model_points_all.append(model_pts_marker[k])
                    observed_points_all.append(observed)
            else:
                # 내부 포인트들의 평균 깊이로 모든 코너 백프로젝션
                avg_depth = np.mean(interior_depths)
                for k in range(4):
                    u, v = float(corner_px[k, 0]), float(corner_px[k, 1])
                    observed = self._backproject(u, v, avg_depth, fx, fy, cx, cy)
                    model_points_all.append(model_pts_marker[k])
                    observed_points_all.append(observed)

        # 최소 포인트 검사 (안정적인 6DOF 추정을 위해)
        if len(observed_points_all) < self.config.pose_estimation.min_points_for_board:
            return None, None

        A = np.stack(model_points_all, axis=0).astype(np.float32)
        B = np.stack(observed_points_all, axis=0).astype(np.float32)
        
        # RANSAC 기반 outlier 제거 정합 (설정에 따라)
        if self.config.ransac.enabled:
            R_cb, t_c = self._rigid_transform_3d_ransac(A, B)
        else:
            R_cb, t_c = self._rigid_transform_3d(A, B)
        if R_cb is None or t_c is None:
            return None, None
        rvec, _ = cv2.Rodrigues(R_cb.astype(np.float32))
        tvec = t_c.reshape(3, 1).astype(np.float32)
        return rvec, tvec

    def _estimate_board_pose_hybrid(
        self,
        ids: np.ndarray,
        corners: List[np.ndarray],
        depth_m: np.ndarray,
        cam_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """PnP와 깊이 정보를 결합한 하이브리드 보드 포즈 추정"""
        
        # 1단계: PnP로 초기 포즈 추정
        rvec_pnp, tvec_pnp = self._estimate_pose_board(corners, ids, cam_matrix, dist_coeffs)
        if rvec_pnp is None or tvec_pnp is None:
            # PnP 실패 시 깊이 기반으로 폴백
            return self._estimate_board_pose_with_depth(ids, corners, depth_m, fx, fy, cx, cy)
        
        # 2단계: 깊이 정보 수집 및 검증
        depth_samples = []
        board_marker_indices = []
        
        for i, marker_id in enumerate(ids.flatten()):
            if int(marker_id) in self.board_marker_ids:
                corner_px = corners[i].reshape(4, 2)
                # 마커 내부 영역에서 깊이 샘플링
                if self.config.depth_sampling.use_interior_sampling:
                    interior_depths = self._sample_marker_interior_depth(
                        depth_m, corner_px, 
                        offset_ratio=self.config.depth_sampling.interior_offset_ratio
                    )
                    if interior_depths and len(interior_depths) >= self.config.depth_sampling.min_valid_samples:
                        depth_samples.extend(interior_depths)
                        board_marker_indices.append(i)
        
        # 깊이 샘플이 충분하지 않으면 PnP 결과 그대로 반환
        if len(depth_samples) < self.config.pose_estimation.min_points_for_board:
            return rvec_pnp, tvec_pnp
        
        # 3단계: 깊이 일관성 검증
        depth_array = np.array(depth_samples)
        depth_mean = np.mean(depth_array)
        depth_std = np.std(depth_array)
        depth_consistency = 1.0 - (depth_std / depth_mean) if depth_mean > 0 else 0
        
        if depth_consistency < self.config.pose_estimation.hybrid_mode.min_depth_consistency:
            # 깊이 일관성이 낮으면 PnP 결과 사용
            return rvec_pnp, tvec_pnp
        
        # 4단계: PnP 포즈에서 예상되는 깊이 계산
        # 보드 중심점의 예상 깊이
        predicted_z = float(tvec_pnp[2])
        
        # 5단계: 스케일 보정
        scale_correction = depth_mean / predicted_z if predicted_z > 0 else 1.0
        
        # 스케일 보정이 임계값 이상일 때만 적용
        if abs(scale_correction - 1.0) > self.config.pose_estimation.hybrid_mode.scale_correction_threshold:
            if self.config.pose_estimation.hybrid_mode.use_weighted_average:
                # 가중 평균 사용
                weight = self.config.pose_estimation.hybrid_mode.depth_weight
                final_scale = weight * scale_correction + (1 - weight) * 1.0
            else:
                final_scale = scale_correction
                
            tvec_hybrid = tvec_pnp * final_scale
            
            if self.config.debug.log_metrics:
                logger.debug(f"Hybrid pose estimation: "
                           f"depth_consistency={depth_consistency:.3f}, "
                           f"scale_correction={scale_correction:.3f}, "
                           f"final_scale={final_scale:.3f}")
            
            return rvec_pnp, tvec_hybrid
        
        # 스케일 변화가 작으면 PnP 결과 그대로 사용
        return rvec_pnp, tvec_pnp

    def _rigid_transform_3d(self, A: np.ndarray, B: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """스케일=1 고정의 3D-3D 강체 정합(Umeyama/Horn)으로 R(3x3), t(3,)를 반환합니다."""
        try:
            assert A.shape == B.shape and A.shape[0] >= 3
            centroid_A = np.mean(A, axis=0)
            centroid_B = np.mean(B, axis=0)
            AA = A - centroid_A
            BB = B - centroid_B
            H = AA.T @ BB
            U, S, Vt = np.linalg.svd(H)
            R_mat = Vt.T @ U.T
            if np.linalg.det(R_mat) < 0:
                Vt[2, :] *= -1
                R_mat = Vt.T @ U.T
            t_vec = centroid_B - R_mat @ centroid_A
            return R_mat.astype(np.float32), t_vec.astype(np.float32)
        except Exception:
            return None, None

    def _rigid_transform_3d_ransac(self, A: np.ndarray, B: np.ndarray, 
                                   inlier_threshold: float = None,
                                   max_iterations: int = None,
                                   min_inlier_ratio: float = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """RANSAC 기반 outlier 제거 후 3D-3D 강체 정합"""
        n_points = len(A)
        if n_points < 3:
            return None, None
            
        # 기본값 설정
        if inlier_threshold is None:
            inlier_threshold = self.config.ransac.inlier_threshold_m
        if max_iterations is None:
            max_iterations = self.config.ransac.max_iterations
        if min_inlier_ratio is None:
            min_inlier_ratio = self.config.ransac.min_inlier_ratio
            
        best_inliers = []
        best_R, best_t = None, None
        best_inlier_count = 0
        
        # 마커 크기에 따라 threshold 조정
        if self.config.marker_size_m <= self.config.depth_sampling.small_marker_threshold_m:
            inlier_threshold = self.config.ransac.inlier_threshold_small_m
        else:
            inlier_threshold = self.config.ransac.inlier_threshold_m
        
        for _ in range(max_iterations):
            # 랜덤하게 최소 3점 선택
            sample_idx = np.random.choice(n_points, min(3, n_points), replace=False)
            
            # 3점으로 변환 계산
            R, t = self._rigid_transform_3d(A[sample_idx], B[sample_idx])
            if R is None or t is None:
                continue
                
            # 모든 점에 대해 변환 후 오차 계산
            A_transformed = (R @ A.T).T + t
            errors = np.linalg.norm(A_transformed - B, axis=1)
            
            # Inlier 판별
            inliers = errors < inlier_threshold
            inlier_count = np.sum(inliers)
            
            # 더 나은 모델 발견 시 업데이트
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_R = R
                best_t = t
                
                # 충분한 inlier가 있으면 조기 종료
                if inlier_count / n_points > 0.9:
                    break
        
        # 최소 inlier 비율 확인
        if best_inlier_count < n_points * min_inlier_ratio:
            return None, None
            
        # 모든 inlier로 최종 정합
        if best_inlier_count > 3:
            final_R, final_t = self._rigid_transform_3d(A[best_inliers], B[best_inliers])
            if final_R is not None and final_t is not None:
                return final_R, final_t
                
        return best_R, best_t
