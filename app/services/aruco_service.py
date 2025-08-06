import asyncio
import cv2
import json
from pathlib import Path
import numpy as np
from typing import Optional, List, Set, Tuple, Dict, Any
from scipy.spatial.transform import Rotation as R

from app.stores.application_store import ApplicationStore
from app.schemas.aruco import Pose, DetectedMarker, Corner, DetectedMarkerPose
from app.schemas.events import SyncFrameReadyPayload, ArucoUpdatePayload, SystemTransformsUpdatePayload
from app.schemas.transforms import SystemTransformSnapshot, SystemTransformSnapshotResponse
from app.core.config import settings
from app.core.event_bus import EventBus
from app.core.logging import logger
from app.core.event_type import EventType

class ArucoService:
    def __init__(self, store: ApplicationStore, event_bus: EventBus, board_config_path: str, robot_pose: Optional[Pose]):
        self.store = store
        self.event_bus = event_bus
        self._is_running = False
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.board_marker_ids: Set[int] = set()
        self.board = self._create_board_from_config(board_config_path)
        if robot_pose:
            self.store.aruco.set_robot_pose(robot_pose)
        self.identity_pose = Pose(translation=[0,0,0], orientation_quaternion=[0,0,0,1])

  
            
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
            # 페이로드에서 직접 이미지 데이터를 사용하므로 Store 조회가 불필요
            await self.detect_and_update_store(payload.color_image_data, payload.timestamp)
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

    async def detect_and_update_store(self, color_image: np.ndarray, timestamp: float):
        board_pose, board_markers, external_markers = None, [], []
        calib_data = self.store.calibration.get_data()
        if calib_data and self.board is not None:
            try:
                intrinsics = calib_data.color_intrinsics
                cam_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]])
                dist_coeffs = np.array(intrinsics.coeffs)
                corners, ids, _ = cv2.aruco.detectMarkers(color_image, self.aruco_dict, parameters=self.aruco_params)
                if ids is not None and len(ids) > 0:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, settings.ARUCO_SINGLE_MARKER_SIZE_M, cam_matrix, dist_coeffs)
                    all_markers = self._convert_to_marker_dtos(ids, corners, rvecs, tvecs)
                    for m in all_markers: (board_markers if m.id in self.board_marker_ids else external_markers).append(m)
                    retval, board_rvec, board_tvec = cv2.aruco.estimatePoseBoard(corners, ids, self.board, cam_matrix, dist_coeffs, None, None)
                    if retval > 0: board_pose = self._convert_to_pose_dto(board_rvec, board_tvec)
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
