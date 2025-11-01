from typing import Any, Dict

from app.core.logging import logger
from app.stores.handlers.device_handler import DeviceHandler
from app.stores.handlers.calibration_handler import CalibrationHandler
from app.stores.handlers.camera_handler import CameraHandler
from app.stores.handlers.image_handler import ImageHandler
from app.stores.handlers.aruco_handler import ArucoHandler
from app.stores.handlers.event_handler import EventHandler
from app.stores.handlers.robot_handler import RobotHandler
from app.stores.handlers.pointcloud_handler import PointcloudHandler

class ApplicationStore:
    """
    The main store for the application. It acts as a container for various
    state handlers, each responsible for a specific domain of application state.
    """
    def __init__(self):
        # Initialize all handlers. Each handler now manages its own state
        # and internal locking, promoting better separation of concerns.
        self.device = DeviceHandler()
        self.calibration = CalibrationHandler()
        self.camera_raw = CameraHandler() # Handles raw BGR/Z16 images
        self.images = ImageHandler()      # Handles processed JPEG/transformed images
        self.aruco = ArucoHandler()
        self.events = EventHandler()
        self.robot = RobotHandler()       # Handles robot URDF data
        self.pointcloud = PointcloudHandler()  # Handles 3D pointcloud data
        logger.info("ApplicationStore initialized with all handlers.")

    def get_status(self) -> Dict[str, Any]:
        """
        Aggregates status from all handlers to provide a comprehensive
        snapshot of the entire application state.
        """
        return {
            "device_status": self.device.get_full_status(),
            "raw_image_streams": self.camera_raw.get_status(),
            "processed_image_streams": self.images.get_all_images_status(),
            "calibration_data": self.calibration.get_data_with_timestamp(),
            "aruco_status": self.aruco.get_board_status(),
            "event_timestamps": self.events.get_status(),
            "event_fps_summary": self.events.get_all_fps(),
            "robot_status": self.robot.get_status(),
            "pointcloud_status": self.pointcloud.get_statistics()
        }
