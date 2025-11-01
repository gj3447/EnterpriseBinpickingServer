import asyncio
from pathlib import Path
from typing import Optional

from app.stores.application_store import ApplicationStore
from app.core.logging import logger


class RobotService:
    """
    로봇 URDF 파일을 로드하고 관리하는 서비스입니다.
    """
    def __init__(self, store: ApplicationStore, urdf_path: Optional[str] = None):
        self.store = store
        self.urdf_path = urdf_path
        self._is_running = False
        logger.info(f"RobotService initialized with URDF path: {urdf_path}")

    async def start(self):
        """서비스를 시작하고 URDF 파일을 로드합니다."""
        if self._is_running:
            logger.warning("RobotService is already running")
            return
        
        logger.info("Starting RobotService...")
        self._is_running = True
        
        if self.urdf_path:
            logger.info(f"Attempting to load URDF from: {self.urdf_path}")
            success = await self.load_urdf(self.urdf_path)
            if success:
                logger.info("RobotService started successfully with URDF loaded")
            else:
                logger.error("RobotService started but URDF loading failed")
        else:
            logger.warning("No URDF path provided. Robot service started without loading URDF.")

    async def stop(self):
        """서비스를 중지합니다."""
        if not self._is_running:
            return
        
        self._is_running = False
        logger.info("RobotService stopped.")

    async def load_urdf(self, file_path: str) -> bool:
        """Pinocchio를 사용하여 URDF 파일을 로드합니다."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"URDF file not found: {file_path}")
                return False
            
            logger.info(f"Loading URDF with Pinocchio: {file_path}")
            try:
                # CPU 바운드(URDF 파싱)를 워커 스레드로 오프로딩
                robot_name, robot_object = await asyncio.to_thread(
                    self._parse_urdf_with_pinocchio, path
                )
            except ImportError as e:
                logger.error(f"❌ Pinocchio not available: {e}")
                logger.error("Please install Pinocchio in your environment")
                return False
            except Exception as e:
                logger.error(f"❌ Error parsing URDF with Pinocchio: {e}", exc_info=True)
                return False

            # Store에 저장 (이벤트 루프 스레드에서 수행)
            self.store.robot.set_urdf_object(robot_object, robot_name, str(path))

            # 저장 확인
            stored_object = self.store.robot.get_urdf_object()
            if stored_object is not None:
                logger.info(f"✅ Pinocchio model successfully stored in ApplicationStore")
                logger.info(f"   Stored object type: {type(stored_object).__name__}")
                return True
            else:
                logger.error("❌ Failed to store Pinocchio model in ApplicationStore")
                return False
                
        except Exception as e:
            logger.error(f"❌ Unexpected error loading URDF file: {e}", exc_info=True)
            return False

    def _parse_urdf_with_pinocchio(self, path: Path):
        """워커 스레드에서 실행되어 URDF를 파싱하고 로봇 객체를 생성합니다."""
        import pinocchio as pin
        logger.info("Pinocchio library found - starting URDF parsing...")

        model = pin.buildModelFromUrdf(str(path))
        data = model.createData()

        robot_name = model.name if model.name else path.stem

        logger.info(f"✅ Pinocchio model loaded successfully!")
        logger.info(f"   Robot name: {robot_name}")
        logger.info(f"   DOF (degrees of freedom): {model.nq}")
        logger.info(f"   Joints: {model.njoints}")
        logger.info(f"   Bodies: {model.nbodies}")

        robot_object = {
            'model': model,
            'data': data,
            'urdf_path': str(path),
            'library': 'pinocchio',
            'robot_name': robot_name,
            'dof': model.nq,
            'joint_names': [model.names[i] for i in range(1, model.njoints)],  # 0번은 'universe'
            'joint_limits': {
                'lower': model.lowerPositionLimit.tolist(),
                'upper': model.upperPositionLimit.tolist(),
                'velocity': model.velocityLimit.tolist()
            }
        }

        return robot_name, robot_object

    def get_robot_object(self):
        """저장된 로봇 객체를 반환합니다."""
        return self.store.robot.get_urdf_object()

    def get_status(self) -> dict:
        """서비스 상태를 반환합니다."""
        return {
            "is_running": self._is_running,
            "robot_status": self.store.robot.get_status()
        }
