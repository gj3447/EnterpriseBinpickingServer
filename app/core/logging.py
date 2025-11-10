import sys
from loguru import logger

from app.core.config import settings

# --- 로거 설정 ---
# 기존 로거를 모두 제거하고 새로운 설정을 적용합니다.
logger.remove()

# 콘솔(터미널) 출력을 위한 로거를 추가합니다.
# format: 로그 메시지의 형식을 지정합니다.
# level: 이 로거가 처리할 최소 로그 레벨을 지정합니다 (INFO 이상).
# colorize: 터미널에서 로그 레벨에 따라 색상을 입힙니다.
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL.upper(),
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
    colorize=True,
)

# 파일 출력을 위한 로거를 추가할 수도 있습니다 (예시).
# logger.add("logs/app.log", rotation="10 MB", level="DEBUG")

# 다른 모듈에서 'from app.core.logging import logger'로 가져가서 사용할 수 있도록 합니다.
__all__ = ["logger"]
