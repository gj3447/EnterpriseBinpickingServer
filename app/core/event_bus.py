import asyncio
import time
from typing import Any, Dict, List, Callable, Awaitable, Optional

from collections import defaultdict
from contextlib import asynccontextmanager
from loguru import logger

# EventHandler를 임포트하고, 순환 참조를 피하기 위해 TYPE_CHECKING을 사용
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.stores.handlers.event_handler import EventHandler

class EventBus:
    """
    애플리케이션 전체에서 사용될 비동기 이벤트 버스입니다.
    이제 이벤트 발행 시간을 EventHandler에 기록하는 기능이 추가되었습니다.
    """
    def __init__(self, event_handler: Optional['EventHandler'] = None):
        self._subscribers: Dict[str, List[Callable[[str, Any], Awaitable[None]]]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._event_handler = event_handler
        self._metrics = {
            "total_events_published": 0,
            "total_callbacks_executed": 0,
            "total_errors": 0,
            "start_time": time.time()
        }

    def set_event_handler(self, event_handler: 'EventHandler'):
        """의존성 주입을 위해 EventHandler를 설정합니다."""
        self._event_handler = event_handler

    async def subscribe(self, event_name: str, callback: Callable[[Any, Any], Awaitable[None]]):
        if not asyncio.iscoroutinefunction(callback):
            raise ValueError(f"콜백 함수는 async 함수여야 합니다: {callback}")
        
        async with self._lock:
            self._subscribers[event_name].append(callback)
            logger.debug(f"이벤트 '{event_name}'에 구독자 추가됨. 총 구독자 수: {len(self._subscribers[event_name])}")

    async def unsubscribe(self, event_name: str, callback: Callable[[Any, Any], Awaitable[None]]):
        async with self._lock:
            if event_name in self._subscribers:
                try:
                    self._subscribers[event_name].remove(callback)
                    logger.debug(f"이벤트 '{event_name}'에서 구독자 제거됨. 남은 구독자 수: {len(self._subscribers[event_name])}")
                except ValueError:
                    logger.warning(f"구독자 제거 실패: 이벤트 '{event_name}'에 해당 콜백이 없습니다")

    async def publish(self, event_name: str, data: Any = None):
        """
        이벤트를 발행하고, EventHandler에 타임스탬프를 기록합니다.
        """
        if self._event_handler:
            self._event_handler.update_event_timestamp(event_name)

        async with self._lock:
            subscribers = self._subscribers.get(event_name, [])
        
        if not subscribers:
            logger.trace(f"이벤트 '{event_name}'에 구독자가 없습니다")
            return
        
        logger.debug(f"이벤트 '{event_name}' 발행 중... 구독자 수: {len(subscribers)}")
        self._metrics["total_events_published"] += 1
        
        tasks = [self._execute_callback(callback, event_name, data) for callback in subscribers]
        if tasks:
            await asyncio.gather(*tasks)

    async def _execute_callback(self, callback: Callable[[str, Any], Awaitable[None]], event_name: str, data: Any):
        try:
            await callback(event_name, data)
            self._metrics["total_callbacks_executed"] += 1
        except Exception as e:
            logger.error(f"콜백 실행 중 오류 발생 (이벤트: {event_name}): {e}", exc_info=True)
            self._metrics["total_errors"] += 1
