# WebSocket Stream Performance Plan

## 문제 요약
- `StreamingService`가 WebSocket 브로드캐스트를 직접 `await`하면서 `EventBus.publish()` 흐름을 차단.
- `ConnectionManager.broadcast_*`가 구독자 전송 전체 동안 `_lock`을 보유해 느린 클라이언트 한 명이 전체 스트림을 지연.
- 느린/끊어진 클라이언트에 대한 전송 타임아웃이 없어 실패 시까지 대기.

## 개선 목표
1. EventBus → StreamingService 경로를 논블로킹화.
2. ConnectionManager 락을 최소화하고 전송을 병렬화.
3. 지연 클라이언트를 빠르게 분리해 스트림 안정성 확보.

## 구현 단계

### 1. ConnectionManager 락 범위 축소
- `broadcast_bytes` / `broadcast_text`에서 `_lock`은 구독자 목록 복사까지만 보유.
- 복사본을 사용해 락 밖에서 `asyncio.gather` 실행.
- `_handle_failed_connection`만 락을 사용하도록 유지.

### 2. StreamingService 비동기 브로드캐스트
- WS 이벤트 핸들러에서 `asyncio.create_task(self.manager.broadcast_...)`로 오프로딩.
- 태스크 예외 로깅용 헬퍼 추가 (`_fire_and_forget`).
- 필요 시 TaskGroup/세마포어로 동시 실행 수 제한.

### 3. 전송 타임아웃 및 실패 처리
- `_send_*_safely`에 `asyncio.wait_for`로 per-client 타임아웃(예: 250ms) 적용.
- 타임아웃/예외 발생 시 즉시 `_handle_failed_connection`.
- 지연 클라이언트 로깅으로 디버깅 용이성 확보.

### 4. 모니터링 & 회귀 방지
- 스트림별 퍼블리시 시간/클라이언트 수를 `logger.debug` 혹은 metrics로 기록.
- 간단한 부하 테스트 스크립트(추후)로 회귀 체크.

## 예상 효과
- EventBus가 프레임당 수 ms 내로 반환되어 ArUco/IK/Pointcloud 파이프라인이 역류 없이 동작.
- 느린 구독자가 있더라도 다른 구독자 전달은 즉시 처리.
- 타임아웃 기반의 자동 분리로 WebSocket 자원 누수 방지.

