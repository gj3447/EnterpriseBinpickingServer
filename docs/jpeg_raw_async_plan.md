# JPEG 패스스루와 RAW 디코딩 분리 계획

## 1. 배경
- 현재 `ImageService.handle_color_jpeg` / `handle_depth_jpeg`는 한 함수에서 다음을 모두 수행한다.
  1. JPEG 바이트를 Store에 저장하고 WS 이벤트 발행
  2. `cv2.imdecode`로 RAW 이미지를 만들고, `EventType.COLOR_IMAGE_RECEIVED` / `DEPTH_IMAGE_RECEIVED`를 재발행
- `CameraService`가 JPEG 프레임을 수신할 때 `EventBus.publish(...)`를 `await` 하므로, 위 과정이 모두 끝날 때까지 **카메라 WebSocket 루프가 다음 프레임을 읽지 못한다**. RAW 변환이 늘어질 경우 컬러 스트림 전체가 지연되는 구조다.

## 2. 목표
- JPEG 패스스루(스토어 업데이트 + WS 브로드캐스트)를 즉시 끝내고, RAW 디코딩 + RAW 이벤트 발행을 별도 백그라운드 태스크로 분리한다.
- 최신 프레임만 처리하도록 coalesce 버퍼를 유지해, 백그라운드 태스크가 밀려도 오래된 프레임을 무의미하게 소모하지 않게 한다.
- ArUco/Pointcloud 파이프라인이 RAW 이벤트를 계속 받도록 동작 보장.

## 3. 설계
### 3.1 인터페이스
- `handle_color_jpeg`는 다음만 수행:
  ```python
  self.store.images.update_color_jpeg(payload.jpeg_data, payload.timestamp)
  await self.event_bus.publish(EventType.WS_COLOR_IMAGE_UPDATE.value, ws_payload)
  self._latest_color_jpeg = payload  # 코얼레싱 버퍼
  self._ensure_color_decode_task()
  ```
- RAW 변환은 새로운 async 태스크에서 처리:
  ```python
  async def _decode_color_jpeg_worker(self):
      while self._latest_color_jpeg:
          payload = self._latest_color_jpeg
          self._latest_color_jpeg = None
          bgr = await asyncio.to_thread(self._decode_jpeg_to_bgr, payload.jpeg_data)
          if bgr is None:
              continue
          self.store.camera_raw.update_color_image(bgr, payload.timestamp)
          raw_payload = ColorImageReceivedPayload(...)
          await self.event_bus.publish(EventType.COLOR_IMAGE_RECEIVED.value, raw_payload)
  ```
  - `_latest_color_jpeg`는 coalesce 용도로 마지막 프레임 하나만 유지.
  - `_color_decode_task`가 없을 때만 새 태스크를 생성하고, 종료 시 `None`으로 리셋.

### 3.2 제약 / 보호 장치
- **Semaphore**: 동시 디코딩 태스크 수를 제한(예: 1~2개)하여 스레드풀 포화 방지.
- **에러 처리**: `create_task`에 done callback을 붙여 예외 로그 기록.
- **취소 가능성**: 서버 종료 시 태스크가 깔끔히 종료되도록 `stop()`에서 태스크 cancel + await.

### 3.3 FrameSync 영향
- RAW 이벤트 발행 시점이 기존보다 늦어질 수 있으므로, FrameSync가 최신 timestamp만 사용하는지 검증.
- 필요 시 `ColorImageReceivedPayload`에 sequence ID를 추가해 디버깅/검증을 돕는다.

## 4. 구현 단계
1. `ImageService`에 `_latest_color_jpeg`, `_color_decode_task`, `_color_decode_semaphore`(선택) 추가.
2. `handle_color_jpeg` / `handle_depth_jpeg`를 패스스루 + coalesce 저장 + 태스크 기동 형태로 변경.
3. `_decode_color_jpeg_worker` / `_decode_depth_jpeg_worker` 구현.
4. 서비스 `stop()`에서 태스크를 cancel/await해 리소스 정리.
5. FrameSync/Aruco/Pointcloud 로그를 보며 RAW 이벤트 순서/타이밍을 검증.

## 5. 기대 효과
- JPEG 패스스루 경로(WS/REST)는 디코딩 지연과 독립적으로 동작 → 카메라 WebSocket 수신이 막히지 않는다.
- RAW 변환 작업이 밀려도 coalesce 버퍼 덕분에 최신 프레임만 처리하여 CPU 낭비 감소.
- 구조상 변화가 크지 않아 안정성을 유지하면서 레이턴시를 줄일 수 있다.


