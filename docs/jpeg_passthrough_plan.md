# JPEG 스트림 패스스루 계획

## 1. 목표
- 컬러/뎁스 카메라가 이미 JPEG 포맷으로 전송할 때, 서버에서 재인코딩 없이 그대로 WebSocket·REST로 전달한다.
- 아루코/포인트클라우드 등 RAW 이미지가 필요한 경로는 그대로 유지한다.
- CPU 부하와 레이턴시를 줄이고, JPEG 모드 설정(`COLOR_STREAM_MODE=jpeg`, `DEPTH_STREAM_MODE=jpeg`)일 때만 동작하도록 한다.

## 2. 현재 파이프라인 요약
1. `CameraService`  
   - `ws://<카메라>/ws/color_jpeg`, `/ws/depth_jpeg`에서 바이트를 수신.  
   - `store.images.update_*_jpeg()` 저장 후 `EventType.COLOR_JPEG_RECEIVED` / `DEPTH_JPEG_RECEIVED` 발행.
2. `ImageService`  
   - JPEG 이벤트를 받아 `cv2.imdecode`로 RAW를 다시 만든 뒤 `EventType.COLOR_IMAGE_RECEIVED` / `DEPTH_IMAGE_RECEIVED` 재발행.
   - RAW 이벤트 처리 루프에서 다시 `cv2.imencode`를 수행해 WebSocket/REST용 JPEG을 만든다.
3. WebSocket/REST  
   - `/ws/color_jpg`, `/ws/depth_jpg`, `/api/images/color.jpg` 등은 재인코딩된 JPEG 데이터를 사용한다.

=> JPEG 모드에서도 “디코딩 → 재인코딩”이 매 프레임 발생하여 CPU와 지연이 커진다.

## 3. 변경 설계
### 3.1 JPEG 경로 패스스루
- `ImageService.start()`  
  - `settings.COLOR_STREAM_MODE == "jpeg"`일 때 RAW 이벤트에 대한 JPEG 변환 루프(`_process_image_loop("color_raw")`)를 시작하지 않는다.
  - 대신 `handle_color_jpeg()`가 `store.images.update_color_jpeg()`와 `EventType.WS_COLOR_IMAGE_UPDATE` 발행까지 맡는다. (현재는 JPEG → RAW → 다시 JPEG으로 변환하는 구조)
  - Depth 스트림도 동일하게 처리.

### 3.2 RAW 경로 유지
- `handle_color_jpeg()`가 JPEG을 받아 `cv2.imdecode`로 RAW 이미지를 만든 뒤 `EventType.COLOR_IMAGE_RECEIVED`를 발행하는 부분은 그대로 둔다.  
  (ArUco·Pointcloud·FrameSync는 이 RAW 이벤트를 계속 사용)

### 3.3 WebSocket/REST
- `store.images.get_color_jpeg()` / `get_depth_jpeg()`는 이미 “카메라에서 받은 JPEG”를 갖고 있으므로 추가 수정 없이 그대로 전달된다.
- StreamingService는 `WS_COLOR_IMAGE_UPDATE` / `WS_DEPTH_IMAGE_UPDATE` 이벤트를 받아 WebSocket 구독자에게 브로드캐스트하므로, 이벤트 발행 위치만 바꿔주면 된다.

## 4. 구현 순서
1. `ImageService.start()` 조건 분기 추가  
   - JPEG 모드일 때 `_process_image_loop("color_raw")` 재기동 로직 생략.
2. `handle_color_jpeg()` / `handle_depth_jpeg()` 수정  
   - JPEG을 Store에 저장 → WS 이벤트 발행.  
   - RAW 재생성·이벤트 발행 로직은 유지.
3. 필요 시 `CameraService` 로깅 강화  
   - JPEG 모드에서 RAW 경로가 비활성화됐는지 확인 가능한 로그 추가(선택).
4. 테스트  
   - JPEG 모드에서 `/ws/color_jpg`, `/api/images/color.jpg` 응답 확인.  
   - `/api/store/events/status`로 RAW 이벤트·JPEG 이벤트가 모두 발행되는지 확인.  
   - CPU 사용률/프레임 지연 비교.

## 5. 예상 효과
- 재인코딩 제거로 CPU 부하 감소, 레이턴시 약화.
- JPEG 모드와 RAW 모드를 명확히 분리하여 필요 없는 변환을 줄임.
- ArUco/Pointcloud 등 다른 파이프라인은 영향을 받지 않는다.

