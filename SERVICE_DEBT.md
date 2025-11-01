### 범위

- 대상 서비스: `camera_service.py`, `frame_sync_service.py`, `aruco_service.py`, `image_service.py`, `robot_service.py`
- 이벤트 흐름: `COLOR_IMAGE_RECEIVED`/`DEPTH_IMAGE_RECEIVED` → `SYNC_FRAME_READY` → `ARUCO_UPDATE` → `WS_*`

### P0 즉시 개선(성능/안정성)

- **메인 이벤트 루프 블로킹 위험**: OpenCV/NumPy/JPEG 연산이 async 핸들러에서 동기 실행됨(`detectMarkers`, `solvePnP`, `imencode`, `warpPerspective`). 루프 지연/프레임 드랍 유발.
  - 수정: CPU 바운드 구간을 `await asyncio.to_thread(...)`로 오프로딩. 필요 시 전용 `ThreadPoolExecutor(max_workers=N)` 도입.
- **백프레셔 부재**: 생산 속도(카메라) > 소비 속도(아루코/이미지)일 때 이벤트 폭주 가능. 최신 프레임만 유지하는 coalesce/drop-old 필요.
  - 수정: 서비스 내부 타임스탬프 기반 스로틀(최신만 처리) 또는 이벤트 버스 레벨 bounded queue + drop 정책.

### P1 성능 최적화(효율/지연)

- **ArUco 입력 그레이스케일**: 컬러로 탐지 호출 중. `cv2.cvtColor(..., COLOR_BGR2GRAY)`로 전처리하여 속도/안정성 개선.
- **정사각 플래너 타겟에 IPPE**: `solvePnP`에 `cv2.SOLVEPNP_IPPE_SQUARE` 플래그 사용으로 정확도/속도 개선.
- **보드 매핑 O(n²) 제거**: `list(...).index()` 대신 초기화 시 `id → obj_points` 딕셔너리 캐시.
- **DTO 생성 최소화**: 내부 파이프라인은 NumPy 유지, 외부 전송 직전만 DTO/JPEG로 변환해 메모리 churn 감소.
- **JPEG 인코딩 오프로딩 + 품질/크기 제어**: `asyncio.to_thread(cv2.imencode, ...)` + `IMWRITE_JPEG_QUALITY` 설정, 디버그/퍼스펙티브 이미지는 리사이즈 기본 적용(예: width=640).
- **퍼스펙티브 변환 입력 형상 명시**: `getPerspectiveTransform`의 `src_points`를 `(4,2) float32`로 reshape하여 안전성 확보.
- **발행 레이트 제한**: 디버그/퍼스펙티브 이미지는 5–10 Hz 한정(가독성 유지, 부하 절감).
- **스레드 수 제어**: OpenCV 내부 스레드와 executor 워커 수를 설정으로 관리(과다 스레딩 방지).

### P2 안정성/유지보수(코드 품질)

- **일관된 정지 시그니처**: 서비스 `stop()`을 모두 async로 통일하면 호출부 단순화.
- **풍부한 컨텍스트 로깅**: 동기화 실패율, 처리 지연(ms), 큐 깊이 등 지표 로그 추가.
- **예외 처리 강화**: 컨텍스트 포함 메시지와 `exc_info=True` 일관 적용.

### 파일별 액션 체크리스트

- **`app/services/camera_service.py`**
  - `stop`을 async로 변경하고 `await self.api_client.aclose()` 호출.
  - 정책 유지: `websockets.connect(..., max_size=None)`로 1280x720 BGR 프레임 수용. 수신된 프레임의 바이트 길이/shape 검증 유지.
  - 필요 시 입력 프레임 스로틀(설정값 기반 최대 FPS).

- **`app/services/frame_sync_service.py`**
  - 현 구조(최근 컬러/깊이 유지)는 적절. 동기화 실패율/지연 로깅 지표 추가.
  - 허용 오차(ms)를 설정값으로 노출.

- **`app/services/aruco_service.py`**
  - 입력을 그레이스케일로 변환 후 탐지 호출.
  - `detectMarkers`/`solvePnP`/DTO 변환을 `to_thread` 오프로딩.
  - 보드 `id→obj_points` 캐시로 인덱스 탐색 제거.
  - `solvePnP(..., flags=cv2.SOLVEPNP_IPPE_SQUARE)` 사용.

- **`app/services/image_service.py`**
  - `cv2.imencode`/투영 연산을 `to_thread`로 오프로딩.
  - JPEG 품질/리사이즈를 설정값으로 노출, 디버그/퍼스펙티브 발행 레이트 제한.
  - `getPerspectiveTransform` 입력 형상 명시적 reshape.

- **`app/services/robot_service.py`**
  - 매우 큰 URDF 로딩 시 `to_thread` 고려. `stop`을 async로 통일(옵션).

### 설정 추가 제안(`app/core/config.py`)

- **`JPG_QUALITY`**: JPEG 품질(기본 80–90)
- **`WS_DEBUG_FPS`**: 디버그/퍼스펙티브 발행 상한(5–10 Hz)
- **`ARUCO_MAX_WORKERS`**: 아루코 스레딩 워커 수
- **`CAMERA_MAX_FPS`**: 카메라 프레임 전파 최대 FPS(스로틀)
- **`FRAME_SYNC_TOLERANCE_MS`**: 동기화 허용 오차

### 운영 지표(권장)

- `SYNC_FRAME_READY` → `WS_*` E2E 지연 p50/p95
- 동기화 실패율(%) 및 discard 원인(앞섬/뒤처짐)
- CPU 사용률, 이벤트 루프 블로킹 시간, 프레임 드랍률

### 기존 메모(원문)

- 메인 이벤트 루프 블로킹 위험: OpenCV/NumPy 연산과 JPEG 인코딩이 모두 async 핸들러 안에서 동기 실행됩니다. detectMarkers, solvePnP, imencode, warpPerspective 등은 CPU 바운드라 이벤트 루프를 막아 프레임 드랍/지연을 유발합니다.
- 수정: CPU 바운드 구간을 await asyncio.to_thread(...)로 오프로딩. 필요 시 전용 ThreadPoolExecutor(max_workers=N) 도입.
- 백프레셔 부재: 생산 속도(카메라) > 소비 속도(아루코/이미지)일 때 이벤트가 무제한 누적될 소지. 최신 프레임만 유지/발행하는 “coalesce/drop-old” 전략 필요.
- 수정: 서비스 내부에 “최신만 처리” 가드(타임스탬프 기준 스로틀) 또는 이벤트 버스 레벨에서 bounded queue + drop.
- 대용량 데이터 전달: 원본 numpy 프레임을 여러 서비스가 공유. 프로세스 내 참조 전달이라도, DTO 변환/복사와 JPEG 바이트 생성이 메모리 churn을 야기.
- 수정: 내부 파이프라인은 가능한 NumPy 그대로, 외부 전송 직전만 DTO/JPEG로 변환. JPEG 품질/리사이즈 기본값 조정으로 부하 감소.
메인 이벤트 루프 블로킹 위험: OpenCV/NumPy 연산과 JPEG 인코딩이 모두 async 핸들러 안에서 동기 실행됩니다. detectMarkers, solvePnP, imencode, warpPerspective 등은 CPU 바운드라 이벤트 루프를 막아 프레임 드랍/지연을 유발합니다.
수정: CPU 바운드 구간을 await asyncio.to_thread(...)로 오프로딩. 필요 시 전용 ThreadPoolExecutor(max_workers=N) 도입.
백프레셔 부재: 생산 속도(카메라) > 소비 속도(아루코/이미지)일 때 이벤트가 무제한 누적될 소지. 최신 프레임만 유지/발행하는 “coalesce/drop-old” 전략 필요.
수정: 서비스 내부에 “최신만 처리” 가드(타임스탬프 기준 스로틀) 또는 이벤트 버스 레벨에서 bounded queue + drop.
대용량 데이터 전달: 원본 numpy 프레임을 여러 서비스가 공유. 프로세스 내 참조 전달이라도, DTO 변환/복사와 JPEG 바이트 생성이 메모리 churn을 야기.
수정: 내부 파이프라인은 가능한 NumPy 그대로, 외부 전송 직전만 DTO/JPEG로 변환. JPEG 품질/리사이즈 기본값 조정으로 부하 감소.
