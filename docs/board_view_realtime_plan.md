# Realtime Board View Plan

## 목표
- 아루코 업데이트 주기가 느려도 컬러 프레임이 들어올 때마다 `board_perspective_jpg`, `aruco_debug_jpg`를 갱신해 UI가 더 부드럽게 반응하게 만든다.
- Pose가 없거나 오래된 경우에는 안전하게 스킵하거나 마지막 Pose를 재사용한다.
- 기존 `EventType.ARUCO_UPDATE` 트리거 방식과 병행할 수 있도록 옵션화한다.

## 현재 구조 복기
- `ImageService`는 `EventType.ARUCO_UPDATE`에서만 디버그/보드 전면 이미지를 생성한다.
- 컬러/뎁스 JPEG 스트림은 카메라 프레임 이벤트로 갱신된다.
- 따라서 자주 들어오는 컬러 프레임을 활용하지 못하고, ArUco가 느리면 전면 이미지도 느려진다.

## 제안 아키텍처

### 1. 상태 캐시 추가
- `ImageService` 필드
  - `_latest_color_frame: Optional[tuple[np.ndarray, float]]`
  - `_latest_board_pose: Optional[Pose]`
  - `_latest_markers: list[DetectedMarker]`
  - `_last_pose_timestamp: float`
  - `_board_render_task: Optional[asyncio.Task]` (중복 렌더 방지)
- `handle_color_image`에서 `_latest_color_frame` 갱신.
- `handle_aruco_update`에서 Pose/마커 캐시 갱신.

### 2. 업데이트 모드 설정
- `AppSettings` 예: `BOARD_VIEW_UPDATE_MODE: Literal["aruco", "frame"] = "aruco"`
- `"aruco"`: 기존 동작 유지.
- `"frame"`: 컬러 프레임 이벤트마다 렌더 파이프라인 실행.
- 추후 `"hybrid"` 모드(아루코 트리거 + 프레임 트리거 병합)도 고려.

### 3. 렌더링 파이프라인
```
handle_color_image -> self._latest_color_frame 저장 -> self._schedule_board_render()
```
- `_schedule_board_render()`:
  - Pose가 없거나 `now - _last_pose_timestamp > pose_ttl_ms`이면 스킵.
  - `_board_render_task`가 이미 동작 중이면 최신 프레임만 남기고 리턴.
  - 아니면 `asyncio.create_task(self._render_board_views())`.
- `_render_board_views()`:
  - 로컬 스냅샷(컬러 이미지, Pose, 마커) 확보.
  - `asyncio.to_thread`로 `get_board_perspective_*` 및 `get_aruco_debug_image_*` 호출.
  - JPEG 생성 성공 시 Store 업데이트 + `EventType.WS_*` 발행.
  - 실패하면 다음 프레임에서 다시 시도.

### 4. 동기화 전략
- Pose가 ArUco에서 들어올 때마다 `_last_pose_timestamp`를 업데이트.
- 컬러 프레임과 Pose 사이에 시차가 있어도, “최근 Pose + 최신 프레임”을 조합해 사용.
- 일정 시간(`pose_ttl_ms`, 예: 300ms) 이상 지난 Pose는 신뢰하지 않고, 렌더를 스킵한다.
- ArUco 이벤트는 계속 Pose 캐시를 갱신하므로, 렌더 주기가 Pose 주기보다 빠르더라도 최신 Pose를 재사용한다.

### 5. 백프레셔 & 성능
- `_board_render_task` + `_pending_board_render` 같은 플래그로 프레임당 하나만 처리.
- 필요 시 LIFO 큐나 debouncing(예: 1/프레임당 최대 1회) 적용.
- 렌더 시간 측정해서 33ms(30fps)을 넘으면 경고.
- `asyncio.to_thread`로 CPU 바운드를 워커에 오프로딩.

### 6. 예외 처리
- Pose 없음: 스킵 로그만 남기고 종료.
- JPEG 변환 실패: 워닝 로그 + 다음 프레임 시도.
- Store/WS 발행은 지금과 동일 패턴 활용.

### 7. 마이그레이션 플랜
1. `AppSettings` 신규 옵션 추가, 기본 `"aruco"`.
2. `ImageService`에 캐시 필드 및 스케줄러 메서드 추가.
3. `"frame"` 모드일 때 `handle_color_image`가 `_schedule_board_render()` 호출하도록 변경.
4. 부하 테스트 후 기본값 유지/변경 여부 결정.
5. 문서(`docs/ws_performance_plan.md` 등) 갱신.

## 예상 효과
- ArUco 감지 속도와 무관하게 전면 보정/디버그 스트림이 컬러 프레임 주기에 가까운 속도로 갱신.
- 아루코 감지가 일시 실패해도 마지막 Pose를 잠시 재사용하여 UI가 멈추지 않음.
- 포즈 신선도 정책을 통해 잘못된 Pose를 사용하지 않도록 안전장치 유지.

