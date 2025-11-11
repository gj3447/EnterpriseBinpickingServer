## ArUco Processing Bottleneck Plan

### 1. 현황 요약
- 컬러/깊이 스트림 수집과 동기화는 별도의 비동기 태스크로 정상 작동 중.
- `ArucoService._detect_markers_and_board_pose()`가 `asyncio.to_thread()`로 실행되지만, 단일 ThreadPoolExecutor에 의존하여 CPU 바운드 시간이 길어지면 후속 프레임 처리가 지연됨.
- 새로 추가된 깊이-컬러 정렬(`_align_depth_to_color`)과 PnP 리파인(`solvePnPRefineLM`)이 프레임당 연산량을 크게 증가시켜 디버그 이미지/포즈 업데이트 주기가 감소함.
- 동일 풀을 사용하는 IK 연산(`RobotServiceIkpy._solve_ik_sync`)도 대기 시간이 늘어나는 부수 효과 발생.

### 2. 병목 분석
| 구간 | 상세 내용 | 영향 |
| --- | --- | --- |
| 깊이 정렬 (`_align_depth_to_color`) | 깊이 프레임 전체 픽셀을 순회하여 컬러 좌표로 투영 | O(hw) 연산, 1280×720일 때 프레임당 수 ms 이상 |
| 마커 포즈 추정 | 내부 샘플링 + 3D-3D 정합 + `solvePnPRefineLM` | 각 마커마다 수차례 기하 연산 |
| 보드 포즈 추정 | 보드 마커 전체에 대해 위 동일 과정 수행 | RANSAC/폴백 포함 시 계산량 증가 |
| ThreadPool 경쟁 | ArUco/IK 모두 기본 executor 공유 | 긴 작업이 있을 때 상호 대기 |

### 3. 개선 전략
#### 3.1 깊이 정렬 최적화
- **ROI 기반 정렬**: 전체 프레임 대신 마커 bounding box 주변 픽셀만 정렬.
- **SDK 정렬 활용**: Realsense `rs.align(rs.stream.color)` 등 하드웨어/SDK 제공 align을 사용해 프레임 수신 시 이미 정렬된 깊이를 획득.
- **캐시/다운샘플**: 정렬 결과를 다운샘플해 보정용으로만 사용하거나, 일정 프레임마다 정렬 수행.

#### 3.2 포즈 계산 가속
- `solvePnPRefineLM`는 마커/보드 특별한 경우에만 호출하도록 조건(깊이 품질 기준) 조정.
- 샘플링 윈도우/offset을 줄여 depth median 계산 비용 최소화.
- 보드 포즈는 이미 PnP로 안정적이라면 깊이 기반 보정 빈도를 낮추거나, 다수 마커가 감지된 프레임에만 수행.

#### 3.3 비동기 스케줄링 조정
- **ThreadPool 분리**: `ThreadPoolExecutor(max_workers=n)` 두 개를 만들어 ArUco와 IK를 각각 별도 풀에서 실행.
- **처리 주기 제한**: `_process_sync_loop`에서 일정 간격(예: 2~3 프레임에 한 번)만 ArUco 계산 수행, 나머지는 최신 결과 재사용.
- **백로그 방지**: `_latest_sync` 큐가 증가하지 않도록 현재 처리 중일 때는 프레임을 건너뛰는 로직 유지 또는 개선.

#### 3.4 프로파일링 & 로깅
- `_detect_markers_and_board_pose` 실행 시간 로깅 → 최댓값/평균 추적.
- `_align_depth_to_color` 및 PnP 리파인 내부에 타이머를 넣어 실제 비용 확인.
- `aruco_config.json`의 `debug.log_metrics` 옵션을 활용해 FPS, skip rate 모니터링.

### 4. 예상 효과 및 주의 사항
- ROI 정렬 + 조건부 리파인으로 프레임당 연산 시간을 크게 줄일 수 있음.
- ThreadPool 분리 시 Python GIL 영향은 여전히 남지만, 서로 다른 작업이 대기하는 시간은 완화.
- 처리 주기 제한은 디버그 뷰 갱신 빈도가 감소할 수 있으므로, UI 요구사항에 맞춰 조절 필요.

### 5. 추후 작업 체크리스트
- [ ] ThreadPool 분리 실험 (ArUco/IK 각각 전용 executor).
- [ ] `_align_depth_to_color` ROI 최적화 및 실제 비용 측정.
- [ ] `solvePnPRefineLM` 호출 조건 최적화 (깊이 일관성/샘플 수 기준).
- [ ] 프로파일링 결과 기반으로 window/offset 등 파라미터 튜닝.
- [ ] 최종적으로 FPS·CPU 사용률 모니터링 후 추가 조정.
