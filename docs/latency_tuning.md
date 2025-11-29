# 레이턴시 튜닝 가이드

## 1. 파이프라인 계측
- `camera_service`, `frame_sync_service`, `image_service`, `pointcloud_service`에 처리 시간 로그가 추가되었다.  
  - `FrameSync` 로그: 컬러·뎁스 페어가 맞춰질 때 `delta=??ms` 출력.  
  - `ImageService` 로그: JPEG 인코딩이 30 ms 이상이면 `JPEG encode slow path` 경고.  
  - `PointcloudService` 로그: 생성에 걸린 시간과 포인트 수 기록.  
- 로그 레벨을 `DEBUG` 이상으로 올리면 구간별 시간을 확인해 병목을 쉽게 찾을 수 있다.

## 2. 설정값 조정 포인트
| 환경 변수 | 기본 | 설명 |
|-----------|------|------|
| `FRAME_SYNC_TOLERANCE_MS` | 150 | 컬러·뎁스 페어 허용 오차. 하드웨어가 안정된 경우 80~120 ms로 낮추면 대기 시간이 줄어든다. |
| `BOARD_VIEW_UPDATE_MODE` | `aruco` | `hybrid` 또는 `frame`으로 바꾸면 최신 컬러 프레임만으로도 디버그/보정 뷰를 갱신할 수 있다. |
| `BOARD_VIEW_POSE_TTL_SECONDS` | 0.3 | Pose TTL을 넉넉하게(1.0~1.5) 주면 아루코 감지 주기가 길어도 디버그 이미지가 끊기지 않는다. |
| `CAMERA_WS_RECV_TIMEOUT_SECONDS` | 3.0 | 카메라 스트림 무응답을 재연결로 간주하는 임계값. 실제 프레임 주기에 맞춰 조정하면 불필요한 재접속을 줄일 수 있다. |

환경 값은 `app/config/.env`에서 수정 후 서버를 재시작하면 적용된다.

## 3. 입력 해상도 / ROI 최적화
- 카메라에서 전송하는 해상도를 한 단계 낮추거나, (장비 허용 시) 흑백 전용 스트림을 쓰면 ArUco·Pointcloud 연산이 바로 줄어든다.
- 아루코 대상 영역이 상판 등 특정 구역이라면, 카메라 측 API에서 ROI(관심 영역) 크롭을 지원하는지 확인해 불필요한 픽셀을 줄인다.
- 뎁스 이미지는 포인트클라우드에서 다운샘플링(`POINTCLOUD_DOWNSAMPLE_FACTOR`)을 더 크게 설정하는 방법도 있다. 예: 4 → 6.

## 4. 권장 워크플로
1. 로그 레벨을 `DEBUG`로 올려 각 구간 시간을 측정한다. (`LOG_LEVEL=DEBUG`)
2. 위 표의 설정값을 장비 상황에 맞게 조정한다.
3. 해상도/ROI를 줄이면서 성능 차이를 비교한다.
4. 병목이 명확해지면(예: 아루코 detect가 항상 80 ms 이상) 해당 구간만 집중적으로 최적화한다.

이 단계들을 따르면 코드 구조를 크게 바꾸지 않고도 레이턴시를 안정적으로 줄일 수 있다.

