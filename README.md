# Enterprise Binpicking Server

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)](https://fastapi.tiangolo.com/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

실시간 3D 비전 스트림과 ArUco 기반 좌표계 추정을 통합해 로봇 픽킹(비전-로봇) 시나리오를 지원하는 고성능 FastAPI 백엔드입니다. 카메라·로봇 하드웨어와 직접 통신하며, 이미지/포인트클라우드/좌표 변환을 이벤트 기반으로 가공하고 REST·WebSocket·대시보드 형태로 제공합니다.

---

## 목차

- [개요](#개요)
- [핵심 기능](#핵심-기능)
- [시스템 구성 요소](#시스템-구성-요소)
- [데이터 파이프라인](#데이터-파이프라인)
- [API & WebSocket 요약](#api--websocket-요약)
- [설치 및 환경 구성](#설치-및-환경-구성)
- [서버 실행](#서버-실행)
- [필수 설정 파일](#필수-설정-파일)
- [로봇 IK 서비스](#로봇-ik-서비스)
- [관측 가능성과 운영](#관측-가능성과-운영)
- [테스트 & 품질 관리](#테스트--품질-관리)
- [참고 문서](#참고-문서)
- [배포 시 고려 사항](#배포-시-고려-사항)

## 개요

Enterprise Binpicking Server는 다음을 목표로 설계되었습니다.

- 고속 카메라 데이터 수집과 동기화
- ArUco 보드·마커 기반 3D Pose 추정 및 다중 좌표계 변환
- Pinocchio / ikpy 기반 로봇 역기구학 계산
- 실시간 스트리밍과 시각화 UI 제공
- 서비스 확장을 고려한 이벤트 주도 아키텍처

## 핵심 기능

- **저지연 이미지 스트리밍**: 컬러(BGR)·뎁스(Z16) 스트림을 JPEG 또는 Raw 형태로 수신·배포.
- **ArUco Pose 추정**: 보드/외부 마커를 감지하고 카메라·보드·로봇 좌표계로 변환.
- **포인트클라우드 생성**: 동기화된 컬러·뎁스 이미지로 RGB 포인트클라우드를 계산해 스트리밍.
- **로봇 IK 백엔드**: Pinocchio와 ikpy 두 엔진을 모두 지원하며 변형별 URDF를 관리.
- **이벤트 기반 파이프라인**: `EventBus`를 활용한 느슨한 결합, 고성능 비동기 처리.
- **관측 및 대시보드**: 주요 상태·이미지·변환을 REST, WebSocket, HTML 대시보드로 노출.

## 시스템 구성 요소

- **ApplicationStore** (`app/stores/`): 도메인별 핸들러(Device, Calibration, Camera, Image, Aruco, Robot, Pointcloud, Event)로 상태를 관리하는 중앙 저장소. 모든 접근은 스레드 세이프.
- **EventBus** (`app/core/event_bus.py`): 비동기 pub/sub 허브. 이벤트 메트릭(FPS, 발행 횟수)을 `EventHandler`에 기록.
- **CameraService** (`app/services/camera_service.py`): 카메라 REST/WS 연동, raw/JPEG 스트림 수신, 주기적 상태 동기화.
- **FrameSyncService**: 컬러·뎁스 프레임을 허용 오차(기본 150ms) 내에서 한 쌍으로 맞추고 `SYNC_FRAME_READY` 이벤트 발행.
- **ArucoService**: 깊이 기반 정합 + PnP 하이브리드 방식으로 3D Pose 계산, 좌표계 변환 스냅샷 작성, 이벤트 발행.
- **ImageService**: JPEG 인코딩/디코딩, 디버그 이미지와 보드 원근 보정 이미지 생성, WebSocket 이벤트 전달.
- **PointcloudService**: 포인트클라우드 생성 및 다운샘플링, 스트리밍 이벤트 발행.
- **RobotServicePinocchio / RobotServiceIkpy**: URDF 로딩, 변형(fixed/prismatic) 관리, 역기구학 연산.
- **StreamingService**: WebSocket 구독자 관리(`ConnectionManager`)와 이벤트 연결, 이미지·좌표·포인트클라우드를 브로드캐스트.
- **API & Views** (`app/api/v1`, `app/static/templates`): REST 엔드포인트와 운영 대시보드 템플릿을 제공.

## 데이터 파이프라인

```
카메라 HW ──WS/REST──▶ CameraService ──┐
                                       ├─▶ EventBus ──▶ FrameSyncService ──▶ SYNC 프레임
                                       │                                ├─▶ ArucoService ──▶ 좌표 변환/디버그
                                       │                                ├─▶ PointcloudService ──▶ PointCloud
                                       │                                └─▶ ImageService ──▶ JPEG 변환
                                       └────────────────────────────────────▶ StreamingService ──▶ WebSocket Clients
```

각 서비스는 최신 프레임만 처리하도록 코얼레싱 버퍼를 두어 지연을 최소화하며, CPU 바운드 작업은 `asyncio.to_thread`로 오프로딩합니다.

## API & WebSocket 요약

### HTTP 주요 엔드포인트 (prefix: `/api`)

| 경로 | 메서드 | 설명 |
| --- | --- | --- |
| `/health/` | GET | 헬스 체크 |
| `/store/status` | GET | 전체 Store 상태 스냅샷 |
| `/store/events/status` | GET | 이벤트 타임라인 및 FPS |
| `/device/camera/status` | GET | 카메라 디바이스/스트림 정보 |
| `/images/color.jpg` | GET | 최신 컬러 JPEG |
| `/images/depth.jpg` | GET | 최신 뎁스 JPEG |
| `/images/color/raw` | GET | 컬러 Raw(BGR) bytes |
| `/aruco/status` | GET | 보드 감지 결과 |
| `/transforms/all?frame=robot` | GET | 선택 좌표계 기준 변환 스냅샷 |
| `/pointcloud/data?max_points=10000` | GET | 다운샘플링된 포인트클라우드 |
| `/robot/ik` | POST | 기본 백엔드 IK 계산 |
| `/robot/ik/{backend}` | POST | 백엔드 선택 IK (pinocchio / ikpy) |

> 전체 목록과 스키마는 `API_DOCUMENTATION.md` 및 Swagger (`/docs`) 참고.

### WebSocket 채널

| 채널 | 설명 | 페이로드 |
| --- | --- | --- |
| `/ws/color_jpg` | 최신 컬러 JPEG 이미지 스트림 | JPEG bytes |
| `/ws/depth_jpg` | 최신 뎁스 JPEG 스트림 | JPEG bytes |
| `/ws/aruco_debug_jpg` | 디버그 오버레이 이미지 | JPEG bytes |
| `/ws/board_perspective_jpg` | 보정된 보드 탑뷰 | JPEG bytes |
| `/ws/transforms_camera` | 카메라 좌표계 변환 스냅샷 | JSON |
| `/ws/transforms_board` | 보드 좌표계 변환 스냅샷 | JSON |
| `/ws/transforms_robot` | 로봇 좌표계 변환 스냅샷 | JSON |
| `/ws/pointcloud` | 다운샘플링된 포인트클라우드 | JSON (points, colors) |

## 설치 및 환경 구성

### 사전 준비

- Python 3.11+
- Conda (Miniconda/Anaconda) 권장
- OpenCV/PyTorch 연산이 가능한 CPU, 포인트클라우드/IK 가속을 위한 GPU(Optional)
- 외부 카메라·로봇 제어 장비와 네트워크 연결

### Conda 환경 생성

```bash
conda env create -f environment.yml
conda activate binpicking_env
# 또는 경량 구성이 필요하면 environment2.yml 참고
```

### 의존성 확인

- `requirements.txt` : pip 기반 설치가 필요할 때 사용
- `environment.yml` : 개발 기본 환경 (OpenCV, Pinocchio, ikpy 등 포함)
- `environment2.yml` : 대안 혹은 경량 환경

## 서버 실행

### 개발 모드(Uvicorn)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 52000 --reload
```

- `--reload`는 코드 변경 시 자동 재시작(개발 전용)
- 포트는 카메라·로봇 장비 설정에 따라 조정

### Run 스크립트

```bash
python run.py  # 기본 포트: 53000
```

### 실행 후 확인

- 메인 대시보드: `http://localhost:52000/`
- OpenAPI 문서: `http://localhost:52000/docs`
- ReDoc: `http://localhost:52000/redoc`

## 필수 설정 파일

`app/config/` 디렉터리에 다음 파일을 준비해야 합니다.

| 파일 | 설명 |
| --- | --- |
| `.env` | 카메라 API URL, WebSocket 타임아웃, 로봇 URDF 경로 등 환경 변수 (`app/core/config.py` 참고) |
| `aruco_place.csv` | 보드 구성 마커 ID 및 3D 위치 |
| `robot_position.json` | 보드 좌표계 대비 로봇 베이스 Pose (JSON: x,y,z) |
| `aruco_config.json` *(선택)* | 감지 파라미터 커스터마이징. 없으면 기본값 로드 |

환경 변수 주요 항목 (`AppSettings`):

- `CAMERA_API_BASE_URL`
- `COLOR_STREAM_MODE` / `DEPTH_STREAM_MODE` (`jpeg` 또는 `raw`)
- `ROBOT_URDF_PATH_FIXED`, `ROBOT_URDF_PATH_PRISMATIC`
- `ROBOT_IK_BACKEND` (`pinocchio` or `ikpy`)
- `POINTCLOUD_DOWNSAMPLE_FACTOR`, `MAX_POINTCLOUD_DEPTH_M`

## 로봇 IK 서비스

- **Pinocchio 백엔드** (`RobotServicePinocchio`): 고정/프리스매틱 URDF를 모두 로드, DLS 기반 반복 IK, 조인트 제한 적용.
- **ikpy 백엔드** (`RobotServiceIkpy`): 체인 기반 역기구학, 활성 조인트 필터링, workspace 검사.
- `POST /api/robot/ik`: 기본 백엔드를 이용한 다중 타깃 IK 계산.
- `POST /api/robot/ik/{backend}`: 특정 엔진 지정.
- `POST /api/robot/ik/*/downward`: 그리퍼 다운어프로치 전용 헬퍼.
- 자세한 파라미터와 예시는 `docs/robot_ik_api.md`, `docs/robot_ik_api_usage.md`, `docs/robot_ik_plan.md` 참고.

서비스 시작 시 URDF는 `ApplicationStore.robot`에 캐시되며, 마지막 관절 값은 반복 호출의 초기값으로 재사용됩니다.

## 관측 가능성과 운영

- **상태 모니터링**
  - `/api/store/status`: 디바이스/캘리브레이션/이벤트/포인트클라우드 한눈에 확인.
  - `/api/store/events/status`, `/api/store/events/fps`: 이벤트 발행 빈도 추적.
- **대시보드**
  - `/api/views/v1/images`: 이미지 스트림 모니터링.
  - `/api/views/v1/transforms`: 좌표 변환 스트림 모니터링.
  - `/`: 메인 대시보드(템플릿 `main_dashboard.html`).
- **로깅**
  - `app/core/logging.py`에서 Loguru 기반 설정.
  - 서비스별 시작/중지 및 오류 상황을 상세 로깅.
- **백그라운드 라이프사이클**
  - FastAPI lifespan hook에서 모든 서비스 start/stop을 통합 관리.
  - 장애 발생 시 이벤트 구독 해제 및 리소스 정리를 보장.

## 테스트 & 품질 관리

- **단위 테스트**
  - `pytest` (예: `pytest tests/test_robot_api_ik.py`)
- **정적 분석 & 포맷팅**
  - `ruff`, `black`, `mypy` (환경에 포함). CI나 pre-commit 훅 구성 권장.
- **샘플 스크립트**
  - `scripts/test_robot_downward.py`, `scripts/test_ikpy_api_simple.py` 등으로 로컬 검증.

## 참고 문서

- `API_DOCUMENTATION.md`: REST/WebSocket 기능 정의와 페이로드 설명.
- `docs/robot_ik_api.md`: IK 설계 및 API 세부 스펙.
- `docs/robot_ik_api_usage.md`: 단계별 활용 예제.
- `docs/robot_ik_implementation.md`: 내부 구현 세부 사항.
- `docs/robot_ik_plan.md`: 계획 및 백로그.
- `docs/opc_tags.md`, `docs/robot_ik_api.md` 등 기타 도메인 문서.
- `backup/` 및 `report/` 폴더: 기술 부채/보고서 기록.

## 배포 시 고려 사항

- **카메라·로봇 네트워크**: 방화벽, 포트(WS/REST) 및 대역폭 확보.
- **GPU/CPU 자원**: 고해상도 스트림의 JPEG 인코딩 및 ArUco 추적은 CPU 부하가 크므로 코어 수 확인.
- **신뢰성**: CameraService는 자동 재연결 로직을 포함하지만, 안정적인 NTP 동기화가 필요.
- **확장**: 새로운 이벤트 타입을 추가할 경우 `app/core/event_type.py`, `EventBus`, 관련 Store 핸들러를 함께 확장.
- **로그/모니터링 통합**: 외부 관측 시스템(예: ELK, Prometheus)과 연동하려면 Loguru sink/metrics exporter 추가.
- **컨테이너화**: Conda 환경 크기를 고려하여 Mamba/Miniforge 또는 slim pip 환경으로 재구성 가능.

---

지속적인 개선이나 신규 기능 제안은 문서(`docs/`, `backup/`)에 정리된 계획과 일정을 참고하세요. Issue나 추가 질문이 있다면 언제든지 환영합니다.
