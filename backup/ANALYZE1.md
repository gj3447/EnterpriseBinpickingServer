# 프로젝트 분석 보고서: EnterpriseBinpickingServer

## 1. 프로젝트 개요 (초기 분석)

이 프로젝트는 **FastAPI 기반의 산업용 빈 피킹(Bin Picking) AI 서버**로 판단됩니다.

- **기반 기술:** Python, FastAPI, Uvicorn
- **핵심 기능:**
    - **웹 API:** FastAPI를 통한 RESTful API 및 WebSocket 통신 제공.
    - **컴퓨터 비전 & 3D 처리:** OpenCV, Open3D를 활용한 이미지 및 3D 데이터 처리.
    - **AI 모델 추론:** PyTorch, ONNX Runtime을 사용한 딥러닝 기반 객체 탐지 및 자세 추정.
    - **로봇 연동:** `doosan-robot` 폴더 및 `roslibpy`를 통한 로봇 시스템과의 연동.
    - **데이터베이스:** MSSQL을 사용하여 데이터 저장 및 관리. [[memory:4462668]]
    - **비동기 처리:** Celery를 이용한 백그라운드 작업 수행.
- **실행 진입점:** `run.py`에서 `app.main:app`을 Uvicorn으로 실행.

---

## 2. 상세 분석 계획

다음 영역에 대해 심층 분석을 진행할 수 있습니다. 분석을 원하는 항목을 선택해주세요.

### 1. API 엔드포인트 분석
*   **분석 내용:**
    *   **라우팅 구조:** `app/api/v1/router.py`에서 각 기능별 API 라우터를 통합하여 관리합니다. (`/api/v1` 접두사 사용)
    *   **HTTP Endpoints:**
        *   `/images/{type}.jpg`: 최신 컬러, 뎁스, 마스크 이미지를 JPEG/PNG 형식으로 제공합니다.
        *   `/health`: 서버 상태를 확인합니다.
        *   `/store`: 내부 데이터 저장소와 관련된 기능을 제공합니다.
        *   `/aruco`, `/device`, `/transforms`: ArUco 마커, 장치 제어, 좌표계 변환과 관련된 API를 제공합니다.
        *   `/views`: 서버 측에서 렌더링된 웹 UI 페이지를 제공합니다.
    *   **WebSocket Endpoints:**
        *   `/ws/images/{type}`: 컬러, 뎁스 등 원본 이미지 스트림을 실시간으로 전송합니다.
        *   `/ws/images/aruco_debug`: ArUco 마커 검출 과정을 디버깅하기 위한 영상 스트림을 전송합니다.
        *   `/ws/masks/{type}`: 보드, 마커 등 특정 영역의 마스크 이미지 스트림을 전송합니다.
        *   `/ws/transforms`: 좌표계 변환 정보를 실시간으로 구독합니다.
*   **관련 파일:**
    *   `app/api/v1/router.py` (라우터 통합)
    *   `app/api/v1/endpoints/images.py` (이미지 조회 API)
    *   `app/api/v1/endpoints/websockets.py` (실시간 스트리밍 API)
    *   `app/api/v1/endpoints/health.py`, `store.py`, `aruco.py`, `device.py`, `transforms.py`, `views.py` 등

### 2. AI 모델 및 추론 로직 분석
*   **분석 내용:**
    *   **현재 상태:** 딥러닝 모델(`.pt`, `.onnx`)을 로드하고 실행하는 직접적인 AI 추론 코드는 발견되지 않았습니다. 현재 로직은 전통적인 OpenCV 기능에 의존하고 있습니다.
    *   **컴퓨터 비전 로직:**
        *   `ArucoService`: `cv2.aruco` 라이브러리를 사용하여 ArUco 마커를 검출하고 3D 공간에서의 위치(Pose)를 계산합니다. 이는 딥러닝 모델이 아닌, 고전적인 컴퓨터 비전 기법입니다.
        *   `ImageService`: 이미지 인코딩, 원근 변환, 마스킹 등 OpenCV를 사용한 이미지 전/후처리 유틸리티를 제공합니다.
    *   **AI 기능 확장 추정:** `environment.yml`에 PyTorch, ONNX Runtime 등이 명시된 것으로 보아, 향후 빈 피킹 대상 객체를 탐지하고 자세를 추정하는 딥러닝 모델이 추가될 것으로 예상됩니다.
    *   **아키텍처:** `CameraService`에서 `IMAGE_RECEIVED` 이벤트가 발생하면 `ArucoService`가 이를 받아 처리하는 이벤트 기반 구조를 가집니다. AI 추론 서비스가 추가된다면 동일한 이벤트를 구독하여 작동할 가능성이 높습니다.
*   **관련 파일:**
    *   `app/services/aruco_service.py` (ArUco 마커 기반 위치 추정)
    *   `app/services/image_service.py` (OpenCV 기반 이미지 처리)
    *   `app/services/camera_service.py` (이미지 스트림 수신 및 이벤트 발행)
    *   `app/core/event_bus.py` (서비스 간 통신)
    *   `environment.yml` (AI 관련 라이브러리 의존성 명시)

### 3. 로봇 연동 로직 분석
*   **분석 내용:**
*   **관련 파일:**

### 4. 데이터베이스 스키마 분석
*   **분석 내용:**
*   **관련 파일:**

### 5. 사용자 인터페이스(UI) 분석
*   **분석 내용:**
*   **관련 파일:**
