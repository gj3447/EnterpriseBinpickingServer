# Enterprise Binpicking Server

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)](https://fastapi.tiangolo.com/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

실시간 3D 비전 데이터를 처리하고, ArUco 마커 기반의 정밀한 좌표 변환을 수행하여 로보틱스 애플리케이션에 필요한 데이터를 제공하는 고성능 백엔드 서버입니다.

---

## 🚀 핵심 기능

- **실시간 카메라 스트리밍**: 저지연 웹소켓을 통해 컬러(BGR) 및 뎁스(Z16) 원본 이미지 스트림을 제공합니다.
- **ArUco 기반 3D Pose 추정**: 카메라 영상에서 ArUco 보드 및 개별 마커를 실시간으로 탐지하고, 3D 공간에서의 정확한 위치와 방향(Pose)을 계산합니다.
- **동적 좌표계 변환**: '카메라', '보드', '로봇' 좌표계를 기준으로 모든 객체(카메라, 보드, 로봇, 마커)의 Pose를 실시간으로 변환하여 제공합니다.
- **비동기 이벤트 기반 아키텍처**: 각 서비스(`Camera`, `Aruco`, `Image`, `WebSocket`)가 독립적으로 동작하고, `EventBus`를 통해 효율적으로 통신하여 높은 성능과 확장성을 보장합니다.
- **상세 모니터링 대시보드**: 모든 이미지 및 좌표 변환 스트림의 상태를 실시간으로 확인할 수 있는 웹 기반 대시보드를 제공합니다.
- **상태 관리 및 관측 가능성**: 시스템의 모든 주요 상태와 이벤트 발생 현황을 API를 통해 실시간으로 모니터링할 수 있습니다.

## 🏗️ 아키텍처

본 서버는 FastAPI를 기반으로 한 현대적인 비동기 이벤트 주도 아키텍처로 설계되었습니다.

- **Services**: `CameraService`, `FrameSyncService`, `ArucoService` 등 각 핵심 비즈니스 로직을 담당하는 독립적인 컴포넌트입니다.
- **EventBus**: 서비스 간의 통신을 담당하는 중앙 허브입니다. 서비스들은 서로를 직접 호출하지 않고, 이벤트를 발행(Publish)하거나 구독(Subscribe)함으로써 느슨한 결합(Loose Coupling)을 유지합니다.
- **Store (`ApplicationStore`)**: 시스템의 모든 상태를 관리하는 중앙 저장소입니다. 각 데이터 도메인(`Camera`, `Image`, `Aruco` 등)은 전용 핸들러 클래스를 통해 관리되며, 모든 데이터 접근은 스레드로부터 안전(Thread-safe)합니다.
- **API & WebSockets**: FastAPI 라우터를 통해 외부 세계에 HTTP API와 실시간 WebSocket 스트림을 노출합니다.

## 🛠️ 기술 스택

- **웹 프레임워크**: FastAPI, Uvicorn, WebSockets
- **컴퓨터 비전 & 3D**: OpenCV, NumPy, SciPy
- **데이터 처리 및 검증**: Pydantic, Pydantic-Settings
- **핵심 유틸리티**: Loguru, Cachetools
- **개발 환경**: Conda
- **코드 품질**: Black, Ruff, Mypy

## 🏁 시작하기

### 1. 전제 조건

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 또는 Anaconda가 설치되어 있어야 합니다.
- CUDA 호환 NVIDIA GPU 및 관련 드라이버 (AI/ML 기능 사용 시)

### 2. 환경 설정

1.  **Conda 환경 생성 및 활성화**:
    프로젝트 루트 디렉토리에서 다음 명령을 실행하여 `environment.yml` 파일에 정의된 모든 의존성을 포함하는 Conda 가상 환경을 생성합니다.

    ```bash
    conda env create -f environment.yml
    conda activate binpicking_env
    ```

2.  **설정 파일 구성**:
    - `app/config/` 디렉토리 안에 다음 설정 파일들을 생성해야 합니다.
    - `.env`: 카메라 API 주소 등 환경별 설정을 정의합니다. (`.env.example` 파일을 참고하세요.)
    - `aruco_place.csv`: ArUco 보드를 구성하는 각 마커의 3D 좌표를 정의합니다.
    - `robot_position.json`: ArUco 보드 좌표계 기준의 로봇 베이스 위치를 정의합니다.

### 3. 서버 실행

프로젝트 루트 디렉토리에서 다음 명령을 실행하여 Uvicorn으로 FastAPI 서버를 시작합니다.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 52000 --reload
```

- `--host 0.0.0.0`: 모든 네트워크 인터페이스에서 접속을 허용합니다.
- `--port 52000`: 서버가 사용할 포트를 지정합니다.
- `--reload`: 코드 변경 시 서버가 자동으로 재시작됩니다. (개발용)

서버가 성공적으로 실행되면, 웹 브라우저에서 `http://localhost:52000` 로 접속하여 메인 대시보드를 확인할 수 있습니다.

## 📚 API 및 WebSocket 문서

서버가 제공하는 모든 API 엔드포인트와 WebSocket 채널에 대한 자세한 정보는 아래 문서에서 확인할 수 있습니다.

- **[API 기능 정의서](./API_DOCUMENTATION.md)**
- **Swagger UI (자동 생성 문서)**: 서버 실행 후 `http://localhost:52000/docs` 에서 확인

## 📂 프로젝트 구조

```
.
├── app/
│   ├── api/                # HTTP API 엔드포인트 라우터
│   ├── config/             # 설정 파일 (aruco_place.csv 등)
│   ├── core/               # 핵심 로직 (EventBus, Config, Logging)
│   ├── dependencies.py     # 의존성 주입 설정
│   ├── schemas/            # Pydantic 데이터 모델 (DTO)
│   ├── services/           # 비즈니스 로직 (Camera, Aruco 등)
│   ├── static/             # 정적 파일 (HTML 템플릿 등)
│   ├── stores/             # 상태 관리 (ApplicationStore 및 핸들러)
│   ├── websockets/         # WebSocket 관련 로직
│   └── main.py             # FastAPI 애플리케이션 진입점
├── config/
│   └── .env                # 환경 변수 설정 파일
├── scripts/                # 보조 스크립트
├── API_DOCUMENTATION.md    # API 기능 정의서
└── environment.yml         # Conda 환경 의존성 파일
```
