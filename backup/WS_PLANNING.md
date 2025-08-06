# 웹소켓 구현 계획서 (최종, 안정성 강화)

## 1. 핵심 목표

카메라에서 이미지가 들어올 때마다(`IMAGE_RECEIVED` 이벤트 발생 시), 다음 두 가지 정보를 실시간으로 웹소켓 클라이언트에게 전송한다.

1.  **이미지 스트림:** `images.py` 엔드포인트에서 제공하는 모든 종류의 이미지.
2.  **좌표 변환 정보:** `transforms.py` 엔드포인트에서 제공하는 모든 Pose 정보.

## 2. 아키텍처: 서비스 계층과 API 계층 분리

-   **서비스 계층 (공용):** 웹소켓의 핵심 로직을 담당하는 `ConnectionManager`와 `StreamingService`는 `app/websockets/`에 위치시킨다.
-   **API 계층 (버전 관리):** 클라이언트가 접속하는 엔드포인트는 `app/api/v1/endpoints/websockets.py`에 위치시켜 일관성을 유지한다.

## 3. 구현 단계

### 1단계: 안정성이 강화된 연결 관리자 (`ConnectionManager`)

-   **목적:** 다수의 웹소켓 연결과 구독 상태를 **안정적으로** 관리한다.
-   **파일 위치:** `app/websockets/connection_manager.py`
-   **주요 책임:**
    -   구독자 관리: `subscriptions: Dict[str, Set[WebSocket]]` 자료구조로 스트림별 구독자를 관리한다.
    -   **불안정한 클라이언트 처리:** 데이터 전송 실패 시 해당 클라이언트를 "죽은 연결"로 간주하고 즉시 구독 목록에서 제거하여, **다른 정상 클라이언트에게 영향이 가지 않도록 격리**한다.
-   **핵심 메서드:**
    -   `connect(stream_id: str, websocket: WebSocket)`: 클라이언트를 구독자로 추가.
    -   `disconnect(stream_id: str, websocket: WebSocket)`: 클라이언트를 구독자 목록에서 제거.
    -   `broadcast_bytes(stream_id: str, data: bytes)`: 이미지 스트림 구독자에게 `bytes` 데이터를 방송. 개별 전송을 `try-except`로 감싸 안정성 확보.
    -   `broadcast_text(stream_id: str, data: str)`: 좌표 정보 등 텍스트 스트림 구독자에게 `JSON 문자열`을 방송. 개별 전송을 `try-except`로 감싸 안정성 확보.

### 2단계: 중앙 스트리밍 서비스 (`StreamingService`)

-   **목적:** 이벤트 수신, 데이터 생성/계산, 방송 요청 등 웹소켓 로직 총괄.
-   **파일 위치:** `app/websockets/streaming_service.py`
-   **주요 책임:**
    -   **이벤트 구독:** `event_bus`에서 `IMAGE_RECEIVED` 이벤트를 구독.
    -   **데이터 일괄 생성/계산:** 이벤트 발생 시, `ImageService`와 `ArucoService`를 호출하여 이미지 데이터와 좌표 변환 정보를 모두 생성.
    -   **타입에 맞는 방송 요청:** 생성된 데이터의 종류에 따라 `ConnectionManager`의 `broadcast_bytes` (이미지) 또는 `broadcast_text` (좌표 정보) 메서드를 호출.

### 3단계: 웹소켓 API 라우터 (`websockets.py`)

-   **목적:** 클라이언트가 접속할 실제 WebSocket 엔드포인트를 정의.
-   **파일 위치:** `app/api/v1/endpoints/websockets.py`
-   **주요 책임:** `/ws/{stream_id}` 엔드포인트를 통해 클라이언트의 연결/해제 요청을 받아 `ConnectionManager`에 전달.

### 4단계: 시스템 통합

-   **`app/dependencies.py`:** `ConnectionManager`와 `StreamingService`의 싱글턴 인스턴스를 의존성으로 주입.
-   **`app/main.py`:** 서버 시작 시 `StreamingService`의 백그라운드 태스크 실행.
-   **`app/api/v1/router.py`:** `websockets.py`의 라우터를 메인 API 라우터에 포함.

---

### **※ 중요 주의사항 (잠재적 위험)**

-   **순환 참조 (Circular Dependency):** `app/dependencies.py`에서 애플리케이션 전역 서비스를 생성할 때, 서비스 간의 의존성 순서를 반드시 지켜야 한다. 예를 들어, `StreamingService`는 `ConnectionManager`, `ImageService`, `ArucoService` 등을 필요로 하므로, **`StreamingService`가 생성되기 전에 이들이 먼저 생성되어 있어야 한다.** 이 순서가 지켜지지 않을 경우, 애플리케이션 시작 시 순환 참조 오류가 발생할 수 있다. 이 부분은 시스템 통합 단계에서 가장 주의 깊게 처리해야 할 항목이다.
