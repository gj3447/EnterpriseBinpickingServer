# 기술 부채 및 아키텍처 개선 목록

본 문서는 현재 프로젝트의 코드베이스에 존재하는 잠재적인 문제점(기술 부채)과, 이를 해결하기 위한 아키텍처 개선 방향을 정의합니다.

---

### 1. (가장 심각) 서비스 간 강한 결합과 순환 의존성 위험

-   **문제점**: `StreamingService`가 다른 서비스(`ArucoService`, `ImageService`)에 직접 의존하고 있어 결합도가 높습니다. 미래에 기능이 추가될 경우, 서비스 간의 양방향 호출이 필요한 상황(예: `ArucoService` -> `StreamingService`)이 발생하면 순환 참조 오류로 이어질 수 있습니다.
-   **위험성**: 기능 확장을 방해하고, 전체 아키텍처를 불안정하게 만드는 가장 큰 잠재적 위험입니다.
-   **개선 방향**: 주기적으로 상태를 "가져가는(Pull)" 현재 방식에서, 계산이 완료되면 "알려주는(Push)" **이벤트 기반 아키텍처(Event-Driven Architecture)**로 전환합니다.
-   **영향을 받는 주요 파일**:
    -   `app/services/streaming_service.py`: 다른 서비스를 직접 호출하는 부분이 문제입니다.
    -   `app/services/aruco_service.py`: 상태를 업데이트한 뒤 이벤트를 발행(publish)하는 역할로 변경되어야 합니다.
    -   `app/dependencies.py`: 서비스 간의 직접적인 의존성 주입 구조가 변경될 수 있습니다.

### 2. "신기루" 상태 (Phantom State) 문제

-   **문제점**: 각 서비스(`ArucoService`, `StreamingService` 등)가 독립적인 `asyncio.sleep()` 주기로 동작합니다. 이로 인해, 한 서비스가 처리하는 데이터와 다른 서비스가 처리하는 데이터 간에 시간적 불일치가 발생할 수 있습니다.
-   **위험성**: 실시간 제어 시스템에서 클라이언트가 보는 정보와 서버의 실제 최신 상태가 달라, 미세한 오차나 잘못된 판단을 유발할 수 있습니다.
-   **개선 방향**: 이벤트 기반 아키텍처(위 1번 문제의 해결책)를 도입하면 이 문제가 자연스럽게 해결됩니다. `ArucoService`가 계산을 완료하고 이벤트를 발행하는 시점에, `StreamingService`가 그 최신 데이터를 즉시 클라이언트에게 전송하게 되므로 상태 불일치 시간이 최소화됩니다.
-   **영향을 받는 주요 파일**:
    -   `app/services/aruco_service.py`: 주기적인 `_periodic_detection` 루프의 구조.
    -   `app/services/streaming_service.py`: 주기적인 `_stream_loop`의 구조.

### 3. 설정 관리의 불완전성

-   **문제점**: 서비스(`ArucoService`)가 자신의 의존성(로봇 위치)을 설정 파일로부터 직접 읽어옵니다. 이는 서비스가 파일 시스템이라는 외부 환경에 직접 의존하게 만듭니다.
-   **위험성**: 단위 테스트(Unit Test) 작성을 매우 어렵게 만듭니다. `ArucoService`를 테스트하려면 항상 실제 설정 파일이 있어야만 합니다.
-   **개선 방향**: 의존성 역전 원칙(Dependency Inversion Principle)에 따라, `ArucoService`는 `Pose` 객체를 직접 주입받아야 합니다. 설정 파일을 읽어서 `Pose` 객체를 생성하는 책임은 애플리케이션의 최상위 구성 계층인 `dependencies.py`로 이전해야 합니다.
-   **영향을 받는 주요 파일**:
    -   `app/services/aruco_service.py`: `__init__` 메서드와 `_load_robot_pose` 메서드.
    -   `app/dependencies.py`: `ArucoService` 인스턴스를 생성하는 부분.

### 4. 하드코딩된 물리적 파라미터

-   **문제점**: 시스템의 중요한 물리적 상수(예: 단일 마커의 크기)가 코드 내에 하드코딩되어 있습니다.
-   **위험성**: 다른 사양의 마커를 사용하는 새로운 시스템을 구성해야 할 때, 코드 자체를 수정해야 합니다. 이는 OCP(개방-폐쇄 원칙)를 위반하며 유연성을 저해합니다.
-   **개선 방향**: 모든 물리적 파라미터를 중앙 설정 시스템(`app/core/config.py`)으로 옮기고, 서비스는 이 설정 값을 주입받아 사용하도록 리팩토링합니다.
-   **영향을 받는 주요 파일**:
    -   `app/services/aruco_service.py`: `self.single_marker_size_meters = 0.05` 부분.
    -   `app/core/config.py`: 새로운 설정 항목을 추가해야 합니다.
    -   `config/.env.example`: 새로운 설정 항목에 대한 예시를 추가해야 합니다.
