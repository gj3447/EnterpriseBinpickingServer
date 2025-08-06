# 웹소켓 지연 시간 개선 및 최적화 계획

## 1. 문제 진단: 왜 "살짝 느리게" 느껴지는가?

현재 웹소켓 시스템은 기능적으로는 동작하지만, 아래 두 가지의 잠재적인 성능 저하 요인을 가지고 있다.

### 1.1. 데이터 처리의 "엇박자" (1프레임 지연)

이것이 체감상 지연의 가장 큰 원인이다. 이벤트 처리 흐름이 병렬적으로 이루어져 데이터의 정합성이 깨질 수 있다.

-   **AS-IS 흐름:**
    1.  `CameraService`가 `IMAGE_RECEIVED` 이벤트를 발행한다.
    2.  `ArucoService`와 `StreamingService`가 이 이벤트를 **동시에** 수신한다.
    3.  `ArucoService`는 이미지 분석 후 `ApplicationStore`에 **결과를 쓰는 중**이다.
    4.  `StreamingService`는 **거의 같은 시점에** `ApplicationStore`에서 **결과를 읽으려** 시도한다.

-   **결과:** `StreamingService`가 **새 이미지**와 **이전 프레임의 분석 결과**를 조합하여 방송할 가능성이 매우 높다. 이로 인해 1프레임의 데이터 지연이 발생한다.

### 1.2. 불필요한 연산 수행

-   **문제점:** `StreamingService`는 현재 구독자가 없는 스트림에 대해서도, `IMAGE_RECEIVED` 이벤트가 발생할 때마다 **매번 모든 종류의 이미지와 좌표 변환 데이터를 생성**하고 있다.
-   **영향:** 이 불필요한 연산이 지속적으로 CPU 자원을 낭비하여, 시스템 전체의 반응성을 미세하게 저하시킨다.

## 2. 해결 전략

### 2.1. 이벤트 흐름 재구성: 데이터 처리 순서 보장

이벤트 체이닝(Event Chaining)을 통해 데이터 처리 순서를 명확하게 보장한다.

-   **TO-BE 흐름:**
    1.  `CameraService` → `IMAGE_RECEIVED` 이벤트 발행
    2.  → `ArucoService`가 수신하여 모든 분석 완료
    3.  → `ArucoService`가 분석 결과를 담아 **`ANALYSIS_COMPLETE`** 이벤트 발행
    4.  → `StreamingService`가 수신하여 최신 분석 결과로 데이터 생성 및 방송

-   **기대 효과:** `StreamingService`는 항상 최신 분석 결과로 작업을 시작하는 것이 보장되어 1프레임 지연 문제가 **원천적으로 해결**된다.
-   **※ 중요:** 이 방식은 **`event_bus.py` 자체의 수정이 전혀 필요 없다.** 단지 이벤트를 발행하고 구독하는 서비스의 로직만 변경하면 된다.

### 2.2. 구독 기반 'On-Demand' 데이터 생성

-   **개념:** `StreamingService`가 이미지나 좌표 데이터를 생성하기 전에, `ConnectionManager`에 해당 스트림의 구독자가 있는지 먼저 확인한다.
-   **기대 효과:** 구독자가 있는 스트림의 데이터만 생성하므로, **불필요한 연산을 완전히 제거**하여 CPU 사용률을 최적화하고 반응 속도를 향상시킨다.

## 3. 상세 구현 계획

1.  **`ArucoService` 수정:**
    -   `detect_and_update_store` 메서드의 마지막 부분에, 분석 결과를 담아 `event_bus.publish("ANALYSIS_COMPLETE", ...)`를 호출하는 로직을 **추가**한다.
2.  **`StreamingService` 수정:**
    -   구독 이벤트를 `IMAGE_RECEIVED`에서 `ANALYSIS_COMPLETE`로 **변경**한다.
    -   `ConnectionManager`에 구독자 확인 로직을 **추가**하고, 이를 사용하여 데이터 생성 여부를 결정한다.
3.  **`ConnectionManager` 수정:**
    -   `has_subscribers(stream_id: str) -> bool` 메서드를 **추가**한다.

---

## 4. 기존 코드에 대한 안정성 보장 (대대적인 수정이 아닌 이유)

이 개선 작업은 **기존 API 및 서비스의 동작에 영향을 주지 않는 안전한 변경**이다. 그 이유는 다음과 같다.

-   **① '데이터 중앙 저장소' 원칙 유지:**
    -   모든 API(HTTP)의 데이터 소스인 `ApplicationStore`의 역할과 데이터 흐름은 **전혀 변경되지 않는다.**
    -   `ArucoService`는 여전히 분석 결과를 `ApplicationStore`에 정상적으로 업데이트하며, 다른 모든 서비스는 이전과 똑같은 방식으로 이 저장소에서 데이터를 읽어간다.

-   **② '추가' 기반의 변경:**
    -   `ArucoService`의 변경점은 기존 로직의 끝에 `ANALYSIS_COMPLETE` 이벤트를 **발행(publish)하는 코드를 한 줄 추가**하는 것뿐이다. 이는 기존의 데이터 저장 로직을 수정하는 것이 아닌, "분석이 끝났다"는 신호를 추가로 보내주는 것과 같다.

-   **③ 영향 범위의 완벽한 고립:**
    -   새로 추가된 `ANALYSIS_COMPLETE` 이벤트는 오직 `StreamingService`만이 구독(subscribe)하고 사용한다.
    -   따라서 이 변경의 영향 범위는 **오직 `ArucoService`와 `StreamingService` 두 파일로 완벽하게 한정**되며, 다른 어떤 API 엔드포인트나 서비스에도 영향을 미치지 않는다.

결론적으로, 이 작업은 기존 시스템의 안정적인 데이터 흐름은 그대로 유지하면서, **웹소켓의 실시간성 향상만을 목표로 하는 고립된 개선 작업**이므로 매우 안전하다.
