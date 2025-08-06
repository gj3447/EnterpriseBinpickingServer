# Event Bus 시스템 개선 제안

## 1. 이벤트 페이로드의 데이터 중복 및 비효율성

**문제점:**
- 이벤트 페이로드에 무거운 이미지 데이터(`numpy` 배열)가 직접 포함되어 메모리 사용량 증가 및 시스템 부하를 유발합니다.
- `CameraService` -> `IMAGE_RECEIVED` (이미지 포함)
- `ArucoService` -> `ARUCO_UPDATE` (이미지 또 포함)

**개선 제안:**
- 이벤트 페이로드에는 이미지 데이터 자체 대신 **고유 식별자(예: 타임스탬프, 프레임 ID)**만 포함시킵니다.
- 실제 이미지 데이터는 `ApplicationStore`에 저장하고, 각 서비스는 이벤트로부터 받은 식별자를 사용해 Store에서 데이터를 조회하도록 변경합니다.

**수정 대상 파일 및 방법:**
- **`app/services/camera_service.py`**:
    - `_listen_to_stream` 메서드에서 `event_bus.publish` 호출 시, 이미지 데이터(`"image": image`) 대신 고유 식별자(예: `"timestamp": time.time()`)를 담아 발행하도록 수정합니다.
- **`app/services/aruco_service.py`**:
    - `handle_image_received` 메서드에서 이벤트 데이터의 `"timestamp"`를 사용하여 `store`에서 이미지를 가져오도록 로직을 변경합니다.
    - `detect_and_update_store` 메서드에서 `ARUCO_UPDATE` 및 `WS_*` 이벤트를 발행할 때, 이미지 데이터 대신 동일한 `"timestamp"`를 포함하도록 수정합니다.
- **`app/services/image_service.py`**:
    - `handle_image_received`, `handle_aruco_update` 두 핸들러 모두 이벤트 데이터의 `"timestamp"`를 이용해 `store`에서 이미지를 가져와 후속 처리를 하도록 변경합니다.
- **`app/stores/image_store.py` (및 관련 Store)**:
    - `update_image`가 `timestamp`와 이미지를 함께 저장하도록 수정하고, `get_image_by_timestamp(ts)`와 같이 타임스탬프로 이미지를 조회하는 새로운 메서드 추가를 고려합니다.

---

## 2. Pydantic 모델 부재로 인한 타입 안정성 부족

**문제점:**
- 이벤트 데이터 구조가 `Dict[str, Any]`로 정의되어 있어, 런타임 오류에 취약하고 가독성과 유지보수성이 떨어집니다.

**개선 제안:**
- 각 이벤트 페이로드를 위한 `Pydantic` 모델을 `app/schemas/events.py`에 정의하여 타입 안정성을 확보합니다.

**수정 대상 파일 및 방법:**
- **`app/schemas/events.py` (신규 생성)**:
    - `ImageReceivedPayload`, `ArucoUpdatePayload` 등 이벤트별 Pydantic 모델을 여기에 정의합니다.
- **`app/services/camera_service.py`**:
    - `event_bus.publish`를 호출하기 전, `ImageReceivedPayload` 모델의 인스턴스를 생성하고 `.model_dump()`(또는 Pydantic v1의 경우 `.dict()`)를 사용하여 데이터를 전달합니다.
- **`app/services/aruco_service.py` / `app/services/image_service.py`**:
    - 모든 이벤트 핸들러(`handle_*`) 시작 부분에서, 수신한 `event_data`(딕셔너리)를 해당 Pydantic 모델(예: `ImageReceivedPayload.model_validate(event_data)`)로 파싱하여 사용합니다.
    - 데이터를 발행할 때도 동일하게 Pydantic 모델을 사용하여 생성 후 전달합니다.

---

## 3. 복잡한 이벤트 처리 흐름

**문제점:**
- `ImageService`가 `IMAGE_RECEIVED`와 `ARUCO_UPDATE` 이벤트를 순차적으로 구독하여, 하나의 프레임 처리 흐름이 여러 핸들러에 분산되어 있어 추적이 복잡합니다.

**개선 제안:**
- 역할 분리와 병렬 처리의 이점이 있어 당장 수정은 필수적이지 않으나, 향후 리팩토링 시 고려할 수 있습니다.

**수정 대상 파일 및 방법 (리팩토링 시):**
- **`app/services/image_service.py`**: 여러 핸들러의 로직을 하나의 통합된 파이프라인으로 재구성합니다.
- **`app/core/event_type.py`**: `IMAGE_PROCESSED`와 같은 새로운 통합 이벤트를 정의할 수 있습니다.
- **`app/services/camera_service.py`, `app/services/aruco_service.py`**: 변경된 흐름에 맞게 이벤트 발행 로직을 수정합니다.

---

## 4. 하드코딩된 문자열 필터링

**문제점:**
- `ArucoService`에서 `"color_raw"`와 같이 스트림 ID를 문자열로 직접 비교하여 유연성이 떨어집니다.

**개선 제안:**
- 처리할 스트림 ID 목록을 설정 파일로 분리하여 관리합니다.

**수정 대상 파일 및 방법:**
- **`app/core/config.py`**:
    - `ARUCO_TARGET_STREAM_IDS: List[str] = ["color_raw"]`와 같이 새로운 설정 변수를 추가합니다.
- **`app/services/aruco_service.py`**:
    - `handle_image_received` 메서드 내의 `if` 조건을 `if event_data.get("stream_id") in settings.ARUCO_TARGET_STREAM_IDS:` 와 같이 변경하여 설정 값을 참조하도록 수정합니다.
