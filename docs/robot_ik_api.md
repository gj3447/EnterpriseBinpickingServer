# IKPy Downward IK API 사용 가이드

## 개요

IKPy 백엔드에서 역기구학을 계산하기 위한 HTTP 엔드포인트 요약입니다. 현재 서비스는 고정형 URDF(`a0509.urdf`)를 기반으로 하고, 요청 시 자동으로 그리퍼 길이 보정과 말단 프레임 변경을 적용합니다.

- 기본 URL: `http://<서버주소>:53000`
- 엔드포인트: `/api/robot/ik/ikpy/downward`
- 메서드: `POST`
- 콘텐츠 타입: `application/json`
- 응답 스키마: [`RobotIkResponse`](#응답-구조)


## 요청 구조

```json
{
  "target_frame": "tool",
  "translations": [[-0.41, 0.30, 0.21], ...],
  "hover_height": 0.04,
  "grip_offsets": [0.0],
  "mode": "auto",
  "coordinate_mode": "base",
  "urdf_variant": "fixed",
  "custom_axes": {
    "up": "z",
    "forward": "x"
  },
  "initial_joint_positions": [0.0, ...],
  "max_iterations": 200,
  "tolerance": 0.0001,
  "damping": 0.6
}
```


### 필드 설명

| 필드 | 필수 | 타입 | 설명 |
| --- | --- | --- | --- |
| `target_frame` | 예 | `string` | 역기구학을 계산할 말단 프레임 이름. 기본 `tool` |
| `translations` | 예 | `List[List[float]]` | 목표 위치 좌표 목록. 각 좌표는 `[x, y, z]` 형태 |
| `hover_height` | 아니오 | `float` | 목표 지점 위에서 유지할 높이 (m). 0 이상 |
| `grip_offsets` | 아니오 | `List[float]` | 추가 그리퍼 오프셋(단위 m). 미지정 시 `[0.0]` |
| `mode` | 아니오 | `"auto" \| "fixed" \| "prismatic"` | IK 계산 모드. IKPy는 내부적으로 `fixed` 모드로 동작 |
| `coordinate_mode` | 아니오 | `"base" \| "custom"` | 좌표계 모드. `custom`이면 `custom_axes` 필수 |
| `urdf_variant` | 아니오 | `"fixed" \| "prismatic"` | 사용할 URDF 변형. IKPy 경로에서는 자동으로 `fixed`로 강제 |
| `custom_axes` | 아니오 | `{ "up": str, "forward": str }` | 사용자 정의 축. 축은 `x, -x, y, -y, z, -z` 중 하나 |
| `initial_joint_positions` | 아니오 | `List[float]` | 초기 관절값. 미전달 시 중립자세 또는 마지막 성공 자세를 시드로 사용 |
| `max_iterations` | 아니오 | `int` | IKPy 반복 최대 횟수. 기본 200 |
| `tolerance` | 아니오 | `float` | 허용 오차 (rad). IKPy 함수에 직접 전달되지는 않지만 로그/튜닝용으로 유지 |
| `damping` | 아니오 | `float` | 댐핑 파라미터. 현재 IKPy 경로에서는 사용하지 않음 |


### 내부 보정

요청은 `_build_downward_request_for_ikpy()`를 통해 아래와 같이 보정됩니다.

- `urdf_variant`를 강제로 `fixed`로 설정합니다.
- `target_frame`을 `settings.IKPY_END_EFFECTOR_FRAME`(기본 `link_6`)으로 변경합니다.

즉, 호출자는 플랜지 기준 좌표만 전달하면 되고, 서비스는 말단 프레임만 맞춰서 IKPy에 전달합니다. 그리퍼 길이는 이제 자동으로 더하지 않습니다.


## 응답 구조

`RobotIkResponse` 스키마는 다음과 같습니다.

```json
{
  "best": {
    "pose_index": 0,
    "grip_offset": 0.0,
    "error": 0.000123,
    "iterations": 0,
    "mode_used": "offset",
    "coordinate_mode_used": "base",
    "urdf_variant": "fixed",
    "joint_positions": [0.1, -0.3, ...]
  },
  "candidates": [...],
  "mode": "fixed",
  "urdf_variant_used": "fixed",
  "has_gripper_joint": false,
  "gripper_joint_name": null
}
```

### 주요 필드 설명

- `best`
  - IKPy가 찾은 최적 후보.
  - `pose_index`: 입력 포즈의 인덱스.
  - `error`: 위치 및 회전 오차.
  - `joint_positions`: 액티브 관절만 추출된 조인트 배열.
- `candidates`
  - 모든 시드와 그립 오프셋 조합에서 생성된 후보 목록.
- `mode`
  - 실제 사용 모드. IKPy 경로에서는 `fixed`.
- `urdf_variant_used`
  - IK 계산에 사용된 URDF 변형.
- `has_gripper_joint`, `gripper_joint_name`
  - 현재 IKPy 변형에는 프리즘틱 그리퍼가 없으므로 `false`, `null`.


## 사용 예시

```python
import requests

payload = {
    "target_frame": "tool",
    "translations": [[-0.41, 0.30, 0.21]],
    "hover_height": 0.04,
    "mode": "auto",
    "coordinate_mode": "base"
}

resp = requests.post(
    "http://192.168.0.196:53000/api/robot/ik/ikpy/downward",
    json=payload,
    timeout=8.0,
)
resp.raise_for_status()
print(resp.json())
```

```powershell
$payload = @{
    target_frame    = "tool"
    translations    = @(@(-0.41, 0.30, 0.21))
    hover_height    = 0.04
    mode            = "auto"
    coordinate_mode = "base"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://192.168.0.196:53000/api/robot/ik/ikpy/downward" `
    -Body $payload `
    -ContentType "application/json"
```


## 참고사항

- **워크스페이스 초과**: URDF 기반으로 추정한 작업 공간 반경을 벗어나면 경고 로그를 남기지만, IK 계산은 시도합니다.
- **시드 전략**: 최근 성공 자세, 중립 자세, 랜덤 노이즈를 조합하여 초기 시드를 생성합니다.
- **로그 레벨**: `LOG_LEVEL=INFO`에서도 포즈별 IK 진행 상황이 찍히도록 구성했습니다.
- **테스트 스크립트**: `scripts/test_robot_downward_api.py --backend ikpy`를 사용해 대량 포인트 테스트 가능.

# IKPy Downward IK API 사용 가이드

## 개요

IKPy 백엔드에서 역기구학을 계산하기 위한 HTTP 엔드포인트를 정리한 문서입니다. 현재 서비스는 고정형 URDF(`a0509.urdf`)를 기반으로 하고, 요청 시 자동으로 그리퍼 길이 보정과 말단 프레임 변경을 적용합니다.

- 기본 URL: `http://<서버주소>:53000`
- 엔드포인트: `/api/robot/ik/ikpy/downward`
- 메서드: `POST`
- 콘텐츠 타입: `application/json`
- 응답 스키마: [`RobotIkResponse`](#응답-구조)


## 요청 구조

```json
{
  "target_frame": "tool",
  "translations": [[-0.41, 0.30, 0.21], ...],
  "hover_height": 0.04,
  "grip_offsets": [0.0],
  "mode": "auto",
  "coordinate_mode": "base",
  "urdf_variant": "fixed",
  "custom_axes": {
    "up": "z",
    "forward": "x"
  },
  "initial_joint_positions": [0.0, ...],
  "max_iterations": 200,
  "tolerance": 0.0001,
  "damping": 0.6
}
```


### 필드 설명

| 필드 | 필수 | 타입 | 설명 |
| --- | --- | --- | --- |
| `target_frame` | 예 | `string` | 역기구학을 계산할 말단 프레임 이름. 기본은 `tool`
| `translations` | 예 | `List[List[float]]` | 목표 위치 좌표 목록. 각 좌표는 `[x, y, z]` 형태 |
| `hover_height` | 아니오 | `float` | 목표 지점 위에서 유지할 높이 (m). 0 이상 |
| `grip_offsets` | 아니오 | `List[float]` | 추가 그리퍼 오프셋(단위 m). 미지정 시 `[0.0]` |
| `mode` | 아니오 | `"auto" \| "fixed" \| "prismatic"` | IK 계산 모드. IKPy는 내부적으로 `fixed` 모드로 동작 |
| `coordinate_mode` | 아니오 | `"base" \| "custom"` | 좌표계 모드. `custom`이면 `custom_axes` 필수 |
| `urdf_variant` | 아니오 | `"fixed" \| "prismatic"` | 사용할 URDF 변형. IKPy 경로에서는 자동으로 `fixed`로 강제 |
| `custom_axes` | 아니오 | `{ "up": str, "forward": str }` | 사용자 정의 축. 각 축은 `x, -x, y, -y, z, -z` 중 하나 |
| `initial_joint_positions` | 아니오 | `List[float]` | 초기 관절값. 미전달 시 중립자세 또는 마지막 성공자세를 시드로 사용 |
| `max_iterations` | 아니오 | `int` | IKPy 반복 최대 횟수. 기본 200 |
| `tolerance` | 아니오 | `float` | 허용 오차 (rad). IKPy에서 직접 사용하지 않지만, 향후 튜닝용으로 유지 |
| `damping` | 아니오 | `float` | 댐핑 파라미터. 현재 IKPy 경로에서는 사용하지 않음 |


### 내부 보정

요청은 `_build_downward_request_for_ikpy()`를 통해 아래와 같이 보정됩니다.

- `hover_height`에 `settings.IKPY_GRIPPER_LENGTH`(기본 0.12m)를 추가.
- `urdf_variant`를 강제로 `fixed`로 설정.
- `target_frame`을 `settings.IKPY_END_EFFECTOR_FRAME`(기본 `link_6`)으로 변경.

즉, 호출자는 플랜지 기준 좌표만 전달하면 되고, 서비스가 자동으로 그리퍼 길이와 프레임 변환을 적용해 줍니다.


## 응답 구조

`RobotIkResponse` 스키마는 다음과 같습니다.

```json
{
  "best": {
    "pose_index": 0,
    "grip_offset": 0.0,
    "error": 0.000123,
    "iterations": 0,
    "mode_used": "offset",
    "coordinate_mode_used": "base",
    "urdf_variant": "fixed",
    "joint_positions": [0.1, -0.3, ...]
  },
  "candidates": [...],
  "mode": "fixed",
  "urdf_variant_used": "fixed",
  "has_gripper_joint": false,
  "gripper_joint_name": null
}
```

### 주요 필드 설명

- `best`
  - IKPy가 찾은 최적 후보.
  - `pose_index`: 입력 포즈의 인덱스.
  - `error`: 위치 및 방향 오차(미터와 라디안 조합).
  - `joint_positions`: 액티브 관절만 추출된 조인트 배열.
- `candidates`
  - 모든 시드/그립 오프셋에 대해 생성된 후보 목록.
- `mode`
  - 실제로 사용한 모드. IKPy 경로에서는 `fixed`.
- `urdf_variant_used`
  - IK 계산에 사용한 URDF 변형 (`fixed`).
- `has_gripper_joint`, `gripper_joint_name`
  - 현재 IKPy 변형에는 프리즘틱 그리퍼가 없으므로 `false`, `null`.


## 사용 예시

### Python (requests)

```python
import requests

payload = {
    "target_frame": "tool",
    "translations": [[-0.41, 0.30, 0.21]],
    "hover_height": 0.04,
    "mode": "auto",
    "coordinate_mode": "base"
}

resp = requests.post(
    "http://192.168.0.196:53000/api/robot/ik/ikpy/downward",
    json=payload,
    timeout=8.0,
)
resp.raise_for_status()
print(resp.json())
```

### PowerShell

```powershell
$payload = @{
    target_frame   = "tool"
    translations   = @(@(-0.41, 0.30, 0.21))
    hover_height   = 0.04
    mode           = "auto"
    coordinate_mode = "base"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://192.168.0.196:53000/api/robot/ik/ikpy/downward" `
    -Body $payload `
    -ContentType "application/json"
```


## 참고 사항

- **워크스페이스 초과**: URDF 기반으로 추정한 작업 공간 반경을 벗어나면 경고 로그를 남기지만, IK 계산은 시도합니다.
- **시드 전략**: 최근 성공 자세, 중립 자세, 랜덤 노이즈 등을 조합하여 초기 시드를 생성합니다.
- **로그 레벨**: `LOG_LEVEL=INFO` 상태에서도 포즈별 IK 진행 상황이 로그로 찍히도록 구성되어 있습니다.
- **테스트 스크립트**: `scripts/test_robot_downward_api.py --backend ikpy`를 사용해 일괄 검증 가능.


