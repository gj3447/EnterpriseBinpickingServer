# 로봇 IK API 사용 설명서

`POST /api/robot/ik` 엔드포인트를 통해 서버에서 Pinocchio 기반 역기구학(IK)을 실행할 수 있습니다. 이 문서는 요청 파라미터, 응답 구조, 사용 예시를 정리합니다.

---

## 1. 요청(Request)

### HTTP 메서드 & 엔드포인트
- `POST /api/robot/ik`

### 헤더
- `Content-Type: application/json`

### 요청 본문(JSON)

| 필드 | 타입 | 필수 | 기본값 | 설명 |
| --- | --- | --- | --- | --- |
| `target_frame` | `string` | ✔ | - | IK를 계산할 말단 프레임 이름 (예: `link_6`, `tool`). URDF에 존재하는 frame 이어야 함 |
| `pose_targets` | `PoseTarget[]` | ✔ | - | 목표 pose 목록. 여러 pose를 넣으면 각 pose에 대해 IK를 수행 후 가장 좋은 후보를 선택 |
| ├ `translation` | `float[3]` | ✔ | - | XYZ 위치 [m] |
| └ `rotation_quaternion` | `float[4]` | ✔ | - | Quaternion (x, y, z, w) |
| `grip_offsets` | `float[]` | ✖ | `[0.0]` | 그리퍼 길이 후보. 프리스매틱 관절 혹은 오프셋으로 취급되며 여러 값 제공 가능 |
| `mode` | `"auto" \| "fixed" \| "prismatic"` | ✖ | `"auto"` | IK 해결 시 관절을 고정(fixed), 프리스매틱(prismatic), 자동(auto: 프리스매틱 관절 존재 시 사용) 중 선택 |
| `coordinate_mode` | `"base" \| "custom"` | ✖ | `"base"` | 목표 pose를 해석할 좌표계 모드 |
| `custom_axes` | `CoordinateAxes` | 조건부 | - | `coordinate_mode="custom"`일 때 필수. `{ "up": "z", "forward": "x" }` 형식 |
| `initial_joint_positions` | `float[]` | ✖ | URDF neutral | 초기 관절값. 모델의 `nq` 길이와 일치해야 하며 수렴 개선에 사용됨 |
| `max_iterations` | `int` | ✖ | `80` | 최대 반복 횟수 (1~500) |
| `tolerance` | `float` | ✖ | `1e-4` | 종료 오차 기준 |
| `damping` | `float` | ✖ | `0.6` | 최소제곱 해를 적용할 때 곱해지는 감쇠 계수 |
| `urdf_variant` | `"fixed" \| "prismatic"` | ✖ | 기본 variant | 로드된 여러 URDF 중 어떤 variant로 IK를 계산할지 지정 (미지정 시 서버 기본값) |

#### `CoordinateAxes` 구조
```json
{
  "up": "z",
  "forward": "x"
}
```
- 허용 축 라벨: `x`, `-x`, `y`, `-y`, `z`, `-z`
- `forward`와 `up`은 서로 평행할 수 없습니다.

---

## 2. 응답(Response)

### 성공(HTTP 200)

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `mode` | `"fixed" \| "prismatic"` | 실제 사용된 그리퍼 모드 (`mode` 입력이 `auto`였을 때 서버가 결정한 값) |
| `urdf_variant_used` | `"fixed" \| "prismatic"` | IK 계산에 사용된 URDF variant |
| `has_gripper_joint` | `bool` | 해당 variant가 프리스매틱 관절을 포함하는지 여부 |
| `gripper_joint_name` | `string?` | 사용된 프리스매틱 관절 이름 (없으면 `null`) |
| `best` | `IkCandidateResult` | 모든 후보 중 가장 오차가 낮은 결과 |
| `candidates` | `IkCandidateResult[]` | pose × grip offset 조합으로 얻은 모든 후보 |

#### `IkCandidateResult`

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `pose_index` | `int` | `pose_targets` 배열에서 사용된 pose 인덱스 |
| `grip_offset` | `float` | 해당 후보에 사용된 그리퍼 offset 값 |
| `error` | `float` | `pin.log` 기반 6D 오차 벡터의 노름 (작을수록 목표 pose와 유사) |
| `iterations` | `int` | 수렴까지 수행된 반복 횟수 |
| `mode_used` | `"prismatic_joint" \| "offset"` | 프리스매틱 관절을 직접 사용했는지, offset 방식인지 |
| `coordinate_mode_used` | `"base" \| "custom"` | 실제 적용된 좌표계 모드 |
| `urdf_variant` | `"fixed" \| "prismatic"` | 해당 후보 계산에 사용된 URDF variant |
| `joint_positions` | `float[]` | 계산된 관절 위치(`model.nq` 길이). 프리스매틱 관절 포함 시 마지막 요소가 그리퍼 길이 |

### 실패 시 주요 HTTP 코드
- `400 Bad Request`: 잘못된 파라미터, `pose_targets` 비어 있음, Jacobian 특이(수렴 실패) 등
- `404 Not Found`: 요청한 `target_frame` 또는 지정한 variant가 존재하지 않음
- `500 Internal Server Error`: Pinocchio 로딩 실패 등 예상치 못한 오류

응답 본문에는 `detail` 키로 오류 메시지가 포함됩니다.

---

## 3. 사용 예시

### 3.1 프리스매틱 관절 + 다중 pose, 다중 그리퍼 길이
```json
POST /api/robot/ik
{
  "target_frame": "link_6",
  "pose_targets": [
    {"translation": [0.0, 0.0, 0.25], "rotation_quaternion": [0, 0, 0, 1]},
    {"translation": [0.05, 0.05, 0.30], "rotation_quaternion": [0, 0, 0, 1]}
  ],
  "grip_offsets": [0.0, 0.04, 0.08],
  "mode": "auto",
  "coordinate_mode": "base",
  "urdf_variant": "prismatic",
  "max_iterations": 80,
  "tolerance": 0.001
}
```
- 모든 pose × offset 조합에 대한 결과가 `candidates`에 포함됩니다.
- 프리스매틱 URDF 로딩이 되어 있으면 `mode`는 `prismatic`, `joint_positions` 마지막 요소가 그리퍼 관절 값.

### 3.2 고정 URDF + 사용자 정의 좌표계
```json
{
  "target_frame": "link_6",
  "pose_targets": [
    {"translation": [0.0, 0.25, 0.10], "rotation_quaternion": [0, 0, 0, 1]}
  ],
  "coordinate_mode": "custom",
  "custom_axes": {"up": "y", "forward": "x"},
  "mode": "fixed",
  "urdf_variant": "fixed",
  "max_iterations": 60,
  "tolerance": 0.001
}
```
- `coordinate_mode_used`가 `custom`으로 반환되고, 그리퍼 관절이 없으므로 항상 `mode_used`는 `offset`.

### 3.3 기본 variant(auto)로 단순 IK
```json
{
  "target_frame": "link_6",
  "pose_targets": [
    {"translation": [0.0, 0.0, 0.2], "rotation_quaternion": [0, 0, 0, 1]}
  ]
}
```
- `mode`, `urdf_variant_used`는 서버 기본 설정에 따라 자동 결정됩니다.

---

## 4. 주의 사항 및 팁

- **초기 관절값**: 실제 로봇 자세와 큰 차이가 나면 수렴이 어려울 수 있으므로 `initial_joint_positions`를 활용해 주세요.
- **오차 해석**: `error` 값은 URDF 기준 오차입니다. 실제 로봇 오차와 다를 수 있으므로 별도의 검증이 필요합니다.
- **좌표계 변환**: `custom_axes` 사용 시 up/forward 축이 서로 직교해야 하며, 지정한 축 조합이 잘못되면 예외가 발생합니다.
- **프리스매틱 사용 조건**: 프리스매틱 variant가 로드되고 `GRIPPER_JOINT_NAME`가 올바르게 설정돼 있어야 `mode='prismatic'`이 제대로 동작합니다.
- **응답 후보 해석**: `candidates` 배열 전체를 활용하면 pose/그리퍼 길이 선택을 클라이언트에서 커스터마이즈할 수 있습니다.

---

추가적인 검증(실제 로봇과의 오차 측정 등)이 필요하면 forward kinematics 결과나 실제 로봇 데이터를 결합한 후처리를 구현하는 것을 권장합니다.

---

## 5. 회전 기본값이 아래(-Z)인 간편 IK 엔드포인트

회전을 직접 계산하지 않고 말단 프레임의 Z축을 항상 로봇 베이스 `-Z` 방향(바닥)으로 향하게 하고 싶다면 `POST /api/robot/ik/downward`를 사용할 수 있습니다. 내부적으로는 `rotation_quaternion = [1, 0, 0, 0]`(X축 기준 180° 회전)을 적용한 뒤 기존 `/api/robot/ik` 로직을 그대로 호출합니다.

### 요청 본문(JSON)

| 필드 | 타입 | 필수 | 기본값 | 설명 |
| --- | --- | --- | --- | --- |
| `target_frame` | `string` | ✔ | - | IK를 계산할 말단 프레임 이름 |
| `translations` | `float[3][]` | ✔ | - | 위치만 지정한 목표 좌표 목록. 각 항목은 `[x, y, z]`(미터) |
| `grip_offsets` | `float[]` | ✖ | `[0.0]` | 기존 IK와 동일하게 여러 그리퍼 길이 후보 지정 가능 |
| `mode` | `"auto" \| "fixed" \| "prismatic"` | ✖ | `"auto"` | 기존 IK와 동일 |
| `coordinate_mode` | `"base" \| "custom"` | ✖ | `"base"` | 좌표계 모드. `custom`일 경우 `custom_axes` 필수 |
| `custom_axes` | `CoordinateAxes` | 조건부 | - | `coordinate_mode="custom"`일 때 축 정의 |
| `initial_joint_positions` | `float[]` | ✖ | URDF neutral | 초기 관절값. 길이는 `model.nq`와 일치해야 함 |
| `max_iterations` | `int` | ✖ | `80` | 최대 반복 횟수 |
| `tolerance` | `float` | ✖ | `1e-4` | 종료 오차 기준 |
| `damping` | `float` | ✖ | `0.6` | 최소제곱 해 감쇠 계수 |

### 응답

응답 구조는 `/api/robot/ik`와 동일하게 `RobotIkResponse`를 반환합니다.

> 참고: `custom_axes`를 사용할 경우 위치뿐만 아니라 내려꽂기 방향도 동일한 축 변환을 거칩니다. DT 좌표계를 사용할 때는 이 변환을 통해 로봇 베이스 좌표와 일치시켜 주세요.

### 사용 예시

```bash
POST /api/robot/ik/downward
Content-Type: application/json

{
  "target_frame": "link_6",
  "translations": [
    [-0.45, 0.12, 0.25],
    [-0.50, 0.10, 0.28]
  ],
  "mode": "auto",
  "grip_offsets": [0.0, 0.05],
  "initial_joint_positions": [
    -0.20, 0.40, 0.10, -1.20, 0.05, 1.57, 0.02
  ]
}
```

위 요청은 두 개의 위치 후보를 내려꽂기 자세(말단 Z축이 `-Z`)로 IK 계산합니다. `initial_joint_positions`를 제공하면 현재 로봇 자세 근처에서 탐색을 시작하여 수렴성이 좋아집니다.

```json
HTTP/1.1 200 OK
{
  "mode": "prismatic",
  "urdf_variant_used": "prismatic",
  "has_gripper_joint": true,
  "gripper_joint_name": "gripper_extension_joint",
  "best": {
    "pose_index": 1,
    "grip_offset": 0.05,
    "error": 0.00042,
    "iterations": 34,
    "mode_used": "prismatic_joint",
    "coordinate_mode_used": "base",
    "urdf_variant": "prismatic",
    "joint_positions": [
      -0.19,
      0.45,
      0.06,
      -1.15,
      0.06,
      1.62,
      0.05
    ]
  },
  "candidates": [...]
}
```

> **Tip**: 응답의 `best.pose_index`는 어떤 translation 후보가 선택되었는지 알려주며, `grip_offset`은 활성화된 그리퍼 길이 값을 의미합니다. 필요하다면 `candidates` 배열 전체를 살펴보고 클라이언트에서 선택 기준을 재정의할 수 있습니다.

