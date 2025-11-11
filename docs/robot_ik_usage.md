# Robot IK API 사용 가이드

이 문서는 `RobotIkRequest` / `RobotIkDownwardRequest`를 사용하는 클라이언트 입장에서,  
그리퍼 길이와 방향을 어떻게 처리해야 하는지 한눈에 이해할 수 있도록 정리한 안내서입니다.

---

## 요청 기본 구조

### 일반 IK (`RobotIkRequest`)

```json
{
  "target_frame": "tool",
  "pose_targets": [
    {
      "translation": [x, y, z],
      "rotation_quaternion": [qx, qy, qz, qw]
    }
  ],
  "grip_offsets": [0.0],
  "initial_joint_positions": null,
  "max_iterations": 200,
  "tolerance": 0.0001,
  "damping": 0.6
}
```

| 필드 | 설명 |
| --- | --- |
| `pose_targets` | TCP(그리퍼 끝)가 도달해야 할 실제 좌표. 여러 개를 넣으면 첫 번째부터 순서대로 평가합니다. |
| `grip_offsets` | TCP와 플랜지 사이의 거리(미터). **+ 값**은 플랜지에서 TCP 방향(+Z)으로 그만큼 길이가 있다는 의미이며, 서버가 자동으로 플랜지를 뒤로 되돌립니다. 음수값은 필요할 경우 반대 방향으로 보정하고 싶을 때 사용할 수 있습니다. |
| `rotation_quaternion` | TCP가 가져야 하는 방향. WXYZ 순서가 아닌 `[x, y, z, w]` 순서를 사용합니다. |
| `initial_joint_positions` | 현재 관절값(동일한 길이의 배열). 제공하면 IK가 이 값 근처의 해를 우선 선택합니다. |

> **주의**: 현재 서버는 `ikpy` 엔진만 지원합니다. Pinocchio 관련 엔드포인트(`/ik/pinocchio`, `/ik/{backend}` 등)는 제거되었습니다.

### Downward IK (`RobotIkDownwardRequest`)

```json
{
  "target_frame": "tool",
  "translations": [
    [x, y, z]
  ],
  "hover_height": 0.10,
  "grip_offsets": [0.12],
  "mode": "auto"
}
```

| 필드 | 설명 |
| --- | --- |
| `translations` | 목표 위치 (플랜지 기준). 보통 보정하지 않은 TCP 목표를 넣으면 서버가 자동으로 처리합니다. |
| `hover_height` | 목표 지점 위에서 유지할 높이. 보통 접근 시 TCP가 물체 위에 잠시 떠 있도록 설정합니다. |
| `grip_offsets` | 일반 IK와 동일하게 TCP-플랜지 사이 거리. Downward 요청도 내부적으로 일반 IK로 변환할 때 이 값을 사용합니다. |

---

## 그리퍼 길이 처리 방식

### 1. 클라이언트 측 입력

1. **TCP가 닿아야 할 실제 좌표**를 `pose_targets.translation` (또는 Downward의 `translations`)에 넣습니다.
2. 그리퍼 끝(TCP)과 플랜지 사이의 길이를 **미터 단위**로 `grip_offsets` 배열에 넣습니다.
   - 예: 그리퍼가 12cm라면 `[0.12]`.
3. 서버는 이 값을 **자동으로** 도구 좌표계의 +Z 방향으로 해석하여, 플랜지를 되돌린 위치를 IK 목표로 사용합니다.

즉, 별도의 수학적 보정을 클라이언트가 직접 할 필요 없이, 항상 TCP 기준 좌표를 전송하면서 `grip_offsets`만 맞춰 주면 됩니다.

### 2. 서버 내부 동작 요약

1. API에서 클램프: Z 좌표가 `IK_MIN_Z` 미만이면 로그를 남기고 높이를 `IK_MIN_Z`까지 끌어올립니다.
2. ikpy IK 서비스에서 보정:
   - 요청된 목표 행렬에 대해 `grip_offset`을 적용합니다 (새 함수를 통해 TCP → 플랜지 방향으로 되돌림).
   - 보정된 행렬을 대상으로 IK를 수행하고, 지면 침투 검사도 함께 진행합니다.

---

## 요청/응답 예시

### (1) 기본 IK

```bash
curl -X POST http://HOST:PORT/api/robot/ik \
  -H "Content-Type: application/json" \
  -d '{
        "target_frame": "tool",
        "pose_targets": [
          {"translation": [0.3, 0.1, -0.01], "rotation_quaternion": [0, 0, 0, 1]}
        ],
        "grip_offsets": [0.12]
      }'
```

- 서버는 TCP 좌표 `[0.3, 0.1, -0.01]`를 `IK_MIN_Z` 이하라면 0까지 끌어올리고, `grip_offset` 0.12m를 적용해 플랜지가 있어야 할 위치로 되돌립니다.
- 응답에서 `best.joint_distance`가 작고 `best.joint_positions`가 정상 범위인지 확인합니다.

### (2) Downward IK

```bash
curl -X POST http://HOST:PORT/api/robot/ik/ikpy/downward \
  -H "Content-Type: application/json" \
  -d '{
        "target_frame": "tool",
        "translations": [
          [0.45, 0.05, 0.0]
        ],
        "hover_height": 0.10,
        "grip_offsets": [0.12]
      }'
```

- Downward 흐름도 동일하게 `[0.12]`를 적용해 플랜지를 뒤로 이동시킨 뒤 내부적으로 일반 IK 요청으로 변환합니다.
- `hover_height`는 원하는 만큼 지정하되, 목표 Z에 `IK_MIN_Z`가 적용될 수 있다는 점은 동일합니다.

---

## 주의 사항 & 팁

1. **`grip_offsets` 길이**: 여러 후보 길이를 넣으면 시드마다 다른 오프셋을 적용해 해를 탐색합니다. 하나만 사용하면 간단합니다.
2. **음수 오프셋**: 특수 상황에서 TCP를 플랜지 뒤쪽(-Z)으로 보내고 싶다면 음수를 넣을 수 있습니다. 보통은 양수만 사용합니다.
3. **현재 관절값 전달**: `initial_joint_positions`를 함께 보내면 IK가 그 값 주변 해를 선호합니다(관절 점프 방지).
4. **지면 제약**: `IK_MIN_Z`보다 낮은 목표는 자동으로 올라가며, 결과 관절도 지면을 뚫지 않도록 필터링됩니다. 모든 후보가 제거되면 400 에러가 반환될 수 있으니, 목표 높이를 재확인하세요.
5. **테스트/검증**: 로컬에서는 `pytest tests/test_robot_service_ikpy_offsets.py` 등을 실행해 보정 로직이 기대대로 동작하는지 검증할 수 있습니다.

---

필요 시 `.env`에서 `IK_MIN_Z`, `IK_JOINT_DISTANCE_WEIGHT`, `IKPY_END_EFFECTOR_FRAME` 등을 조정해 환경에 맞게 튜닝하세요.  
추가 요청이나 개선 사항이 있으면 문서와 함께 업데이트해 주시면 됩니다.

