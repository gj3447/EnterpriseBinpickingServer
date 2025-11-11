# 로봇 IK 제약 개선 계획

> **주의:** 이 문서는 Pinocchio 백엔드 개선을 위한 기록이며, 현재 서버는 ikpy 백엔드만 활성화되어 있습니다.

## 배경

최근 역기구학(IK) 계산 결과가 **현재 관절 상태 대비 큰 변위**를 만들어 내거나, **링크가 지면(Z=0) 아래로 내려가는 자세**를 반환하는 문제가 발생했습니다.  
이는 Pinocchio 기반 IK가 오차 최소화만을 목표로 하고, 전역 제약(ground plane)을 고려하지 않기 때문입니다.

## 개선 목표

1. **관절 변위 최소화**  
   - 동일한 목표 포즈가 여러 해를 갖는 경우, 현재 관절 상태에 가장 가까운 해를 우선 선택합니다.
2. **지면 침투 방지**  
   - IK 결과에서 모든 링크가 Z=0 평면 위(또는 사용자 지정 임계값 위)에 있도록 보장합니다.

## 적용 범위 및 영향

- 서비스: `app/services/robot_service_pinocchio.py`
- 스키마: `app/schemas/robot.py` (`IkCandidateResult`에 추가 필드가 필요할 수 있음)
- API: `app/api/v1/endpoints/robot.py` (입력 검증 강화)
- 테스트: `tests/test_robot_service_ik.py` 등

## 설계 개요

### 1. 관절 변위 비용 추가

| 단계 | 설명 |
| --- | --- |
| 1 | `_solve_ik_sync()`에서 초기 기준 관절값 `initial_q`를 확보 (`initial_joint_positions` 활용) |
| 2 | 각 후보(`IkCandidateResult`) 생성 시 `joint_delta = ||q_candidate - initial_q||` 계산 |
| 3 | `combined_cost = error + weight * joint_delta` 방식을 도입해 `best_result` 선택 |
| 4 | 필요 시 `IkCandidateResult`에 `joint_distance` 필드를 추가해 응답/로그에서 확인 |

> **Weight 제안**: 0.05 ~ 0.2 범위를 실험해 가장 자연스러운 이동을 찾습니다.  
> **시드 구성**은 기존 로직(초기값 → 직전 결과 → 뉴트럴 → 랜덤)을 유지하되, 필요 시 랜덤 시드 수를 줄입니다.

### 2. 지면 침투 감지 및 차단

| 단계 | 설명 |
| --- | --- |
| 1 | `RobotServicePinocchio`에 `_violates_ground(model, q, min_z=0.0)` 헬퍼를 추가 |
| 2 | IK 반복이 끝난 후 후보 해 평가 전에 `_violates_ground`로 필터링 |
| 3 | `pose_target.translation[2] < min_z`인 요청은 사전에 거부하거나 경고 로그를 남기고 스킵 |
| 4 | (선택) 반복 과정에서 Z<min_z에 가까워질수록 페널티를 주는 보정 항 추가 |

구현 예시:

```python
def _violates_ground(self, model: pin.Model, q: np.ndarray, min_z: float = 0.0) -> bool:
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    for frame in model.frames:
        pose = data.oMf[frame.parent]
        if pose.translation[2] < min_z - 1e-6:
            return True
    return False
```

`_solve_ik_sync()` 내부:

```python
if self._violates_ground(model, q_candidate):
    logger.debug("Candidate rejected: ground collision")
    continue
```

### 3. API 입력 검증 강화

- `/api/robot/ik` 요청에서 `pose_targets`의 Z 좌표가 `min_z`보다 낮으면 `HTTP 400` 반환.
- Downward API(`RobotIkDownwardRequest`)도 동일하게 검사.

### 4. 테스트 계획

| 테스트 | 목적 |
| --- | --- |
| `test_ik_returns_nearby_solution` | `initial_joint_positions`를 제공했을 때 결과가 초기값과 가까운지 확인 |
| `test_ik_rejects_ground_collision` | Z<0 목표가 거부되거나 후보에서 제외되는지 확인 |
| `test_ground_constraint_all_links` | 결과 링크 위치가 항상 `min_z` 이상인지 검증 |

테스트 시나리오 팁:

- Pinocchio 모형을 이용해 임의의 관절값을 생성하고 `_violates_ground`가 정확히 동작하는지 체크.
- 랜덤 타겟 중 일부를 의도적으로 Z<0으로 만들어 예외 처리를 검증.

## 향후 고려 사항

- **동적 페널티**: Z=0에 가까워질수록 오차 가중치를 높이는 방식으로 더 부드러운 회피 가능.
- **Trajectory Planning**: 관절 해 사이를 선형/스플라인으로 보간해 실제 로봇에 보다 안전한 명령 제공.
- **URDF 한계**: URDF 조인트 하한을 0으로 두더라도 전체 기구의 궤적을 보장하지 못한다는 점을 문서화.

## 진행 순서 제안

1. 코드 수정 전, 해당 계획을 리뷰 받고 weight/min_z 값 합의
2. 비용 함수 및 ground 필터 구현
3. API 입력 검증 보완
4. 테스트 작성 및 실행
5. 로컬/시뮬레이터에서 실제 동작 확인 후 배포

---

위 로드맵을 기준으로 작업을 진행하면, 원하는 “현재 자세에 가까운 해 선택”과 “지면 침투 방지”를 안정적으로 달성할 수 있습니다.

