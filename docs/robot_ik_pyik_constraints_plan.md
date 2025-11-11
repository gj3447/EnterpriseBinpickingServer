# pyik(ikpy) IK 제약 개선 계획

## 개요

현재 `RobotServiceIkpy` 기반 IK 구현은 아래 두 문제로 인해 실제 기구가 불안정하게 움직일 수 있습니다.

1. **최소 오차 해만 선택** → 현재 관절 상태와 동떨어진 대칭 해가 선택돼 큰 동작을 유발.
2. **Ground plane 제약 부재** → 링크가 Z=0 이하(바닥)로 내려가는 해도 그대로 반환.

본 문서는 **pyik(ikpy) 백엔드만을 대상으로** 위 문제를 해결하는 구현 계획을 정리합니다.

---

## 목표

| 목표 | 설명 |
| --- | --- |
| 관절 변위 최소화 | 목표 포즈를 만족하는 해 중, 현재 관절 상태에 가장 가까운 해를 우선 선택 |
| 지면 침투 방지 | IK 결과의 모든 링크가 Z=0 평면 위(또는 설정한 최소 높이)로 유지되도록 필터링 |

---

## 적용 범위

- 서비스: `app/services/robot_service_ikpy.py`
- 스키마: `app/schemas/robot.py` (필요 시 `IkCandidateResult` 확장)
- API 입력 검증: `app/api/v1/endpoints/robot.py`
- 테스트: `tests/test_robot_service_ik.py` 및 관련 도우미

서버 요청/응답 JSON 구조는 유지합니다. (필드 추가 시 backwards-compatible 하게 처리)

---

## 구현 상세

### 1. 관절 변위 최소화

1. **기준 관절값 확보**
   - `_solve_ik_sync()`에서 이미 `initial_full`을 계산함 (`initial_joint_positions` 또는 `last_known` → `neutral` 순).
   - 후보 평가 시 `initial_active = self._extract_active_positions(initial_full, active_indices)`로 Active DOF 기준 벡터를 준비.

2. **후보별 비용 계산**
   - `active_solution = self._extract_active_positions(solution, active_indices)` 이후,
     ```python
     joint_delta = np.linalg.norm(active_solution - initial_active)
     combined_cost = error + joint_cost_weight * joint_delta
     ```
   - `joint_cost_weight`는 설정값 또는 상수(초기 0.05~0.2)로 두고 실험을 통해 조정.

3. **최적 해 선택 조건 수정**
   - 기존 `if best_result is None or result.error < best_result.error` → `combined_cost` 비교.
   - 필요 시 `IkCandidateResult`에 `joint_distance: float` 필드 추가(응답 및 로깅 용도).

4. **로그/테스트**
   - `logger.debug`에 joint_delta, combined_cost를 기록해 튜닝 상황을 파악.
   - 테스트에서 `initial_joint_positions`를 제공했을 때 결과가 초기 관절과 큰 차이가 없는지 검증.

### 2. 지면(Z=0) 침투 방지

1. **헬퍼 함수 추가**
   ```python
   def _violates_ground(self, chain: Chain, configuration: np.ndarray, min_z: float = 0.0) -> bool:
       transforms = chain.forward_kinematics(configuration)
       for idx, transform in enumerate(transforms):
           z_value = transform[2, 3]
           if z_value < min_z - 1e-6:
               return True
       return False
   ```
   - `ikpy.chain.Chain.forward_kinematics`는 각 링크의 4x4 변환 행렬 리스트를 반환.
   - 필요 시 검사 대상 링크를 subset(예: 유효 조인트에 대응하는 링크)로 제한해 성능 확보.

2. **후보 필터링**
   - `_solve_ik_sync()`에서 `solution`을 얻은 직후 `_violates_ground` 호출.
   - Ground 위반 시 해당 seed를 `continue`로 건너뛰고 `pose_candidates_found = False` 유지.
   - Ground 위반 사실을 `logger.debug`로 로깅.

3. **요청 선행 검증**
   - `RobotIkRequest.pose_targets`의 Z 좌표가 `min_z`보다 낮으면 API 단계에서 400 반환.
     ```python
     for pose in request.pose_targets:
         if pose.translation[2] < settings.IK_MIN_Z:
             raise HTTPException(400, "Target pose lies below ground plane.")
     ```
   - Downward API(`RobotIkDownwardRequest`)에도 동일 검증 적용.
   - `settings.IK_MIN_Z` 등을 `app/core/config.py`에 추가하면 환경별 조정 가능.

4. **URDF 조인트 하한 재확인**
   - `_enforce_joint_limits`가 정상 작동하는지, URDF의 joint lower limits가 ground 계획과 충돌하지 않는지 점검.

### 3. 설정 값

| 이름 | 설명 | 기본값 제안 |
| --- | --- | --- |
| `IK_JOINT_DISTANCE_WEIGHT` | 관절 변위 비용 가중치 | 0.1 |
| `IK_MIN_Z` | 허용 최소 Z 값 | 0.0 |

설정 경로:
- `app/core/config.py`에 추가 → `.env`로 override 가능.
- 초기에는 상수로 구현 후, 테스트가 끝나면 설정화해도 무방.

### 4. 테스트 전략

| 테스트 | 목적 |
| --- | --- |
| `test_pyik_prefers_nearby_solution` | `initial_joint_positions` 제공 시 결과 관절 거리가 작아졌는지 확인 |
| `test_pyik_rejects_ground_violation` | Z<0 타깃을 넣어 ground 필터가 작동하는지 검증 |
| `test_pyik_ground_filter_all_links` | 모든 링크가 `min_z` 이상인지 확인 |
| `test_pyik_combined_cost_logging` *(선택)* | combined_cost 계산이 정상인지 로그/결과 비교 |

테스트 작성 시 주의:
- 체인 forward_kinematics가 반환하는 행렬 구조를 정확히 이해하고, 비교 오차(`np.allclose`)를 적절히 설정.
- Ground 필터가 성능 저하를 일으키지 않는지 많은 요청을 반복하는 부하 테스트도 권장.

---

## 구현 순서 제안

1. `joint_cost_weight`, `min_z` 기본 상수 정의 및 설정화 여부 결정
2. `_solve_ik_sync`에 combined cost 적용 및 best-result 로직 업데이트
3. `_violates_ground` 헬퍼 추가 및 Ground 필터링 도입
4. API 입력 검증 보완 (`RobotIkRequest`, `RobotIkDownwardRequest`)
5. 테스트 케이스 작성 및 실행
6. 로깅/튜닝 후, 실제 장비/시뮬레이터로 회귀 테스트

---

## 추가 고려 사항

- **성능 최적화**: Ground 검사 대상 링크 수를 최소화하거나 특정 iteration마다 검사하도록 최적화 가능.
- **Trajectory 보간**: IK 결과를 바로 실행하지 말고, 현재 관절 ↔ 최종 관절 사이를 보간하는 후처리를 추가하면 안전성이 높아짐.
- **문서 공유**: 본 계획 문서를 팀 내 공유 후, 구현 상세/파라미터 값에 대한 합의를 거쳐 개발 착수.

---

이 계획을 따라 적용하면, pyik(ikpy) 기반 IK 서비스가 현재 관절에 보다 친화적인 해를 선택하고, 로봇이 지면 아래로 내려가는 문제를 방지할 수 있습니다.

