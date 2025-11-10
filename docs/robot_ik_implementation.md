# 로봇 IK 기능 구현 가이드

이 문서는 현재 코드베이스(`app/` 이하 FastAPI + Pinocchio 구조)에 맞춰 역기구학(IK) 기능을 실제로 도입하는 순서를 정리합니다.

## 0. 사전 체크
- Conda 환경에 `pinocchio`, `numpy`, `scipy`가 설치돼 있는지 확인 (`environment.yml` 반영 필요 시 업데이트)
- `settings.ROBOT_URDF_PATH`가 로드 가능한 URDF 파일을 가리키는지 확인
- `RobotService`가 서버 기동 시 정상적으로 URDF를 로드하는지 로그 확인

## 1. URDF 전략
1. 기본 URDF(그리퍼 고정)와 프리스매틱 조인트가 포함된 URDF 두 버전을 준비
   - `.env`에 아래 값을 추가해 선택적으로 로드
     - `ROBOT_URDF_PATH_FIXED` : 기본(고정) URDF 경로
     - `ROBOT_URDF_PATH_PRISMATIC` : 프리스매틱 URDF 경로(선택)
     - `ROBOT_URDF_MODE` : `fixed` / `prismatic` / `auto`
       - `fixed` → 항상 `ROBOT_URDF_PATH_FIXED`
       - `prismatic` → 프리스매틱 경로 존재 시 사용, 없으면 자동으로 fixed로 폴백
       - `auto` → 둘 다 존재하면 프리스매틱을 기본값으로, 없으면 fixed 로드
     - `GRIPPER_JOINT_NAME` : 프리스매틱 URDF에서 사용된 조인트 이름(예: `gripper_extension_joint`)
   - 서버 기동 시 두 경로가 모두 존재하면 **두 variant를 동시에 로드**하고, `ROBOT_URDF_MODE`에 따라 기본 variant 를 결정한다.
2. 서버 기동 시 `RobotService`가 선택된 URDF를 Pinocchio로 로드하고, 모델 객체에 `has_gripper_joint` 플래그를 기록 (존재 여부 판별)
3. `/api/robot/status` 및 `/api/robot/urdf?variant=...`에서 로딩된 variant 목록, 기본값, `has_gripper_joint` 등을 확인

## 2. RobotService 확장 (`app/services/robot_service.py`)
1. Pinocchio 모델을 가져와 IK를 수행하는 `solve_ik(...)` 메서드 추가
   - 입력: 목표 pose 목록(List), 초기 관절 상태(Optional), 최대 반복, 허용 오차, 가변 그리퍼 offset 목록 등
   - 내부:
     - `model = robot_object["model"]`, `data = model.createData()`
     - `pin.forwardKinematics` → `pin.updateFramePlacements`
     - `pin.log`로 오차 계산 후 최소제곱(`np.linalg.lstsq`)으로 `dq` 산출 → `pin.integrate`
     - 관절 한계 체크: `model.lowerPositionLimit`, `upperPositionLimit`
     - Pose × 그리퍼 offset 조합 반복 → 최적 해 선택 및 전체 결과 목록 반환
     - **프리스매틱 조인트가 존재하면서 길이를 외부에서 고정하고 싶다면** 해당 관절 index를 찾아 `q[index]=L`로 초기화하고, 반복 중에도 `dq[index]=0` 또는 Jacobian에서 해당 열을 제거해 관절 업데이트를 막는다.
     - **URDF에 프리스매틱 조인트가 없고 길이를 후보로 시험하고 싶다면** 목표 pose에 `pin.SE3(I, [0,0,L])` 같은 오프셋을 곱해 여러 길이를 순회한다.
   - 출력: 최적 joint 벡터, 잔류 오차, 반복 횟수, 선택된 Pose/offset, 후보별 상세 리스트
2. 예외 관리
   - Pinocchio 미로딩, 프레임 ID 미지정, 수렴 실패 등은 `RobotServiceError`(새로운 예외 클래스)로 감싸기

## 3. API 업데이트 (`app/api/v1/endpoints/robot.py`)
1. 기존 GET `/api/robot/urdf` 응답을 Pinocchio 딕셔너리 구조에 맞게 정리
   - 쿼리 파라미터 `variant` 로 `fixed`/`prismatic` 중 원하는 정보를 조회
2. POST `/api/robot/ik` 엔드포인트에서 IK 실행
   - 요청 모델에 `mode` 필드 외에 `urdf_variant`(선택)를 추가해 variant 를 지정 가능
   - 응답에는 `urdf_variant_used`가 포함되어 실제 사용된 모델을 확인할 수 있다.
   - 요청 모델에 `mode` 필드 추가 (`"fixed"`, `"prismatic"`, `"auto"` 등)
   - `robot_service.solve_ik` 호출 시, `mode`, 현재 URDF 상태(`has_gripper_joint`), `coordinate_mode`(예: `"base"`, `"custom"`)를 함께 전달
   - `coordinate_mode`가 `custom`이면 요청에 포함된 축 정의(예: `up_axis`, `forward_axis`)를 이용해 회전 행렬을 생성하고 pose를 변환한 뒤 IK 수행
   - 응답에는 joint 벡터, 잔류 오차, 반복 횟수와 함께 최종 선택된 `pose_index`/`grip_offset`/`mode`/`coordinate_mode`를 포함
   - 수렴 실패/모델 부재 시 HTTP 400/404 예외 처리

## 4. Pydantic 스키마 (`app/schemas/aruco.py` 또는 신규 파일)
1. HTTP 요청/응답 모델 정의
   - Pose 입력: 위치(float[3]) + 회전(quaternion or RPY), `mode`, `grip_offsets`, `coordinate_mode`, `custom_axes`(선택)
   - IK 응답: joint 배열, error, iterations, 선택된 pose index, grip offset, 후보 리스트, 실제 사용된 `mode`, `coordinate_mode`

## 5. 서비스 wiring
- `app/dependencies.py`는 이미 싱글턴 인스턴스를 제공하므로 별도 수정 불필요
- 단, 새 예외나 설정 값(예: IK 기본 모드, 기본 grip offset) 필요 시 `app/core/config.py`에 추가

## 6. 테스트
1. 단위 테스트(`tests/`)
   - Pinocchio 모형(간단한 URDF 또는 mock)을 이용해 IK가 수렴하는지 확인
   - 실패 케이스(도달 불가능한 pose, joint limit 초과) 검증
2. 통합 테스트
   - FastAPI 클라이언트로 `/api/robot/ik` 호출 → 응답 구조 검증

## 7. 문서/운영
- `docs/robot_ik_plan.md` 갱신 및 README에 사용법 추가
- 필요 시 Swagger(OpenAPI)에 예시 데이터 포함
- 운영에서는 Pose 후보 및 그리퍼 offset의 허용 범위를 환경 변수로 관리 고려

---

### 구현 완료 시 기대 기능
- 다수의 Pose 후보와 그리퍼 길이를 조합해 최적 joint 상태를 계산
- IK 실패 원인/오차 로그를 통해 디버깅 가능
- 향후 충돌 회피나 동적 계획 모듈로 확장할 수 있는 기반 확보

