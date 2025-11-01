# 로봇 IK 기능 구현 가이드

이 문서는 현재 코드베이스(`app/` 이하 FastAPI + Pinocchio 구조)에 맞춰 역기구학(IK) 기능을 실제로 도입하는 순서를 정리합니다.

## 0. 사전 체크
- Conda 환경에 `pinocchio`, `numpy`, `scipy`가 설치돼 있는지 확인 (`environment.yml` 반영 필요 시 업데이트)
- `settings.ROBOT_URDF_PATH`가 로드 가능한 URDF 파일을 가리키는지 확인
- `RobotService`가 서버 기동 시 정상적으로 URDF를 로드하는지 로그 확인

## 1. URDF 전략
1. 기본 URDF(그리퍼 고정)와 프리스매틱 조인트가 포함된 URDF 두 버전을 준비
   - `.env`의 `ROBOT_URDF_PATH`로 어느 버전을 로드할지 선택하도록 구성
   - 예: `ROBOT_URDF_PATH=app/static/urdf/dsr_description2/urdf/a0509.urdf` (고정), `.../a0509_var_grip.urdf` (프리스매틱)
2. 서버 기동 시 `RobotService`가 선택된 URDF를 Pinocchio로 로드하고, 모델 객체에 `has_gripper_joint` 플래그를 기록 (존재 여부 판별)
3. `/api/device/robot/status`에서 `loaded=True`를 확인하고, 필요 시 현재 URDF 버전을 응답에 포함

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
1. 기존 GET `/urdf` 엔드포인트를 Pinocchio 딕셔너리 구조에 맞게 수정
   - `robot_object`가 dict 인지 확인 후 `robot_name`, `dof`, `joint_names`, `joint_limits` 등 반환
2. POST `/api/device/robot/ik` 추가
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
   - FastAPI 클라이언트로 `/api/device/robot/ik` 호출 → 응답 구조 검증

## 7. 문서/운영
- `docs/robot_ik_plan.md` 갱신 및 README에 사용법 추가
- 필요 시 Swagger(OpenAPI)에 예시 데이터 포함
- 운영에서는 Pose 후보 및 그리퍼 offset의 허용 범위를 환경 변수로 관리 고려

---

### 구현 완료 시 기대 기능
- 다수의 Pose 후보와 그리퍼 길이를 조합해 최적 joint 상태를 계산
- IK 실패 원인/오차 로그를 통해 디버깅 가능
- 향후 충돌 회피나 동적 계획 모듈로 확장할 수 있는 기반 확보

