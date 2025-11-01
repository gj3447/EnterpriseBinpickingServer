# 로봇 IK 기능 구현 가이드

이 문서는 현재 코드베이스(`app/` 이하 FastAPI + Pinocchio 구조)에 맞춰 역기구학(IK) 기능을 실제로 도입하는 순서를 정리합니다.

## 0. 사전 체크
- Conda 환경에 `pinocchio`, `numpy`, `scipy`가 설치돼 있는지 확인 (`environment.yml` 반영 필요 시 업데이트)
- `settings.ROBOT_URDF_PATH`가 로드 가능한 URDF 파일을 가리키는지 확인
- `RobotService`가 서버 기동 시 정상적으로 URDF를 로드하는지 로그 확인

## 1. URDF (선택)
1. 가변 그리퍼가 필요 없다면 이 단계는 생략 가능
2. 필요할 경우 `app/static/urdf/...` 하위 URDF를 복사 → 프리스매틱 조인트 추가 → `.env`에서 경로 지정
   - 예: `ROBOT_URDF_PATH=app/static/urdf/dsr_description2/urdf/a0509_var_grip.urdf`
3. 서버 재시작 후 `RobotService` 로드 로그와 `/api/device/robot/status`에서 `loaded=True` 확인

## 2. RobotService 확장 (`app/services/robot_service.py`)
1. Pinocchio 모델을 가져와 IK를 수행하는 `solve_ik(...)` 메서드 추가
   - 입력: 목표 pose 목록(List), 초기 관절 상태(Optional), 최대 반복, 허용 오차, 가변 그리퍼 offset 목록 등
   - 내부:
     - `model = robot_object["model"]`, `data = model.createData()`
     - `pin.forwardKinematics` → `pin.updateFramePlacements`
     - `pin.log`로 오차 계산 후 최소제곱(`np.linalg.lstsq`)으로 `dq` 산출 → `pin.integrate`
     - 관절 한계 체크: `model.lowerPositionLimit`, `upperPositionLimit`
     - Pose × 그리퍼 offset 조합 반복 → 최적 해 선택 및 전체 결과 목록 반환
   - 출력: 최적 joint 벡터, 잔류 오차, 반복 횟수, 선택된 Pose/offset, 후보별 상세 리스트
2. 예외 관리
   - Pinocchio 미로딩, 프레임 ID 미지정, 수렴 실패 등은 `RobotServiceError`(새로운 예외 클래스)로 감싸기

## 3. API 업데이트 (`app/api/v1/endpoints/robot.py`)
1. 기존 GET `/urdf` 엔드포인트를 Pinocchio 딕셔너리 구조에 맞게 수정
   - `robot_object`가 dict 인지 확인 후 `robot_name`, `dof`, `joint_names`, `joint_limits` 등 반환
2. POST `/api/device/robot/ik` 추가
   - `RobotIkRequest` Pydantic 모델 정의 (목표 pose 리스트, 초기 q, grip_offsets 등)
   - `robot_service.solve_ik` 호출 → 성공 시 joint 벡터와 메타데이터 반환
   - 수렴 실패/모델 부재 시 HTTP 400/404 예외 처리

## 4. Pydantic 스키마 (`app/schemas/aruco.py` 또는 신규 파일)
1. HTTP 요청/응답 모델 정의
   - Pose 입력: 위치(float[3]) + 회전(quaternion or RPY)
   - IK 응답: joint 배열, error, iterations, 선택된 pose index, grip offset, 후보 리스트

## 5. 서비스 wiring
- `app/dependencies.py`는 이미 싱글턴 인스턴스를 제공하므로 별도 수정 불필요
- 단, 새 예외나 설정 값이 필요하면 `app/core/config.py`에 추가

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

