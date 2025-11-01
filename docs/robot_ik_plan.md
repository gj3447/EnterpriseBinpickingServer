# 로봇 IK/그리퍼 확장 구현 계획

## 1. 목표 개요
- Pinocchio 기반 역기구학(IK)을 서버 내부 서비스로 제공
- 그리퍼 길이를 가변적으로 제어하여, 후보 Pose 또는 prismatic joint를 활용한 계산 지원
- FastAPI API 또는 내부 서비스에서 IK 결과를 활용할 수 있도록 구조화

## 2. 현재 구조 요약
- `RobotService`가 URDF를 Pinocchio 모델로 로드하고 `ApplicationStore`에 저장
- `/api/device/robot/pinocchio`로 모델 정보 확인 가능
- IK 계산 로직은 아직 서비스/엔드포인트로 구현되지 않음
- 그리퍼 길이 가변성은 URDF 수정 여부에 따라 두 가지 접근이 존재

## 3. 구현 단계

### 3.1 URDF 조정 (가변 그리퍼)
1. 대상 URDF 파일 확인 (`settings.ROBOT_URDF_PATH`)
2. 말단 링크(`tool0` 등)에 prismatic joint와 가상 링크(`gripper_slide_link`) 추가
   - `<joint name="gripper_extension_joint" type="prismatic">`
   - `<limit lower="0.0" upper="0.1" velocity="0.2"/>` 등 물리 범위 정의
   - 실제 그리퍼 링크는 `gripper_slide_link`에 고정 조인트로 연결
3. 관성/시각/충돌 모델 보정
4. 새 URDF 파일 저장 후 `.env` 또는 설정 값 업데이트
5. 서버 재시작 → Pinocchio 모델 재로딩 확인

### 3.2 IK 서비스 설계
1. `app/services/robot_service.py`에 IK 계산 메서드 추가
   - 입력: 목표 pose(SE3), 초기 관절 상태(optional), 허용 오차, 최대 반복 등
   - 출력: `q` 벡터 및 잔류 오차
2. `model.createData()`를 호출해 독립적인 data 객체 사용 → 동시에 여러 IK 계산 가능
3. IK 로직
   - `pin.forwardKinematics`, `pin.updateFramePlacements`
   - `pin.log(current.inverse() * target)`로 6D 오차 계산
   - `pin.computeFrameJacobian` → 최소제곱으로 `dq` 계산
   - `pin.integrate(model, q, dq)` 적용
   - 관절 한계 검증(필요 시 클리핑)과 수렴 조건
4. 다중 후보 pose 지원
   - 입력을 Pose 리스트로 받아 각각 반복 → 베스트 솔루션 선택 혹은 목록 반환
5. 가변 그리퍼 고려 방법
   - URDF에 prismatic joint가 추가된 경우: 관절에 포함되어 자동 처리
   - URDF 변경이 어려울 때: IK 실행 전 목표 pose에 가상 오프셋 적용(그리퍼 길이 후보)

### 3.3 API/사용자 인터페이스
1. `app/api/v1/endpoints/robot.py`에 POST `/api/device/robot/ik`
   - 요청: 목표 pose(회전+위치), 초기 상태, 후보 그리퍼 길이 등
   - 응답: joint 벡터, 오차 norm, 사용된 그리퍼 길이
2. 검증 및 예외 처리
   - 로봇 모델 미로딩, Pinocchio 미설치, 수렴 실패 등
3. 필요 시 스웨거 문서 및 README 업데이트

## 4. 테스트 계획
- 단위 테스트: IK 함수에 대해 알려진 포즈 → Expected joint 비교
- 시나리오 테스트: 여러 길이 후보에 대해 솔루션 비교
- 로그 확인: 수렴 여부, 반복 횟수, 관절 한계 위반 여부

## 5. 향후 과제
- Jacobian 기반 방법 외에도 Levenberg-Marquardt, 전역 탐색 등 대안 고려
- 관절 한계/충돌 회피를 위한 최적화 로직 고도화
- Websocket/Visualization과 연동해 IK 결과 시각화 또는 시뮬레이션

