# OPC UA 태그 사용 가이드

이 문서는 `OPC_SERVER` 프로젝트에서 제공하는 OPC UA 서버의 태그 구조와 사용 방법을 정리한 것입니다. 현재 게이트웨이는 로봇(Modbus TCP)과 OPC UA 클라이언트 사이를 중계하며, 모든 태그는 `GatewayServerV2` 클래스(`GatewayServerv2.py`)에서 정의됩니다.

- **엔드포인트**: `opc.tcp://0.0.0.0:4840/doosan/server/`
- **서버 이름**: `Doosan Robot OPC UA Gateway`
- **네임스페이스 URI**: `http://dooly.net/DoosanRobot`
- **루트 오브젝트**: `Objects/MyRobot`

## 노드 트리 개요

| 경로 | 노드 키 | 데이터 타입 | R/W | 설명 |
| --- | --- | --- | --- | --- |
| `MyRobot/Status/CurrentJoints` | `current_joints` | `Double[6]` | R | 현재 조인트 각도 (도 단위) |
| `MyRobot/Status/CurrentTCP` | `current_tcps` | `Double[6]` | R | 현재 TCP 위치/자세 (로봇 컨트롤러 포맷) |
| `MyRobot/Status/JointTorques` | `joint_torques` | `Double[6]` | R | 각 조인트 토크 |
| `MyRobot/Status/ToolForces` | `tool_forces` | `Double[6]` | R | TCP 힘/모멘트 |
| `MyRobot/Status/MotionState` | `motion_state` | `Int32` | R | 로봇 모션 상태 플래그 |
| `MyRobot/Status/CommandStatus` | `cmd_status` | `Int32` | R | 최근 명령 실행 상태 코드 (`0=완료`, `1=실행`, `2=에러`) |
| `MyRobot/Status/ErrorCode` | `error_code` | `Int32` | R | 에러 코드 |
| `MyRobot/Status/Custom/...` | `each_*` | `Double` | R | 배열 태그를 단일 값으로 분리한 전용 노드 |
| `MyRobot/Gripper/Status/In1~3` | `gripper_in{1,2,3}` | `Boolean` | R | 그리퍼 입력 상태 |
| `MyRobot/Commands/TargetJoints` | `target_joints` | `Double[6]` | R/W | moveJ 목표 각도 |
| `MyRobot/Commands/TargetTCP` | `target_tcp` | `Double[6]` | R/W | moveL 목표 TCP |
| `MyRobot/Commands/Mode` | `move_mode` | `Int32` | R/W | `1=moveJ`, `2=moveL` |
| `MyRobot/Commands/JVel` | `movej_vel` | `Double` | R/W | moveJ 속도 |
| `MyRobot/Commands/JAcc` | `movej_acc` | `Double` | R/W | moveJ 가속도 |
| `MyRobot/Commands/LVel` | `movel_vel` | `Double` | R/W | moveL 속도 |
| `MyRobot/Commands/LAcc` | `movel_acc` | `Double` | R/W | moveL 가속도 |
| `MyRobot/Commands/Trigger` | `trigger` | `Int32` | R/W | 모션 실행 트리거 (1→실행, 자동으로 0 복귀) |
| `MyRobot/Gripper/Commands/Open` | `gripper_out1` | `Boolean` | R/W | 그리퍼 출력 1 (예: 오픈) |
| `MyRobot/Gripper/Commands/Close` | `gripper_out2` | `Boolean` | R/W | 그리퍼 출력 2 (예: 클로즈) |
| `MyRobot/ModbusConnection/Connected` | `modbus_connected` | `Boolean` | R | Modbus 연결 상태 |
| `MyRobot/ModbusConnection/LastDisconnectTime` | `last_disconnect_time` | `String` | R | 마지막 끊김 시각 (ISO8601) |
| `MyRobot/ModbusConnection/LastReconnectTime` | `last_reconnect_time` | `String` | R | 마지막 재연결 시각 |
| `MyRobot/ModbusConnection/ReconnectElapsed` | `reconnect_elapsed` | `Int32` | R | 예약 필드 (현재 미사용) |
| `MyRobot/ModbusConnection/ReconnectAttempts` | `reconnect_attempts` | `Int32` | R | 연속 재접속 실패 횟수 |
| `MyRobot/ModbusConnection/TotalDisconnected` | `total_disconnected` | `Int32` | R | 누적 끊김 횟수 |

### Custom 폴더(`Status/Custom`)

배열 데이터(`CurrentJoints` 등)를 단일 값으로 접근할 수 있도록 추가된 노드입니다. 예를 들어 `MyRobot/Status/Custom/CurrentJoint0`는 `J1`의 현재 각도를, `CurrentTcp3`는 TCP Rx 등을 나타냅니다. 모든 값은 `Double`이며 읽기 전용입니다.

## 데이터 갱신 및 스케일

- Modbus 레지스터를 0.1 단위(`SCALE_FACTOR = 100.0`)로 스케일링하여 읽지만, OPC UA 노드에는 **실제 물리 단위로 변환된 실수(Double)**가 기록됩니다.
- 상태 갱신 주기: 약 100 ms (`_update_opc_nodes_task` 내부 `await asyncio.sleep(0.1)`).
- 명령 결과(`CommandStatus`, `ErrorCode`)는 30~100 ms 간격으로 확인됩니다.

## 모션 명령 사용 절차

1. `MyRobot/Commands/Mode`에 사용할 모션 유형을 설정합니다.
   - `1`: moveJ (조인트 공간)
   - `2`: moveL (TCP 공간)
2. 모션 유형에 따라 목표와 속성 값을 기입합니다.
   - moveJ: `TargetJoints[0..5]`, `JVel`, `JAcc`
   - moveL: `TargetTCP[0..5]`, `LVel`, `LAcc`
3. **필요 시 자동 제한 적용**
   - `TargetJoints`와 `JVel`은 `robot_config.json`의 `joint_angle_limits_applied`, `joint_velocity_limits_applied`에 따라 즉시 클램핑됩니다. 허용 범위를 벗어나면 서버가 값을 보정하고 경고 로그를 남깁니다.
4. 연결 상태를 확인합니다.
   - `ModbusConnection/Connected`가 `true`인지 확인합니다. 끊겨 있을 경우 `Trigger`는 자동으로 0으로 되돌아가며 명령이 실행되지 않습니다.
5. `MyRobot/Commands/Trigger`에 `1`을 기록합니다.
   - 서버는 내부적으로 Modbus 명령을 전송하고, 성공/실패와 관계없이 트리거 값을 `0`으로 리셋합니다.
6. 실행 상태 확인
   - `CommandStatus == 0`: 완료
   - `CommandStatus == 1`: 실행 중
   - `CommandStatus == 2`: 에러 (세부 코드는 `ErrorCode` 확인)

### 예시 시퀀스 (moveJ)

1. `Connected == true`인지 확인
2. `Mode = 1`
3. `TargetJoints = [0, 0, 0, 90, -90, 30]`
4. `JVel = 50`, `JAcc = 50`
5. `Trigger = 1`
6. `CommandStatus`와 `ErrorCode` 모니터링

## 그리퍼 I/O

- 읽기: `Gripper/Status/In1~In3`는 Modbus 입력(0/1)을 Boolean으로 제공하며, 주기적으로 업데이트됩니다.
- 쓰기: `Gripper/Commands/Open`, `Gripper/Commands/Close`는 각각 Modbus 출력 186, 187에 매핑됩니다. `true/false` 값을 쓰면 즉시 Modbus에 반영됩니다.

## Modbus 연결 모니터링

- 게이트웨이는 백그라운드에서 지속적으로 Modbus TCP 연결을 확인합니다.
- 연결이 끊기면 즉시 재연결을 시도하며, 실패 횟수와 시각을 OPC 노드로 노출합니다.
- 재연결 성공 후에는 `Connected`가 `true`로 바뀌고 `reconnect_attempts`가 0으로 초기화됩니다.

## 로깅 및 문제 해결

- 모든 경고/오류 메시지는 프로젝트 루트의 `gateway.log`에 기록됩니다.
- 모션 제한이나 Modbus 예외가 발생하면 `[WARNING]` 또는 `[ERROR]` 로그와 함께 원인이 남습니다.
- OPC UA 서버 자체가 정상적으로 기동하면 `[INFO] OPC UA 서버가 ... 실행 중입니다.` 로그가 출력됩니다.

## 참고 파일

- `GatewayServerv2.py`: OPC 태그 생성, 연결 관리, 명령 처리 로직
- `setting.py`: OPC/Modbus 주소, 엔드포인트, 스케일 팩터 정의
- `robot_config.json`: 조인트/속도 제한, Modbus 주소 매핑 정의
- `func.py`: Modbus <-> 실수 변환, 로그 함수

---

문의나 추가 개선 사항이 있으면 `gatewayserver 개선_포인트.txt` 등을 참고하거나 로그를 기반으로 원인을 파악한 뒤 태그 구조를 확장하십시오.

