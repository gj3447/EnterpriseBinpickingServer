# Enterprise Binpicking Server API 기능 정의서

이 문서는 서버가 제공하는 모든 HTTP API 엔드포인트와 실시간 WebSocket 스트림의 기능과 사용법을 정의합니다.

---

## 1. HTTP API

### 1.1 Health (`/api`)
- **`GET /`**: 서버의 상태를 확인하는 간단한 Health Check 엔드포인트입니다.
  - **응답**: `{"status": "ok"}`

### 1.2 Store (`/api/store`)
- **`GET /status`**: 시스템의 모든 주요 상태 정보(장치, 이미지, 이벤트 등)를 종합하여 반환합니다. 디버깅 및 전체 모니터링에 사용됩니다.
- **`GET /events/status`**: 각 내부 이벤트가 마지막으로 발행된 시간을 ISO 8601 형식의 문자열로 반환합니다. 시스템의 실시간 동작 상태를 확인하는 데 유용합니다.
- **`GET /calibration`**: 현재 시스템에 적용된 카메라 캘리브레이션 데이터를 반환합니다.

### 1.3 Device (`/api/device`)
- **`GET /status`**: 연결된 카메라 장치의 정보(이름, 시리얼 번호 등)와 현재 활성화된 스트림의 설정(해상도, FPS)을 반환합니다.

### 1.4 Images (`/api/images`)
- **`GET /color.jpg`**: 가장 최신의 컬러 이미지를 JPEG 형식으로 반환합니다.
- **`GET /depth.jpg`**: 가장 최신의 뎁스 이미지를 시각화하여 JPEG 형식으로 반환합니다.
- **`GET /aruco_debug.jpg`**: ArUco 마커, 보드 영역, 좌표축 등 모든 디버깅 정보가 그려진 최신 이미지를 JPEG 형식으로 반환합니다.
- **`GET /board_perspective.jpg`**: ArUco 보드를 위에서 수직으로 내려다보는 시점으로 변환한 가상 이미지를 JPEG 형식으로 반환합니다. (보드 인식이 성공했을 때만 사용 가능)
- **`GET /color/raw`**: 가장 최신의 원본 컬러 이미지를 raw bytes 형식(`application/octet-stream`)으로 반환합니다.
- **`GET /depth/raw`**: 가장 최신의 원본 뎁스 이미지를 raw bytes 형식(`application/octet-stream`)으로 반환합니다.

### 1.5 Masks (`/api/masks`)
- **`GET /board.jpg`**: ArUco 보드 영역만 흰색으로 표시된 마스크 이미지를 JPEG 형식으로 반환합니다.
- **`GET /marker.jpg`**: 쿼리 파라미터 `id`로 지정된 특정 ArUco 마커 영역만 흰색으로 표시된 마스크 이미지를 JPEG 형식으로 반환합니다.
  - **쿼리**: `id` (int, 필수) - 마커의 ID

### 1.6 ArUco (`/api/aruco`)
- **`GET /status`**: ArUco 보드의 탐지 상태, 마지막 탐지 시간, 계산된 3D Pose 정보를 반환합니다.
- **`GET /board_markers`**: 탐지된 마커 중 보드에 속한 마커들의 ID와 Pose 목록을 반환합니다.
- **`GET /external_markers`**: 보드에 속하지 않은 외부 마커들의 ID와 Pose 목록을 반환합니다.

### 1.7 Transforms (`/api/transforms`)
- **`GET /board`**: ArUco 보드의 3D Pose를 특정 좌표계 기준으로 변환하여 반환합니다.
- **`GET /robot`**: 로봇 베이스의 3D Pose를 특정 좌표계 기준으로 변환하여 반환합니다.
- **`GET /camera`**: 카메라의 3D Pose를 특정 좌표계 기준으로 변환하여 반환합니다.
- **`GET /external_markers`**: 모든 외부 마커들의 3D Pose를 특정 좌표계 기준으로 변환하여 반환합니다.
- **`GET /all`**: 시스템의 모든 주요 객체(보드, 로봇, 카메라, 외부 마커)의 Pose를 특정 좌표계 기준으로 한번에 계산한 전체 스냅샷을 반환합니다.
  - **쿼리**: `frame` (string, 선택) - 기준 좌표계. "camera", "board", "robot" 중 하나. (기본값: "camera")

### 1.8 Robot (`/api/robot`)
- **`GET /status`**: 로봇 URDF 로드 상태를 조회합니다.
- **`GET /urdf`**: 로봇 URDF 객체의 정보(로봇 이름, 링크 수, 조인트 수, 조인트 이름 등)를 조회합니다.

### 1.9 Pointcloud (`/api/pointcloud`)
- **`GET /status`**: 포인트클라우드 생성 상태와 통계 정보(전체 포인트 수, 유효 포인트 수 등)를 조회합니다.
- **`GET /data`**: 현재 저장된 포인트클라우드 데이터를 조회합니다.
  - **쿼리**: `max_points` (int, 선택) - 반환할 최대 포인트 수. 지정하면 다운샘플링된 데이터 반환
- **`GET /metadata`**: 포인트클라우드의 메타데이터(타임스탬프, 업데이트 시간, 데이터 유무 등)를 조회합니다.
- **`GET /bounds`**: 포인트클라우드의 경계 상자(bounding box) 정보(최소/최대 좌표, 중심점, 크기)를 조회합니다.
- **`DELETE /clear`**: 저장된 포인트클라우드 데이터를 삭제합니다.

### 1.10 Views (`/api/views`)
- **`GET /images`**: 모든 이미지 스트림을 한눈에 볼 수 있는 개발자용 대시보드 HTML 페이지를 반환합니다.
- **`GET /transforms`**: 모든 좌표 변환 스트림을 한눈에 볼 수 있는 개발자용 대시보드 HTML 페이지를 반환합니다.

---

## 2. WebSocket Streams (`/ws`)

### 2.1 이미지 스트림
- **`ws://<host>/ws/color_jpg`**: 실시간 컬러 JPEG 이미지 스트림.
- **`ws://<host>/ws/depth_jpg`**: 실시간 뎁스 JPEG 이미지 스트림.
- **`ws://<host>/ws/aruco_debug_jpg`**: 실시간 ArUco 디버그 이미지 스트림.
- **`ws://<host>/ws/board_perspective_jpg`**: 실시간 원근 보정 이미지 스트림.

### 2.2 좌표 변환 스트림
- **`ws://<host>/ws/transforms_camera`**: '카메라' 좌표계 기준의 실시간 변환 정보(JSON) 스트림.
- **`ws://<host>/ws/transforms_board`**: '보드' 좌표계 기준의 실시간 변환 정보(JSON) 스트림.
- **`ws://<host>/ws/transforms_robot`**: '로봇' 좌표계 기준의 실시간 변환 정보(JSON) 스트림.

### 2.3 포인트클라우드 스트림
- **`ws://<host>/ws/pointcloud`**: 실시간 3D 포인트클라우드 데이터(JSON) 스트림. 포인트 좌표와 RGB 색상 정보를 포함합니다.
