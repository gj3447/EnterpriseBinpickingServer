# Enterprise Binpicking Server API & WebSocket Map


## HTTP API Endpoints


### Ai


- **`POST`** `/api/ai/ai/predict`




### Aruco


- **`GET`** `/api/aruco/board_markers`
    - 탐지된 마커 중 ArUco 보드에 속하는 마커들의 목록을 반환합니다.

- **`GET`** `/api/aruco/external_markers`
    - 탐지된 마커 중 ArUco 보드에 속하지 않는 외부 마커들의 목록을 반환합니다.

- **`GET`** `/api/aruco/status`
    - ArUco 보드의 3D 자세(Pose) 추정 결과를 반환합니다.



### Calibration


- **`GET`** `/api/calibration/`
    - Store에 저장된 최신 카메라 캘리브레이션 데이터를 반환합니다.



### Default


- **`GET`** `/`




### Device


- **`GET`** `/api/device/status`
    - Store에 저장된 최신 장치 상태 정보를 반환합니다.



### Health


- **`GET`** `/api/`




### Images


- **`GET`** `/api/images/aruco_debug.jpg`
    - 최신 컬러 이미지 위에 ArUco 보드, 외부 마커 등을 시각화한 디버그 이미지를 반환합니다.

- **`GET`** `/api/images/board_perspective.jpg`
    - ArUco 보드를 기준으로 원근을 보정하여, 보드가 똑바로 보이도록 변환한

- **`GET`** `/api/images/color.jpg`
    - Store에 저장된 최신 원본 컬러 이미지를 JPEG 형식으로 반환합니다.

- **`GET`** `/api/images/depth.jpg`
    - Store에 저장된 최신 깊이 이미지를 시각화하여 JPEG 형식으로 반환합니다.



### Masks


- **`GET`** `/api/masks/board.jpg`
    - 탐지된 ArUco 보드 영역만 흰색으로 표시된 마스크 이미지를 반환합니다.

- **`GET`** `/api/masks/marker.jpg`
    - 쿼리 파라미터로 받은 `id`에 해당하는 마커 영역만 흰색으로 표시된



### Robots


- **`GET`** `/api/robots/urdf/{robot_name}`




### Store


- **`GET`** `/api/store/status`
    - 애플리케이션의 모든 주요 상태 정보(장치, 이미지, 캘리브레이션, ArUco 등)를



### Transforms


- **`GET`** `/api/transforms/`
    - 시스템의 모든 주요 컴포넌트(보드, 로봇, 카메라, 외부 마커)의 6D Pose를



### Views


- **`GET`** `/api/views/images`
    - 모든 이미지 스트리밍 웹소켓을 테스트할 수 있는 대시보드 HTML 페이지를 렌더링합니다.

- **`GET`** `/api/views/images/embed/{stream_name}`
    - 지정된 단일 이미지 스트림만 보여주는 임베딩용 HTML 페이지를 렌더링합니다.

- **`GET`** `/api/views/transforms`
    - 좌표 변환 스트리밍 웹소켓을 테스트할 수 있는 HTML 페이지를 렌더링합니다.



## WebSocket Endpoints


- **`WEBSOCKET`** `/api/ws/images`


- **`WEBSOCKET`** `/api/ws/transforms`

