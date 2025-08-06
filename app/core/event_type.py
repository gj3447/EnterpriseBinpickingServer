from enum import Enum

class EventType(Enum):
    """
    애플리케이션 전체에서 사용되는 이벤트 타입들입니다.
    
    카테고리별 구분:
    - CORE: 핵심 서비스 계층 이벤트
    - WS: 웹소켓 스트리밍 관련 이벤트
    """
    
    # --- CORE EVENTS (서비스 계층) ---
    COLOR_IMAGE_RECEIVED        = "COLOR_IMAGE_RECEIVED"        # 카메라에서 컬러 이미지 수신 (저수준)
    DEPTH_IMAGE_RECEIVED        = "DEPTH_IMAGE_RECEIVED"        # 카메라에서 깊이 이미지 수신 (저수준)
    
    SYNC_FRAME_READY            = "SYNC_FRAME_READY"            # 컬러/뎁스 프레임 쌍 동기화 완료 (고수준)

    ARUCO_UPDATE                = "ARUCO_UPDATE"                # ArUco 마커 감지 결과 업데이트 (내부용)
    
    # --- WEBSOCKET EVENTS (실시간 스트리밍) ---
    # StreamingService가 직접 구독하는 명시적인 이벤트들입니다.
    WS_COLOR_IMAGE_UPDATE       = "WS_COLOR_IMAGE_UPDATE"       # 웹소켓: 컬러 JPEG 준비 완료
    WS_DEPTH_IMAGE_UPDATE       = "WS_DEPTH_IMAGE_UPDATE"       # 웹소켓: 뎁스 JPEG 준비 완료
    WS_DEBUG_IMAGE_UPDATE       = "WS_DEBUG_IMAGE_UPDATE"       # 웹소켓: ArUco 디버그 이미지 준비 완료
    WS_PERSPECTIVE_IMAGE_UPDATE = "WS_PERSPECTIVE_IMAGE_UPDATE" # 웹소켓: 원근 보정 이미지 준비 완료
    
    SYSTEM_TRANSFORMS_UPDATE    = "SYSTEM_TRANSFORMS_UPDATE"    # 웹소켓: 모든 좌표계 변환 정보 스냅샷 준비 완료
