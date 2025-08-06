from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import traceback
import asyncio

from app.api.v1.router import api_router
from app.websockets.v1.router import router as websocket_router
from app.dependencies import (
    get_camera_service, 
    get_aruco_service, 
    get_streaming_service, 
    get_image_service,
    get_frame_sync_service
)
from app.core.logging import logger

# 템플릿 렌더러를 main.py에서 직접 관리
templates = Jinja2Templates(directory="app/static/templates")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 의존성에서 모든 서비스 가져오기
    camera_service = get_camera_service()
    frame_sync_service = get_frame_sync_service()
    aruco_service = get_aruco_service()
    image_service = get_image_service()
    streaming_service = get_streaming_service()

    logger.info("Application startup: Starting background services...")
    
    # 각 서비스의 백그라운드 리스너/작업 시작
    await camera_service.start()
    await frame_sync_service.start()
    await aruco_service.start()
    await image_service.start()
    streaming_service.start_listening()
    
    yield
    
    logger.info("Application shutdown: Stopping background services...")
    # 서비스 중지 (이벤트 구독 해제 및 태스크 캔슬 포함)
    await camera_service.stop()
    await frame_sync_service.stop()
    await aruco_service.stop()
    await image_service.stop()
    streaming_service.stop_listening()

    logger.info("All background services have been signaled to stop.")


app = FastAPI(title="Enterprise Binpicking Server", lifespan=lifespan)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_details = traceback.format_exc()
    logger.error(f"Unhandled exception for request {request.method} {request.url}:\n{error_details}")
    return JSONResponse(status_code=500, content={"detail": "An internal server error occurred."})

# HTTP API 라우터 등록
app.include_router(api_router, prefix="/api")

# WebSocket 라우터 등록
app.include_router(websocket_router)


@app.get("/", response_class=HTMLResponse, summary="Main application dashboard")
def read_root(request: Request):
    """Renders the main dashboard HTML page."""
    return templates.TemplateResponse("main_dashboard.html", {"request": request})
