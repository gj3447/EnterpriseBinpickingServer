from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/static/templates")
router = APIRouter()

# --- 대시보드 ---

@router.get("/images", response_class=HTMLResponse, summary="이미지 스트리밍 대시보드")
async def get_image_streaming_dashboard(request: Request):
    """모든 이미지 스트림을 테스트할 수 있는 대시보드 페이지를 반환합니다."""
    return templates.TemplateResponse("image_streaming.html", {"request": request})

@router.get("/transforms", response_class=HTMLResponse, summary="좌표 변환 대시보드")
async def get_transform_streaming_dashboard(request: Request):
    """모든 좌표 변환 스트림을 테스트할 수 있는 대시보드 페이지를 반환합니다."""
    return templates.TemplateResponse("transform_dashboard.html", {"request": request})

# --- 임베딩 가능한 이미지 뷰 ---

@router.get("/embed/images/color", response_class=HTMLResponse)
async def embed_color_stream(request: Request):
    """오직 컬러 이미지 스트림만 보여주는 임베딩용 페이지입니다."""
    return templates.TemplateResponse("embed_image_stream.html", {
        "request": request,
        "title": "Color Stream",
        "stream_id": "color_jpg"
    })

@router.get("/embed/images/depth", response_class=HTMLResponse)
async def embed_depth_stream(request: Request):
    """오직 뎁스 이미지 스트림만 보여주는 임베딩용 페이지입니다."""
    return templates.TemplateResponse("embed_image_stream.html", {
        "request": request,
        "title": "Depth Stream",
        "stream_id": "depth_jpg"
    })

@router.get("/embed/images/aruco_debug", response_class=HTMLResponse)
async def embed_aruco_debug_stream(request: Request):
    """오직 ArUco 디버그 이미지 스트림만 보여주는 임베딩용 페이지입니다."""
    return templates.TemplateResponse("embed_image_stream.html", {
        "request": request,
        "title": "Aruco Debug Stream",
        "stream_id": "aruco_debug_jpg"
    })

@router.get("/embed/images/board_perspective", response_class=HTMLResponse)
async def embed_board_perspective_stream(request: Request):
    """오직 '전면 보정'된 보드 이미지 스트림만 보여주는 임베딩용 페이지입니다."""
    return templates.TemplateResponse("embed_image_stream.html", {
        "request": request,
        "title": "Board Perspective Stream",
        "stream_id": "board_perspective_jpg"
    })

# --- 임베딩 가능한 좌표 변환 뷰 ---

@router.get("/embed/transforms/in_camera_frame", response_class=HTMLResponse)
async def embed_transforms_in_camera_frame(request: Request):
    """카메라 좌표계 기준의 변환 정보 스트림만 보여주는 임베딩용 페이지입니다."""
    return templates.TemplateResponse("embed_transform_stream.html", {
        "request": request,
        "title": "Transforms in Camera Frame",
        "stream_id": "transforms_camera"
    })

@router.get("/embed/transforms/in_board_frame", response_class=HTMLResponse)
async def embed_transforms_in_board_frame(request: Request):
    """보드 좌표계 기준의 변환 정보 스트림만 보여주는 임베딩용 페이지입니다."""
    return templates.TemplateResponse("embed_transform_stream.html", {
        "request": request,
        "title": "Transforms in Board Frame",
        "stream_id": "transforms_board"
    })

@router.get("/embed/transforms/in_robot_frame", response_class=HTMLResponse)
async def embed_transforms_in_robot_frame(request: Request):
    """로봇 좌표계 기준의 변환 정보 스트림만 보여주는 임베딩용 페이지입니다."""
    return templates.TemplateResponse("embed_transform_stream.html", {
        "request": request,
        "title": "Transforms in Robot Frame",
        "stream_id": "transforms_robot"
    })
