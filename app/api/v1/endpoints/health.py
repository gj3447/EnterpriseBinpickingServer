from fastapi import APIRouter

router = APIRouter()

@router.get(
    "/",
    summary="Simple health check"
)
def health_check():
    """
    서버가 살아있는지 확인하기 위한 간단한 Health Check 엔드포인트입니다.
    """
    return {"status": "ok"}
