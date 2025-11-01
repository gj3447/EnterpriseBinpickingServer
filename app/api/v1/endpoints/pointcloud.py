from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, Optional, List

from app.stores.application_store import ApplicationStore
from app.dependencies import get_store
from app.core.logging import logger

router = APIRouter()

@router.get("/status", response_model=Dict[str, Any])
def get_pointcloud_status(
    store: ApplicationStore = Depends(get_store)
):
    """포인트클라우드 생성 상태와 통계 정보를 조회합니다."""
    return store.pointcloud.get_statistics()

@router.get("/data", response_model=Dict[str, Any])
def get_pointcloud_data(
    store: ApplicationStore = Depends(get_store),
    max_points: Optional[int] = Query(10000, description="반환할 최대 포인트 수 (기본값: 10000, 0=전체)")
):
    """
    현재 저장된 포인트클라우드 데이터를 조회합니다.
    
    - max_points: 반환할 최대 포인트 수 (기본값: 10000, 0으로 설정하면 전체 반환)
    """
    if max_points and max_points > 0:
        points, colors = store.pointcloud.get_downsampled_pointcloud(max_points)
    else:
        points, colors = store.pointcloud.get_pointcloud()
    
    if points is None:
        raise HTTPException(status_code=404, detail="No pointcloud data available")
    
    num_points = len(points)
    
    # 대용량 데이터 경고
    if num_points > 50000:
        logger.warning(f"Returning large pointcloud data: {num_points} points")
    
    return {
        "num_points": num_points,
        "points": points.tolist(),  # numpy array를 list로 변환
        "colors": colors.tolist() if colors is not None else None,
        "has_colors": colors is not None
    }

@router.get("/metadata", response_model=Dict[str, Any])
def get_pointcloud_metadata(
    store: ApplicationStore = Depends(get_store)
):
    """포인트클라우드의 메타데이터를 조회합니다."""
    metadata = store.pointcloud.get_pointcloud_with_metadata()
    
    # numpy 배열은 제외하고 메타데이터만 반환
    return {
        "timestamp": metadata["timestamp"],
        "last_updated_utc": metadata["last_updated_utc"],
        "total_points": metadata["total_points"],
        "valid_points": metadata["valid_points"],
        "has_data": metadata["points"] is not None,
        "has_colors": metadata["colors"] is not None
    }

@router.get("/bounds", response_model=Dict[str, Any])
def get_pointcloud_bounds(
    store: ApplicationStore = Depends(get_store)
):
    """포인트클라우드의 경계 상자(bounding box) 정보를 조회합니다."""
    points, _ = store.pointcloud.get_pointcloud()
    
    if points is None:
        raise HTTPException(status_code=404, detail="No pointcloud data available")
    
    # 각 축의 최소/최대값 계산
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    
    return {
        "min": {
            "x": float(min_vals[0]),
            "y": float(min_vals[1]),
            "z": float(min_vals[2])
        },
        "max": {
            "x": float(max_vals[0]),
            "y": float(max_vals[1]),
            "z": float(max_vals[2])
        },
        "center": {
            "x": float((min_vals[0] + max_vals[0]) / 2),
            "y": float((min_vals[1] + max_vals[1]) / 2),
            "z": float((min_vals[2] + max_vals[2]) / 2)
        },
        "size": {
            "x": float(max_vals[0] - min_vals[0]),
            "y": float(max_vals[1] - min_vals[1]),
            "z": float(max_vals[2] - min_vals[2])
        }
    }

@router.delete("/clear", response_model=Dict[str, str])
def clear_pointcloud(
    store: ApplicationStore = Depends(get_store)
):
    """저장된 포인트클라우드 데이터를 삭제합니다."""
    store.pointcloud.clear()
    return {"message": "Pointcloud data cleared successfully"}
