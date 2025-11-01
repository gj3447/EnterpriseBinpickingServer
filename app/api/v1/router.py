from fastapi import APIRouter
from app.api.v1.endpoints import (
    health, 
    store, 
    images, 
    aruco,
    device,
    transforms,
    masks,
    views,
    robot,
    pointcloud
)

api_router = APIRouter()

# --- HTTP API Endpoints ---
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(store.router, prefix="/store", tags=["Store"])
api_router.include_router(images.router, prefix="/images", tags=["Images"])
api_router.include_router(masks.router, prefix="/masks", tags=["Masks"])
api_router.include_router(aruco.router, prefix="/aruco", tags=["ArUco"])
api_router.include_router(device.router, prefix="/device", tags=["Device"])
api_router.include_router(robot.router, prefix="/robot", tags=["Robot"])
api_router.include_router(pointcloud.router, prefix="/pointcloud", tags=["Pointcloud"])
api_router.include_router(transforms.router, prefix="/transforms", tags=["Transforms"])

# --- Web View Endpoints ---
api_router.include_router(views.router, prefix="/views/v1", tags=["Views (v1)"])
