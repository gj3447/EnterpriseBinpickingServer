"""
This module defines the Pydantic models for various event payloads used in
the application's event bus system. Payloads are designed to carry all
necessary data to minimize the need for subscribers to query the store.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

from app.schemas.aruco import Pose, DetectedMarker
from app.schemas.transforms import SystemTransformSnapshot, SystemTransformSnapshotResponse

class EventPayload(BaseModel):
    """Base model for all event payloads."""
    timestamp: float = Field(..., description="The unix timestamp when the event was generated.")

    class Config:
        arbitrary_types_allowed = True

# --- Low-level Image Events ---
class ColorImageReceivedPayload(EventPayload):
    """Payload for COLOR_IMAGE_RECEIVED event."""
    image_data: np.ndarray

class DepthImageReceivedPayload(EventPayload):
    """Payload for DEPTH_IMAGE_RECEIVED event."""
    image_data: np.ndarray

# --- High-level Synchronized Event ---
class SyncFrameReadyPayload(EventPayload):
    """
    Payload for SYNC_FRAME_READY event, carrying the synchronized pair of images.
    """
    color_image_data: np.ndarray
    depth_image_data: np.ndarray

# --- Analysis Result Events ---
class ArucoUpdatePayload(EventPayload):
    """
    Payload for ARUCO_UPDATE event, carrying the analysis result and the
    source image used for the analysis.
    """
    source_image_data: np.ndarray
    board_pose: Optional[Pose]
    board_markers: List[DetectedMarker]
    external_markers: List[DetectedMarker]

# --- WebSocket Events ---
class WsColorImageUpdatePayload(EventPayload):
    """Payload for WS_COLOR_IMAGE_UPDATE event."""
    jpeg_data: bytes

class WsDepthImageUpdatePayload(EventPayload):
    """Payload for WS_DEPTH_IMAGE_UPDATE event."""
    jpeg_data: bytes

class WsDebugImageUpdatePayload(EventPayload):
    """Payload for WS_DEBUG_IMAGE_UPDATE event."""
    jpeg_data: bytes

class WsPerspectiveImageUpdatePayload(EventPayload):
    """Payload for WS_PERSPECTIVE_IMAGE_UPDATE event."""
    jpeg_data: bytes

class SystemTransformsUpdatePayload(EventPayload):
    """
    Payload for SYSTEM_TRANSFORMS_UPDATE event, containing a snapshot
    of all transformations for each coordinate frame, formatted for external clients.
    """
    snapshots: List[SystemTransformSnapshotResponse]
