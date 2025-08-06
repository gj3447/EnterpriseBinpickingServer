import threading
from typing import Any, Dict, Optional, Union
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from cachetools import TTLCache

@dataclass
class StoredImage:
    """A container to hold image data and its timestamp together to ensure integrity."""
    data: Union[np.ndarray, bytes]
    timestamp: float

class ImageHandler:
    """
    Handles storage and retrieval for processed images, with dedicated fields
    and locks for each specific image type to ensure clarity and concurrency.
    """
    def __init__(self):
        # Dedicated locks for each specific processed image type.
        self._color_jpeg_lock = threading.RLock()
        self._depth_jpeg_lock = threading.RLock()
        self._aruco_debug_lock = threading.RLock()
        self._board_perspective_lock = threading.RLock()

        # Explicit caches for each image type.
        self._color_jpeg_cache: TTLCache[str, StoredImage] = TTLCache(maxsize=1, ttl=600)
        self._depth_jpeg_cache: TTLCache[str, StoredImage] = TTLCache(maxsize=1, ttl=600)
        self._aruco_debug_cache: TTLCache[str, StoredImage] = TTLCache(maxsize=1, ttl=600)
        self._board_perspective_cache: TTLCache[str, StoredImage] = TTLCache(maxsize=1, ttl=600)

    # --- Update methods ---
    def update_color_jpeg(self, data: bytes, timestamp: float):
        with self._color_jpeg_lock:
            self._color_jpeg_cache['latest'] = StoredImage(data=data, timestamp=timestamp)

    def update_depth_jpeg(self, data: bytes, timestamp: float):
        with self._depth_jpeg_lock:
            self._depth_jpeg_cache['latest'] = StoredImage(data=data, timestamp=timestamp)

    def update_aruco_debug_image(self, data: bytes, timestamp: float):
        with self._aruco_debug_lock:
            self._aruco_debug_cache['latest'] = StoredImage(data=data, timestamp=timestamp)

    def update_board_perspective_image(self, data: bytes, timestamp: float):
        with self._board_perspective_lock:
            self._board_perspective_cache['latest'] = StoredImage(data=data, timestamp=timestamp)

    # --- Get methods ---
    def get_color_jpeg(self) -> Optional[bytes]:
        with self._color_jpeg_lock:
            item = self._color_jpeg_cache.get('latest')
            return item.data if item and isinstance(item.data, bytes) else None

    def get_depth_jpeg(self) -> Optional[bytes]:
        with self._depth_jpeg_lock:
            item = self._depth_jpeg_cache.get('latest')
            return item.data if item and isinstance(item.data, bytes) else None

    def get_aruco_debug_image(self) -> Optional[bytes]:
        with self._aruco_debug_lock:
            item = self._aruco_debug_cache.get('latest')
            return item.data if item and isinstance(item.data, bytes) else None

    def get_board_perspective_image(self) -> Optional[bytes]:
        with self._board_perspective_lock:
            item = self._board_perspective_cache.get('latest')
            return item.data if item and isinstance(item.data, bytes) else None

    # --- Status method ---
    def get_all_images_status(self) -> Dict[str, Any]:
        """Returns a consolidated status of all managed processed images."""
        status = {}
        
        def get_item_status(cache, lock, name):
            with lock:
                item = cache.get('latest')
                if item and isinstance(item.data, bytes):
                    return {
                        "timestamp_utc": datetime.fromtimestamp(item.timestamp).isoformat(),
                        "size_bytes": len(item.data),
                        "format": "jpeg"
                    }
            return None

        status['color_jpg'] = get_item_status(self._color_jpeg_cache, self._color_jpeg_lock, 'color_jpg')
        status['depth_jpg'] = get_item_status(self._depth_jpeg_cache, self._depth_jpeg_lock, 'depth_jpg')
        status['aruco_debug_jpg'] = get_item_status(self._aruco_debug_cache, self._aruco_debug_lock, 'aruco_debug_jpg')
        status['board_perspective_jpg'] = get_item_status(self._board_perspective_cache, self._board_perspective_lock, 'board_perspective_jpg')
        
        return {k: v for k, v in status.items() if v is not None}
