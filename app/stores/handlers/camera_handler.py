import threading
from typing import Any, Dict, Optional
from dataclasses import dataclass
import numpy as np
from cachetools import TTLCache

@dataclass
class StoredImage:
    """A container to hold an image numpy array and its timestamp together."""
    data: np.ndarray
    timestamp: float

class CameraHandler:
    """
    Handles storage and retrieval for raw color and depth images, each with its
    own dedicated field, lock, and methods for maximum type safety and clarity.
    """
    def __init__(self):
        # Dedicated locks for each image type to ensure maximum concurrency.
        self._color_lock = threading.RLock()
        self._depth_lock = threading.RLock()

        # Explicit fields for color and depth images. Using a list to hold a single
        # StoredImage object in a TTLCache-like manner (but simpler).
        self._color_image_cache: TTLCache[str, StoredImage] = TTLCache(maxsize=1, ttl=600)
        self._depth_image_cache: TTLCache[str, StoredImage] = TTLCache(maxsize=1, ttl=600)

    def update_color_image(self, image_data: np.ndarray, timestamp: float):
        """Stores the latest raw color image."""
        with self._color_lock:
            self._color_image_cache['latest'] = StoredImage(data=image_data, timestamp=timestamp)

    def update_depth_image(self, image_data: np.ndarray, timestamp: float):
        """Stores the latest raw depth image."""
        with self._depth_lock:
            self._depth_image_cache['latest'] = StoredImage(data=image_data, timestamp=timestamp)

    def get_color_image(self) -> Optional[StoredImage]:
        """Retrieves the latest stored color image object."""
        with self._color_lock:
            return self._color_image_cache.get('latest')

    def get_depth_image(self) -> Optional[StoredImage]:
        """Retrieves the latest stored depth image object."""
        with self._depth_lock:
            return self._depth_image_cache.get('latest')

    def get_status(self) -> Dict[str, Any]:
        """Returns the status of the currently stored raw images."""
        status = {}
        color_item = self.get_color_image()
        if color_item:
            status['color_raw'] = {
                "timestamp": color_item.timestamp,
                "shape": color_item.data.shape,
                "dtype": str(color_item.data.dtype)
            }
            
        depth_item = self.get_depth_image()
        if depth_item:
            status['depth_raw'] = {
                "timestamp": depth_item.timestamp,
                "shape": depth_item.data.shape,
                "dtype": str(depth_item.data.dtype)
            }
        return status
