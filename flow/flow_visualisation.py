import numpy as np
import cv2


class Visualiser:
    COLOR_CHANNEL_COUNT = 3

    def __init__(self, width: int, height: int):
        self._hsv_mask = np.zeros((height, width, self.COLOR_CHANNEL_COUNT), dtype=np.uint8)
        self._hsv_mask[..., 1] = 255

    def get_frame(self, magnitude, angle):
        self._hsv_mask[..., 0] = angle * 180 / np.pi / 2
        self._hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(self._hsv_mask, cv2.COLOR_HSV2BGR)

