import numpy as np
import cv2


class DisplayImageProcessor:
    COLOR_CHANNEL_COUNT = 3

    def __init__(self, width: int, height: int, window_name: str = 'Tracker'):
        self._hsv_mask = np.zeros((height, width, self.COLOR_CHANNEL_COUNT), dtype=np.uint8)
        self._hsv_mask[..., 1] = 255
        self._window_name = window_name
        self._current_frame = cv2.cvtColor(self._hsv_mask, cv2.COLOR_HSV2BGR)

    @classmethod
    def from_video_capture(cls, capture: cv2.VideoCapture):
        return cls(
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

    @property
    def current_frame(self):
        return self._current_frame

    def display_frame(self, magnitude: np.ndarray, angle: np.ndarray, rep_count: int):
        self._hsv_mask[..., 0] = angle * 180 / np.pi / 2
        self._hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        frame = cv2.cvtColor(self._hsv_mask, cv2.COLOR_HSV2BGR)

        frame = cv2.putText(frame, f"Rep count: {int(rep_count)}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imshow(self._window_name, frame)

        self._current_frame = frame


class CustomVideoWriter(cv2.VideoWriter):
    @classmethod
    def from_video_capture(cls, capture: cv2.VideoCapture, file='untitled.mp4'):
        # For some reason capture FPS is double than it should be
        fps = capture.get(cv2.CAP_PROP_FPS) // 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        resolution = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        return cls(file, fourcc, fps, resolution)

