import numpy as np
from cv2 import (
    cvtColor, COLOR_HSV2BGR, CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT, normalize, NORM_MINMAX,
    VideoCapture, putText, FONT_HERSHEY_SIMPLEX,
    imshow, VideoWriter, CAP_PROP_FPS, VideoWriter_fourcc
)


class DisplayImageProcessor:
    COLOR_CHANNEL_COUNT = 3

    def __init__(self, width: int, height: int, window_name: str = 'Excercise Tracker'):
        self._hsv_mask = np.zeros((height, width, self.COLOR_CHANNEL_COUNT), dtype=np.uint8)
        self._hsv_mask[..., 1] = 255
        self._window_name = window_name
        self._current_frame = cvtColor(self._hsv_mask, COLOR_HSV2BGR)

    @classmethod
    def from_video_capture(cls, capture: VideoCapture):
        return cls(
            int(capture.get(CAP_PROP_FRAME_WIDTH)),
            int(capture.get(CAP_PROP_FRAME_HEIGHT))
        )

    @property
    def current_frame(self):
        return self._current_frame

    def display_frame(self, magnitude: np.ndarray, angle: np.ndarray, rep_count: int):
        self._hsv_mask[..., 0] = angle * 180 / np.pi / 2
        self._hsv_mask[..., 2] = normalize(magnitude, None, 0, 255, NORM_MINMAX)

        frame = putText(
            cvtColor(self._hsv_mask, COLOR_HSV2BGR),
            f"Rep count: {int(rep_count)}",
            (0, 50), FONT_HERSHEY_SIMPLEX, 2, 255
        )

        imshow(self._window_name, frame)

        self._current_frame = frame


class CustomVideoWriter(VideoWriter):

    @classmethod
    def from_video_capture(cls, capture: VideoCapture, file='untitled.mp4'):
        # For some reason capture FPS is double of what it should be
        fps = capture.get(CAP_PROP_FPS) // 2
        fourcc = VideoWriter_fourcc(*'mp4v')
        resolution = (
            int(capture.get(CAP_PROP_FRAME_WIDTH)),
            int(capture.get(CAP_PROP_FRAME_HEIGHT))
        )
        return cls(file, fourcc, fps, resolution)

