import numpy as np
from flow_pose import Landmark
from typing import Collection, Tuple
from cv2 import (
    cvtColor, COLOR_HSV2BGR, CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT, normalize, NORM_MINMAX,
    VideoCapture, putText, FONT_HERSHEY_SIMPLEX,
    imshow, VideoWriter, CAP_PROP_FPS, VideoWriter_fourcc,
    circle
)


class DisplayImageProcessor:
    COLOR_CHANNEL_COUNT = 3

    def __init__(self, width: int, height: int, window_name: str = 'Excercise Tracker'):
        self._width = width
        self._height = height
        print(f"({self._width}x{self._height})")
        self._hsv_mask = np.zeros((self._height, self._width, self.COLOR_CHANNEL_COUNT), dtype=np.uint8)
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

    def display_frame(
            self,
            *,
            rep_count: int,
            magnitude: np.ndarray = None,
            angle: np.ndarray = None,
            frame: np.ndarray = None,
            landmarks: Tuple[Landmark] = None
    ):
        if magnitude is not None and angle is not None and frame is None and landmarks is None:
            self._display_frame_optical_flow(rep_count, magnitude, angle)
        elif magnitude is None and angle is None and frame is not None:
            self._display_frame_pose_flow(rep_count, frame, landmarks)

    def _display_frame_optical_flow(self, rep_count: int, magnitude: np.ndarray, angle: np.ndarray):
        self._hsv_mask[..., 0] = angle * 180 / np.pi / 2
        self._hsv_mask[..., 2] = normalize(magnitude, None, 0, 255, NORM_MINMAX)

        frame = putText(
            cvtColor(self._hsv_mask, COLOR_HSV2BGR),
            f"Rep count: {int(rep_count)}",
            (0, 50), FONT_HERSHEY_SIMPLEX, 2, 255
        )

        imshow(self._window_name, frame)

        self._current_frame = frame

    def _display_frame_pose_flow(self, rep_count: int, frame: np.ndarray, landmarks: Collection[Landmark]):
        if landmarks is not None:
            for landmark in landmarks:
                x = max(0, min(
                    self._width,
                    int(landmark.x * self._width)
                ))
                y = max(0, min(
                    self._height,
                    int(landmark.y * self._height)
                ))

                if not (0 <= landmark.x <= 1) or \
                        not (0 <= landmark.y <= 1):
                    marker_color = (0, 0, 255)
                else:
                    marker_color = (0, 255, 0)
                frame = circle(frame, center=(x, y), radius=2, color=marker_color, thickness=3)

        frame = putText(
            frame,
            f"Rep count: {int(rep_count)}",
            (0, 50), FONT_HERSHEY_SIMPLEX, 2, 255
        )

        imshow(self._window_name, frame)

        self._current_frame = frame


class CustomVideoWriter(VideoWriter):

    @classmethod
    def from_video_capture(cls, capture: VideoCapture, file='untitled.mp4'):
        # For some reason capture FPS is double of what it should be
        fps = capture.get(CAP_PROP_FPS)# // 2
        fourcc = VideoWriter_fourcc(*'mp4v')
        resolution = (
            int(capture.get(CAP_PROP_FRAME_WIDTH)),
            int(capture.get(CAP_PROP_FRAME_HEIGHT))
        )
        return cls(file, fourcc, fps, resolution)

