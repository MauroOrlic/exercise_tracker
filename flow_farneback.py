import cv2
from typing import Generator
import numpy as np


def init_frame(capture: cv2.VideoCapture):
    status, frame_raw = capture.read()

    if not status:
        raise IOError('Failed to read frame from capture.')

    return frame_raw


def generate_flow_from_capture(capture: cv2.VideoCapture, magnitude_threshold=2) -> Generator[np.ndarray, None, None]:
    frame_current = cv2.cvtColor(init_frame(capture), cv2.COLOR_BGR2GRAY)

    while True:
        frame_previous = frame_current

        status, frame_current = capture.read()
        if not status:
            break
        frame_current = cv2.cvtColor(frame_current, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev=frame_previous,
            next=frame_current,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Ignore very small movements (magnitude is in range 0.0 - 100.0)
        magnitude[magnitude < magnitude_threshold] = 0.0

        yield magnitude, angle
