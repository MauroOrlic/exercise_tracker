import cv2
from typing import Generator
import numpy as np

MAGNITUDE_THRESHOLD = 2


def process_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def init_frame(capture: cv2.VideoCapture):
    status, frame_raw = capture.read()

    if not status:
        raise IOError('Failed to read frame from capture.')

    return frame_raw


def generate_flow(capture: cv2.VideoCapture) -> Generator[np.ndarray, None, None]:
    frame_current = process_frame(init_frame(capture))

    while True:
        frame_previous = frame_current

        status, frame_current = capture.read()
        if not status:
            break
        frame_current = process_frame(frame_current)

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

        yield magnitude, angle
