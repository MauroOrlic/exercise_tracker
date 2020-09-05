from typing import Generator

import numpy as np
from cv2 import (
    VideoCapture,
    cvtColor,
    COLOR_BGR2GRAY,
    calcOpticalFlowFarneback,
    cartToPolar
)


def init_frame(capture: VideoCapture):
    status, frame_raw = capture.read()

    if not status:
        raise IOError('Failed to read first frame from capture.')

    return frame_raw


def generate_flow_from_capture(capture: VideoCapture, magnitude_threshold=2) -> Generator[np.ndarray, None, None]:
    # Get first frame and convert it to grayscale
    frame_current = cvtColor(init_frame(capture), COLOR_BGR2GRAY)

    while True:
        frame_previous = frame_current

        # Get new frame and check if capture is working (detects last frame in a video file).
        status, frame_current = capture.read()
        if not status:
            break
        # Convert new frame to grayscale
        frame_current = cvtColor(frame_current, COLOR_BGR2GRAY)

        # Calculate optical flow (movement) between the previous and current frame
        flow = calcOpticalFlowFarneback(
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
        magnitude, angle = cartToPolar(flow[..., 0], flow[..., 1])

        # Ignores very small movements by setting magnitude to 0 (magnitude is in range 0.0 - 100.0)
        magnitude[magnitude < magnitude_threshold] = 0.0

        yield magnitude, angle
